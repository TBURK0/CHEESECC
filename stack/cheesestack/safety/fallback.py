"""Emergency fallback: safety mode manager and minimal-risk maneuver (MRM).

Modes (ordered; higher = more restrictive):

    NOMINAL                full function
    DEGRADED               reduced envelope (speed / accel / steering caps)
    MINIMAL_RISK_MANEUVER  controlled jerk-limited deceleration to standstill
    SAFE_STOP              at standstill, brake held; terminal until explicit reset

Transition rules (`ModeManager`):
  * Escalation (to a more restrictive mode) is immediate on recommendation.
  * MRM -> SAFE_STOP happens automatically once standstill (|v| <= standstill_speed)
    is held for standstill_hold_s.
  * De-escalation is NEVER implicit:
      - DEGRADED -> NOMINAL only via `request_recovery()` with the recommendation
        NOMINAL held for recovery_hold_s, or automatically after the same hold
        only if `allow_auto_recover_degraded=True` (off by default).
      - MRM / SAFE_STOP -> NOMINAL only via `request_recovery(operator_ack=True)`
        at standstill with recommendation NOMINAL held for recovery_hold_s.
      - There is no path from MRM back to DEGRADED; and an aborted MRM never
        resumes motion without an operator.

MRM (`MinimalRiskManeuver`): longitudinal profile with bounded deceleration
`d` and jerk `j`. Acceleration starts at min(a0, 0) (throttle cut), ramps to
-d at rate j, and - if `release_near_stop` - ramps back toward 0 when
v <= a^2 / (2 j), so the vehicle comes to rest without a jerk spike. The same
`step()` is used online (closed loop on measured speed) and for open-loop
planning (`plan()`), which returns a sequence of `core.ControlCommand`.
Brake command = -a / full_brake_decel (simple linear pedal map).

# TODO(safety): lateral part of the MRM (lane keeping / move to shoulder) needs
# the planning layer; here the steering is passed through (validated) or held.
# TODO(safety): calibrated pedal map (brake pressure vs decel), slope compensation,
# hazard lights / horn requests via vehicle layer.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cheesestack.core import ControlCommand, Header, Severity


class SafetyMode(enum.IntEnum):
    NOMINAL = 0
    DEGRADED = 1
    MINIMAL_RISK_MANEUVER = 2
    SAFE_STOP = 3


def severity_to_mode(sev: Severity) -> SafetyMode:
    if sev >= Severity.CRITICAL:
        return SafetyMode.MINIMAL_RISK_MANEUVER
    if sev >= Severity.DEGRADED:
        return SafetyMode.DEGRADED
    return SafetyMode.NOMINAL


@dataclass
class ModeTransition:
    t: float
    from_mode: SafetyMode
    to_mode: SafetyMode
    reason: str


@dataclass
class ModeManagerConfig:
    recovery_hold_s: float = 5.0
    allow_auto_recover_degraded: bool = False
    standstill_speed_mps: float = 0.1
    standstill_hold_s: float = 1.0


class ModeManager:
    def __init__(self, config: Optional[ModeManagerConfig] = None, t0: float = 0.0) -> None:
        self.config = config or ModeManagerConfig()
        self.mode = SafetyMode.NOMINAL
        self.entered_at = t0
        self.history: list[ModeTransition] = []
        self.emergency = False  # FATAL seen during the current MRM
        self._nominal_since: Optional[float] = t0
        self._standstill_since: Optional[float] = None
        self.last_recommendation = SafetyMode.NOMINAL

    def _go(self, to: SafetyMode, t: float, reason: str) -> None:
        if to != self.mode:
            self.history.append(ModeTransition(t, self.mode, to, reason))
            self.mode = to
            self.entered_at = t
            if to <= SafetyMode.DEGRADED:
                self.emergency = False

    def update(self, t: float, recommended: SafetyMode, v: Optional[float] = None,
               emergency: bool = False, reason: str = "") -> SafetyMode:
        """Apply a recommendation. Only escalates (plus MRM->SAFE_STOP at standstill)."""
        c = self.config
        recommended = SafetyMode(recommended)
        self.last_recommendation = recommended
        if recommended == SafetyMode.NOMINAL:
            if self._nominal_since is None:
                self._nominal_since = t
        else:
            self._nominal_since = None

        if recommended > self.mode:
            self._go(recommended, t, reason or f"escalate to {recommended.name}")
        if emergency and self.mode >= SafetyMode.MINIMAL_RISK_MANEUVER:
            self.emergency = True

        if v is not None and math.isfinite(v) and abs(v) <= c.standstill_speed_mps:
            if self._standstill_since is None:
                self._standstill_since = t
        else:
            self._standstill_since = None

        if (self.mode == SafetyMode.MINIMAL_RISK_MANEUVER and self._standstill_since is not None
                and t - self._standstill_since >= c.standstill_hold_s):
            self._go(SafetyMode.SAFE_STOP, t, "standstill reached")

        if (self.mode == SafetyMode.DEGRADED and c.allow_auto_recover_degraded and self._nominal_held(t)):
            self._go(SafetyMode.NOMINAL, t, "auto recovery from DEGRADED")
        return self.mode

    def _nominal_held(self, t: float) -> bool:
        return self._nominal_since is not None and t - self._nominal_since >= self.config.recovery_hold_s

    def at_standstill(self, t: float) -> bool:
        return self._standstill_since is not None and t - self._standstill_since >= self.config.standstill_hold_s

    def request_recovery(self, t: float, operator_ack: bool = False) -> tuple[bool, str]:
        """Explicit request to return to NOMINAL. Returns (granted, reason)."""
        if self.mode == SafetyMode.NOMINAL:
            return True, "already nominal"
        if not self._nominal_held(t):
            return False, f"recommendation not NOMINAL for {self.config.recovery_hold_s}s"
        if self.mode == SafetyMode.DEGRADED:
            self._go(SafetyMode.NOMINAL, t, "explicit recovery from DEGRADED")
            return True, "recovered"
        if not operator_ack:
            return False, "operator acknowledgement required"
        if not self.at_standstill(t):
            return False, "vehicle not at standstill"
        self._go(SafetyMode.NOMINAL, t, "operator reset after stop")
        return True, "recovered"


# --------------------------------------------------------------------------- MRM

@dataclass
class MRMConfig:
    decel_mps2: float = 3.0
    jerk_mps3: float = 2.5
    emergency_decel_mps2: float = 6.0
    emergency_jerk_mps3: float = 10.0
    release_near_stop: bool = True
    min_release_decel_mps2: float = 0.2  # decel floor kept during release until stop
    stop_speed_mps: float = 0.05
    full_brake_decel_mps2: float = 8.0  # decel produced by brake=1.0
    hold_brake: float = 0.4  # brake held at standstill
    plan_dt_s: float = 0.02
    max_plan_s: float = 60.0
    unknown_speed_mps: float = 40.0  # assumed speed when v0 is not finite
    max_dt_s: float = 0.2  # clamp step dt (missed cycles must not cause jerk spikes)


def stopping_distance(v0: float, decel: float, jerk: float) -> float:
    """Distance to stop with a jerk-limited ramp 0 -> -decel then constant decel
    (no release phase). Closed form; used for envelope checks."""
    v0 = max(0.0, v0)
    t_r = decel / jerk
    v_ramp = 0.5 * jerk * t_r ** 2  # speed lost during ramp
    if v0 <= v_ramp:
        t = math.sqrt(2 * v0 / jerk)
        return v0 * t - jerk * t ** 3 / 6.0
    s_ramp = v0 * t_r - jerk * t_r ** 3 / 6.0
    v1 = v0 - v_ramp
    return s_ramp + v1 ** 2 / (2 * decel)


class MinimalRiskManeuver:
    def __init__(self, config: Optional[MRMConfig] = None, source: str = "safety.mrm") -> None:
        self.config = config or MRMConfig()
        self.source = source
        self.active = False
        self.stopped = False
        self.emergency = False
        self.a = 0.0
        self.v = 0.0
        self.t_last = 0.0
        self.t_start = 0.0
        self._releasing = False
        self._seq = 0

    @property
    def decel(self) -> float:
        return self.config.emergency_decel_mps2 if self.emergency else self.config.decel_mps2

    @property
    def jerk(self) -> float:
        return self.config.emergency_jerk_mps3 if self.emergency else self.config.jerk_mps3

    def start(self, t: float, v0: float, a0: float = 0.0, emergency: bool = False) -> None:
        self.active, self.stopped, self.emergency = True, False, emergency
        # Unknown speed: assume moving (conservative) so the full braking profile runs.
        self.v = max(0.0, float(v0)) if math.isfinite(v0) else self.config.unknown_speed_mps
        self.a = min(float(a0), 0.0) if math.isfinite(a0) else 0.0
        self.t_last = self.t_start = float(t)
        self._releasing = False
        if self.v <= self.config.stop_speed_mps:
            self.stopped, self.v, self.a = True, 0.0, 0.0

    def escalate_emergency(self) -> None:
        """Switch an active MRM to the emergency profile (never the other way)."""
        self.emergency = True

    def step(self, t: float, v_measured: Optional[float] = None, steer_wheel_deg: float = 0.0) -> ControlCommand:
        if not self.active:
            raise RuntimeError("MRM not started")
        c = self.config
        dt = min(max(0.0, float(t) - self.t_last), c.max_dt_s)
        self.t_last = float(t)
        if not self.stopped:
            d, j = self.decel, self.jerk
            # Speed: prefer measurement (closed loop), else integrate the profile.
            v_meas_ok = v_measured is not None and math.isfinite(v_measured)
            v_now = max(0.0, float(v_measured)) if v_meas_ok else max(0.0, self.v + self.a * dt)
            if c.release_near_stop and not self._releasing and self.a < 0 and v_now <= self.a ** 2 / (2 * j):
                self._releasing = True  # latched: no chatter between ramp-down and release
            if self._releasing:
                # Ramp decel back toward a small floor so the car cannot end up creeping at a=0.
                floor = -min(c.min_release_decel_mps2, d)
                a_new = min(self.a + j * dt, floor) if self.a < floor else self.a
            elif self.a > -d:
                a_new = max(self.a - j * dt, -d)
            else:
                a_new = min(self.a + j * dt, -d)  # started beyond -d
            self.a = a_new
            self.v = v_now
            if self.v <= c.stop_speed_mps:
                self.stopped, self.v = True, 0.0
        self._seq += 1
        if self.stopped:
            brake = max(c.hold_brake, min(1.0, -self.a / c.full_brake_decel_mps2))
            self.a = 0.0
        else:
            brake = min(1.0, max(0.0, -self.a / c.full_brake_decel_mps2))
        return ControlCommand(header=Header(stamp=float(t), seq=self._seq, source=self.source),
                              steer_wheel_deg=float(steer_wheel_deg), throttle=0.0, brake=brake,
                              curvature=0.0, target_speed_mps=self.v, source=self.source)

    def plan(self, t0: float, v0: float, a0: float = 0.0, emergency: bool = False,
             steer_wheel_deg: float = 0.0) -> tuple[list[ControlCommand], np.ndarray]:
        """Open-loop MRM. Returns (commands, profile) with profile rows (t, v, a)
        ending at the first standstill command."""
        self.start(t0, v0, a0, emergency)
        c = self.config
        cmds, rows = [], []
        n_max = int(math.ceil(c.max_plan_s / c.plan_dt_s))
        for k in range(n_max + 1):
            t = t0 + k * c.plan_dt_s
            if k == 0:
                a_rec = self.a
                cmd = ControlCommand(header=Header(stamp=t, seq=0, source=self.source), steer_wheel_deg=steer_wheel_deg,
                                     brake=min(1.0, max(0.0, -self.a / c.full_brake_decel_mps2)) if not self.stopped
                                     else c.hold_brake, target_speed_mps=self.v, source=self.source)
            else:
                cmd = self.step(t, None, steer_wheel_deg)
                a_rec = self.a
            cmds.append(cmd)
            rows.append((t, self.v, a_rec))
            if self.stopped:
                break
        return cmds, np.array(rows)
