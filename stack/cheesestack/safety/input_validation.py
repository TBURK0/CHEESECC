"""Validation and clamping of `ControlCommand`, `Trajectory` and `VehicleState`.

Each validator returns a `ValidationResult(ok, faults, value)`:
  * `ok`    False when the input is structurally unusable (non-finite fields,
            non-monotonic time, empty trajectory, ...). The caller must not use it.
  * `value` a sanitized copy (clamped to limits) when ok, else None.
  * clamping alone keeps `ok=True` but emits a WARNING-level fault
    (CMD_CLAMPED / TRAJ_LIMIT) so repeated clamping shows up in health.

Control command rules (conservative):
  * any non-finite field -> invalid (CMD_INVALID)
  * throttle, brake clamped to [0, 1]; throttle and brake both > deadband -> brake wins, throttle = 0
  * |steer_wheel_deg| <= max_steer_wheel_deg; steering rate limited vs previous command
  * |curvature| <= max_curvature; 0 <= target_speed <= max_speed
  * command stamp older than max_cmd_age_s (when `t` given) -> CMD_STALE, invalid

# TODO(safety): load limits from the vehicle calibration (vehicle layer params)
# via the integration pipeline; speed-dependent steering limits (lateral accel).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Optional

import numpy as np

from cheesestack.core import ControlCommand, Fault, Header, Trajectory, TrajectoryPoint, VehicleState

from .faults import DEFAULT_REGISTRY, FaultRegistry


@dataclass
class ControlLimits:
    max_steer_wheel_deg: float = 450.0
    max_steer_rate_deg_s: float = 360.0
    max_curvature: float = 0.2  # 1/m
    max_speed_mps: float = 40.0
    pedal_deadband: float = 0.02
    max_cmd_age_s: float = 0.25


@dataclass
class TrajectoryLimits:
    max_speed_mps: float = 40.0
    max_accel: float = 3.0
    max_decel: float = 8.0
    max_curvature: float = 0.25
    max_lat_accel: float = 4.0
    max_dt_s: float = 1.0  # max gap between consecutive points
    max_start_offset_s: float = 0.5  # first point t must be within [-x, x]
    continuity_slack_m: float = 1.0  # |ds - v*dt| tolerance
    max_points: int = 10000


@dataclass
class StateLimits:
    max_speed_mps: float = 70.0
    min_speed_mps: float = -5.0  # small reverse allowed
    max_abs_accel: float = 15.0
    max_abs_yaw_rate: float = 2.0
    max_abs_steer_rad: float = 1.0
    max_abs_curvature: float = 0.5
    max_future_s: float = 0.05
    max_age_s: float = 0.5


@dataclass
class ValidationResult:
    ok: bool
    faults: list[Fault] = field(default_factory=list)
    value: Any = None

    @property
    def codes(self) -> list[str]:
        return [f.code for f in self.faults]


def _finite(*xs: float) -> bool:
    try:
        return all(math.isfinite(float(x)) for x in xs)
    except (TypeError, ValueError):
        return False


class InputValidator:
    def __init__(self, control: Optional[ControlLimits] = None, trajectory: Optional[TrajectoryLimits] = None,
                 state: Optional[StateLimits] = None, registry: Optional[FaultRegistry] = None,
                 source: str = "safety.input") -> None:
        self.control = control or ControlLimits()
        self.trajectory = trajectory or TrajectoryLimits()
        self.state = state or StateLimits()
        self.registry = registry or DEFAULT_REGISTRY
        self.source = source

    def _f(self, code: str, t: float, msg: str, **data) -> Fault:
        return self.registry.make(code, self.source, t, msg, **data)

    # ------------------------------------------------------------------ control
    def validate_command(self, cmd: ControlCommand, prev: Optional[ControlCommand] = None,
                         t: Optional[float] = None) -> ValidationResult:
        L = self.control
        now = cmd.header.stamp if t is None or not _finite(t) else float(t)
        fields_ = (cmd.steer_wheel_deg, cmd.throttle, cmd.brake, cmd.curvature, cmd.target_speed_mps, cmd.header.stamp)
        if not _finite(*fields_):
            return ValidationResult(False, [self._f("CMD_INVALID", now, "non-finite command field",
                                                    cmd_source=cmd.source)])
        if t is not None and now - cmd.header.stamp > L.max_cmd_age_s:
            return ValidationResult(False, [self._f("CMD_STALE", now,
                                                    f"command age {now - cmd.header.stamp:.3f}s > {L.max_cmd_age_s}s")])
        notes: list[str] = []
        throttle = min(max(float(cmd.throttle), 0.0), 1.0)
        brake = min(max(float(cmd.brake), 0.0), 1.0)
        if throttle != cmd.throttle or brake != cmd.brake:
            notes.append("pedal range")
        if throttle > L.pedal_deadband and brake > L.pedal_deadband:
            throttle = 0.0
            notes.append("throttle+brake conflict: brake wins")
        steer = min(max(float(cmd.steer_wheel_deg), -L.max_steer_wheel_deg), L.max_steer_wheel_deg)
        if steer != cmd.steer_wheel_deg:
            notes.append("steer range")
        if prev is not None and _finite(prev.steer_wheel_deg, prev.header.stamp):
            dt = cmd.header.stamp - prev.header.stamp
            if dt > 0:
                max_d = L.max_steer_rate_deg_s * dt
                lim = min(max(steer, prev.steer_wheel_deg - max_d), prev.steer_wheel_deg + max_d)
                if lim != steer:
                    notes.append("steer rate")
                steer = lim
        curv = min(max(float(cmd.curvature), -L.max_curvature), L.max_curvature)
        if curv != cmd.curvature:
            notes.append("curvature range")
        vs = min(max(float(cmd.target_speed_mps), 0.0), L.max_speed_mps)
        if vs != cmd.target_speed_mps:
            notes.append("target speed range")
        out = replace(cmd, steer_wheel_deg=steer, throttle=throttle, brake=brake, curvature=curv,
                      target_speed_mps=vs)
        faults = [self._f("CMD_CLAMPED", now, "; ".join(notes), notes=notes)] if notes else []
        return ValidationResult(True, faults, out)

    # ------------------------------------------------------------------ trajectory
    def validate_trajectory(self, traj: Trajectory) -> ValidationResult:
        L = self.trajectory
        now = traj.header.stamp if _finite(traj.header.stamp) else 0.0
        bad = lambda msg, **d: ValidationResult(False, [self._f("TRAJ_INVALID", now, msg, **d)])
        n = len(traj.points)
        if n == 0:
            return bad("empty trajectory")
        if n > L.max_points:
            return bad(f"{n} points > {L.max_points}")
        arr = traj.as_array()
        if not np.all(np.isfinite(arr)) or not _finite(traj.header.stamp):
            return bad("non-finite trajectory values")
        t, x, y, yaw, v, a, k = arr.T
        if abs(t[0]) > L.max_start_offset_s:
            return bad(f"first point t={t[0]:.3f}s not near 0")
        if n > 1:
            dt = np.diff(t)
            if np.any(dt <= 0):
                return bad("trajectory time not strictly increasing")
            if np.any(dt > L.max_dt_s):
                return bad(f"trajectory time gap {dt.max():.3f}s")
            ds = np.hypot(np.diff(x), np.diff(y))
            v_mid = 0.5 * (np.abs(v[1:]) + np.abs(v[:-1]))
            err = np.abs(ds - v_mid * dt)
            if np.any(err > L.continuity_slack_m + 0.1 * ds):
                i = int(np.argmax(err))
                return bad(f"spatial discontinuity at point {i}: ds={ds[i]:.2f}m vs v*dt={v_mid[i] * dt[i]:.2f}m",
                           index=i)
        notes = []
        v_c = np.clip(v, 0.0, L.max_speed_mps)
        if np.any(v_c != v):
            notes.append("speed")
        a_c = np.clip(a, -L.max_decel, L.max_accel)
        if np.any(a_c != a):
            notes.append("accel")
        k_c = np.clip(k, -L.max_curvature, L.max_curvature)
        if np.any(k_c != k):
            notes.append("curvature")
        # Lateral accel: reduce speed where v^2 |k| exceeds the limit.
        k_abs = np.abs(k_c)
        v_lat = np.where(k_abs > 1e-9, np.sqrt(L.max_lat_accel / np.maximum(k_abs, 1e-9)), np.inf)
        if np.any(v_c > v_lat):
            notes.append("lateral accel")
            v_c = np.minimum(v_c, v_lat)
        pts = [TrajectoryPoint(float(ti), float(xi), float(yi), float(yawi), float(vi), float(ai), float(ki))
               for ti, xi, yi, yawi, vi, ai, ki in zip(t, x, y, yaw, v_c, a_c, k_c)]
        out = Trajectory(traj.header, pts)
        faults = [self._f("TRAJ_LIMIT", now, "clamped: " + ", ".join(notes), notes=notes)] if notes else []
        return ValidationResult(True, faults, out)

    # ------------------------------------------------------------------ state
    def validate_state(self, st: VehicleState, t: Optional[float] = None) -> ValidationResult:
        L = self.state
        now = float(t) if t is not None and _finite(t) else (st.header.stamp if _finite(st.header.stamp) else 0.0)
        bad = lambda msg: ValidationResult(False, [self._f("STATE_INVALID", now, msg)])
        vals = (st.header.stamp, st.pose.x, st.pose.y, st.pose.yaw, st.v, st.a, st.yaw_rate, st.steer, st.curvature)
        if not _finite(*vals):
            return bad("non-finite state field")
        if t is not None:
            if st.header.stamp > now + L.max_future_s:
                return bad(f"state stamp {st.header.stamp:.3f} in the future (now {now:.3f})")
            if now - st.header.stamp > L.max_age_s:
                return bad(f"state age {now - st.header.stamp:.3f}s > {L.max_age_s}s")
        checks = (
            (L.min_speed_mps <= st.v <= L.max_speed_mps, f"speed {st.v}"),
            (abs(st.a) <= L.max_abs_accel, f"accel {st.a}"),
            (abs(st.yaw_rate) <= L.max_abs_yaw_rate, f"yaw rate {st.yaw_rate}"),
            (abs(st.steer) <= L.max_abs_steer_rad, f"steer {st.steer}"),
            (abs(st.curvature) <= L.max_abs_curvature, f"curvature {st.curvature}"),
        )
        for ok, msg in checks:
            if not ok:
                return bad(f"implausible {msg}")
        if st.covariance is not None:
            c = np.asarray(st.covariance, dtype=float)
            if c.ndim != 2 or c.shape[0] != c.shape[1] or not np.all(np.isfinite(c)):
                return bad("covariance malformed")
            if not np.allclose(c, c.T, atol=1e-9 * max(1.0, float(np.abs(c).max()))):
                return bad("covariance not symmetric")
            if np.linalg.eigvalsh(0.5 * (c + c.T)).min() < -1e-9:
                return bad("covariance not positive semi-definite")
        return ValidationResult(True, [], st)
