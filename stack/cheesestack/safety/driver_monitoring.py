"""Driver-monitoring integration and takeover-request (TOR) escalation.

Inputs: `DriverState` samples from a driver-monitoring system (camera gaze /
head pose, steering-wheel torque or capacitive hands-on sensor). The safety
layer does not estimate attention itself; it consumes these signals.

`DriverMonitor.update(t)` classifies the driver as AVAILABLE, DISTRACTED or
UNAVAILABLE (UNKNOWN when the input is stale - treated as unavailable):

  * attentive  = eyes_on_road and attention >= min_attention
  * distracted continuously > distraction_s, or hands off > hands_off_s -> DISTRACTED
  * distracted continuously > unavailable_s                            -> UNAVAILABLE
  * input older than max_input_age_s                                   -> UNKNOWN (DRV_MONITOR_LOST)

`TakeoverManager` runs the TOR timeline once a takeover is requested (by the
supervisor because the system is degraded, or because the driver is
distracted while supervision is required):

    t_req + 0            VISUAL     (icon / message)
    t_req + audible_s    AUDIBLE    (chime)
    t_req + haptic_s     HAPTIC     (seat/wheel vibration, brake jerk)
    t_req + mrm_s        MRM        (driver unresponsive -> DRV_UNRESPONSIVE, CRITICAL)

The TOR is resolved (TAKEN_OVER) only when the driver has been continuously
ready (attentive AND hands on) for `confirm_s`. Once MRM is reached the TOR
stays in MRM; the MRM is not cancelled by a late driver response here -
handing control back mid-MRM is a vehicle-level decision (# TODO(safety)).

# TODO(safety): stage times per ODD/speed (UN R157 style), driver override
# detection via steering torque / pedal input, eyes-off-road glance statistics.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Optional

from cheesestack.core import Fault

from .faults import DEFAULT_REGISTRY, FaultRegistry


@dataclass
class DriverState:
    stamp: float
    attention: float = 1.0  # [0, 1] from DMS
    eyes_on_road: bool = True
    hands_on: bool = True


class DriverAvailability(enum.IntEnum):
    AVAILABLE = 0
    DISTRACTED = 1
    UNAVAILABLE = 2
    UNKNOWN = 3


@dataclass
class DriverMonitorConfig:
    min_attention: float = 0.5
    distraction_s: float = 2.0
    hands_off_s: float = 10.0
    unavailable_s: float = 6.0
    max_input_age_s: float = 0.5


class DriverMonitor:
    def __init__(self, config: Optional[DriverMonitorConfig] = None, registry: Optional[FaultRegistry] = None,
                 source: str = "safety.driver") -> None:
        self.config = config or DriverMonitorConfig()
        self.registry = registry or DEFAULT_REGISTRY
        self.source = source
        self.last: Optional[DriverState] = None
        self._distracted_since: Optional[float] = None
        self._hands_off_since: Optional[float] = None
        self._ready_since: Optional[float] = None

    def feed(self, s: DriverState) -> None:
        if self.last is not None and s.stamp < self.last.stamp:
            return  # ignore out-of-order samples
        attentive = s.eyes_on_road and math.isfinite(s.attention) and s.attention >= self.config.min_attention
        if attentive:
            self._distracted_since = None
        elif self._distracted_since is None:
            self._distracted_since = s.stamp
        if s.hands_on:
            self._hands_off_since = None
        elif self._hands_off_since is None:
            self._hands_off_since = s.stamp
        if attentive and s.hands_on:
            if self._ready_since is None:
                self._ready_since = s.stamp
        else:
            self._ready_since = None
        self.last = s

    def availability(self, t: float) -> DriverAvailability:
        c = self.config
        if self.last is None or t - self.last.stamp > c.max_input_age_s:
            return DriverAvailability.UNKNOWN
        dist = t - self._distracted_since if self._distracted_since is not None else 0.0
        hoff = t - self._hands_off_since if self._hands_off_since is not None else 0.0
        if dist > c.unavailable_s:
            return DriverAvailability.UNAVAILABLE
        if dist > c.distraction_s or hoff > c.hands_off_s:
            return DriverAvailability.DISTRACTED
        return DriverAvailability.AVAILABLE

    def ready_for(self, t: float) -> float:
        """Seconds the driver has been continuously attentive with hands on (0 if stale/not)."""
        if self.last is None or t - self.last.stamp > self.config.max_input_age_s or self._ready_since is None:
            return 0.0
        return t - self._ready_since

    def check(self, t: float) -> list[Fault]:
        av = self.availability(t)
        out = []
        if av == DriverAvailability.UNKNOWN:
            out.append(self.registry.make("DRV_MONITOR_LOST", self.source, t, "driver monitoring input stale"))
        elif av in (DriverAvailability.DISTRACTED, DriverAvailability.UNAVAILABLE):
            if self._distracted_since is not None and t - self._distracted_since > self.config.distraction_s:
                out.append(self.registry.make("DRV_INATTENTIVE", self.source, t,
                                              f"driver inattentive {t - self._distracted_since:.1f}s"))
            if self._hands_off_since is not None and t - self._hands_off_since > self.config.hands_off_s:
                out.append(self.registry.make("DRV_HANDS_OFF", self.source, t,
                                              f"hands off {t - self._hands_off_since:.1f}s"))
        return out

    def health(self, t: float) -> float:
        return {DriverAvailability.AVAILABLE: 1.0, DriverAvailability.DISTRACTED: 0.5,
                DriverAvailability.UNAVAILABLE: 0.0, DriverAvailability.UNKNOWN: 0.25}[self.availability(t)]


class TORStage(enum.IntEnum):
    NONE = 0
    VISUAL = 1
    AUDIBLE = 2
    HAPTIC = 3
    MRM = 4
    TAKEN_OVER = 5  # resolved; driver in control (stage value is not an escalation level)


@dataclass
class TakeoverConfig:
    audible_s: float = 4.0
    haptic_s: float = 7.0
    mrm_s: float = 10.0
    confirm_s: float = 1.0


class TakeoverManager:
    def __init__(self, config: Optional[TakeoverConfig] = None, registry: Optional[FaultRegistry] = None,
                 source: str = "safety.tor") -> None:
        c = config or TakeoverConfig()
        if not (0 < c.audible_s <= c.haptic_s <= c.mrm_s):
            raise ValueError("TOR stage times must satisfy 0 < audible <= haptic <= mrm")
        self.config = c
        self.registry = registry or DEFAULT_REGISTRY
        self.source = source
        self.stage = TORStage.NONE
        self.requested_at: Optional[float] = None
        self.reason = ""

    @property
    def pending(self) -> bool:
        return self.stage in (TORStage.VISUAL, TORStage.AUDIBLE, TORStage.HAPTIC)

    def request(self, t: float, reason: str = "") -> None:
        """Start a TOR. Re-requesting while one is pending keeps the original start time."""
        if self.pending or self.stage == TORStage.MRM:
            return
        self.requested_at, self.reason, self.stage = t, reason, TORStage.VISUAL

    def cancel(self) -> None:
        """System no longer needs the takeover (e.g. condition cleared before MRM)."""
        if self.stage != TORStage.MRM:
            self.stage, self.requested_at = TORStage.NONE, None

    def reset(self) -> None:
        """Explicit reset (after a completed stop / operator ack)."""
        self.stage, self.requested_at, self.reason = TORStage.NONE, None, ""

    def update(self, t: float, driver_ready_s: float) -> TORStage:
        """Advance the timeline. `driver_ready_s` = DriverMonitor.ready_for(t)."""
        if not self.pending:
            return self.stage
        c = self.config
        if driver_ready_s >= c.confirm_s:
            self.stage = TORStage.TAKEN_OVER
            return self.stage
        el = t - self.requested_at
        if el >= c.mrm_s:
            self.stage = TORStage.MRM
        elif el >= c.haptic_s:
            self.stage = TORStage.HAPTIC
        elif el >= c.audible_s:
            self.stage = TORStage.AUDIBLE
        else:
            self.stage = TORStage.VISUAL
        return self.stage

    def check(self, t: float) -> list[Fault]:
        if self.stage == TORStage.MRM:
            return [self.registry.make("DRV_UNRESPONSIVE", self.source, t,
                                       f"no takeover within {self.config.mrm_s}s ({self.reason})")]
        return []
