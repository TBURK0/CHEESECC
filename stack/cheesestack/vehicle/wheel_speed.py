"""Wheel-speed validation: cross-check wheels against each other and against GPS.

Per tick:
1. Convert wheel angular speeds to linear speed (omega * R * scale).
2. Remove turning geometry: each wheel's speed is divided by its kinematic
   ratio v_wheel / v_cg (from yaw rate, track, steer) so all four should agree.
3. Robust reference = median of the compensated speeds; a wheel whose
   residual exceeds max(abs_tol, rel_tol * ref) is an outlier for this tick.
4. Optionally compare the reference against GPS ground speed (when the GPS
   accuracy is good and speed is above `gps_min_speed`).
Outliers and GPS mismatch are debounced (N consecutive bad ticks) before they
become `Fault`s, and a slow EMA of gps/wheel speed estimates the effective
tire-radius scale factor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from cheesestack.core import Fault, Severity

from .params import VehicleParams
from .slip import WHEELS, wheel_ground_speeds


@dataclass
class WheelSpeedConfig:
    abs_tol: float = 0.5  # m/s
    rel_tol: float = 0.05
    gps_abs_tol: float = 1.0  # m/s
    gps_rel_tol: float = 0.08
    gps_min_speed: float = 3.0  # m/s
    gps_max_accuracy: float = 1.0  # m/s; ignore GPS speed with worse 1-sigma
    debounce_ticks: int = 5
    scale_ema_alpha: float = 0.01
    scale_bounds: tuple = (0.9, 1.1)
    source: str = "vehicle.wheel_speed"


@dataclass
class WheelSpeedReport:
    stamp: float
    wheel_speeds: dict  # m/s, raw linear
    compensated: dict  # m/s, geometry-removed
    reference_speed: float  # m/s, median of valid wheels
    residuals: dict
    outliers: tuple  # this tick
    faulted_wheels: tuple  # debounced
    gps_speed: Optional[float] = None
    gps_consistent: Optional[bool] = None
    scale_factor: float = 1.0
    valid: bool = True
    faults: list = field(default_factory=list)


class WheelSpeedValidator:
    def __init__(self, params: VehicleParams, config: Optional[WheelSpeedConfig] = None) -> None:
        self.params = params
        self.config = config or WheelSpeedConfig()
        self.reset()

    def reset(self) -> None:
        self._bad = {w: 0 for w in WHEELS}
        self._gps_bad = 0
        self.scale_factor = 1.0

    def geometry_ratios(self, v_ref: float, yaw_rate: float, road_wheel: float) -> dict[str, float]:
        if abs(v_ref) < 0.5:
            return {w: 1.0 for w in WHEELS}
        p = self.params
        beta = math.atan(p.lr * math.tan(road_wheel) / p.wheelbase)
        g = wheel_ground_speeds(p, v_ref, v_ref * math.tan(beta), yaw_rate, road_wheel)
        return {w: (g[w] / v_ref if abs(g[w] / v_ref) > 0.2 else 1.0) for w in WHEELS}

    def update(self, t: float, wheel_omega: Sequence[float], yaw_rate: float = 0.0, road_wheel: float = 0.0,
               gps_speed: Optional[float] = None, gps_speed_accuracy: float = 0.5) -> WheelSpeedReport:
        c = self.config
        if len(wheel_omega) != 4:
            raise ValueError("wheel_omega must have 4 entries (fl, fr, rl, rr)")
        R = self.params.tire_radius * self.scale_factor
        raw = {w: float(om) * R for w, om in zip(WHEELS, wheel_omega)}
        v0 = float(np.median(list(raw.values())))
        ratios = self.geometry_ratios(v0, yaw_rate, road_wheel)
        comp = {w: raw[w] / ratios[w] for w in WHEELS}
        ref = float(np.median(list(comp.values())))
        tol = max(c.abs_tol, c.rel_tol * abs(ref))
        resid = {w: comp[w] - ref for w in WHEELS}
        outliers = tuple(w for w in WHEELS if abs(resid[w]) > tol)
        good = [comp[w] for w in WHEELS if w not in outliers]
        if good:
            ref = float(np.mean(good))

        faults: list[Fault] = []
        for w in WHEELS:
            self._bad[w] = self._bad[w] + 1 if w in outliers else 0
        faulted = tuple(w for w in WHEELS if self._bad[w] >= c.debounce_ticks)
        for w in faulted:
            faults.append(Fault(f"WSS_{w.upper()}_IMPLAUSIBLE", Severity.DEGRADED, c.source, t,
                                f"wheel {w} disagrees with others by {resid[w]:.2f} m/s",
                                {"residual": resid[w]}))

        gps_ok: Optional[bool] = None
        if gps_speed is not None and gps_speed_accuracy <= c.gps_max_accuracy and abs(gps_speed) >= c.gps_min_speed:
            gtol = max(c.gps_abs_tol, c.gps_rel_tol * abs(gps_speed))
            gps_ok = abs(ref - gps_speed) <= gtol
            self._gps_bad = 0 if gps_ok else self._gps_bad + 1
            if gps_ok and len(good) >= 3 and abs(ref) > 0:
                # Tire-radius scale estimate (only from agreeing data to stay robust).
                target = self.scale_factor * gps_speed / ref
                self.scale_factor += c.scale_ema_alpha * (target - self.scale_factor)
                self.scale_factor = float(np.clip(self.scale_factor, *c.scale_bounds))
            if self._gps_bad >= c.debounce_ticks:
                faults.append(Fault("WSS_GPS_MISMATCH", Severity.WARNING, c.source, t,
                                    f"wheel ref {ref:.2f} vs gps {gps_speed:.2f} m/s",
                                    {"wheel": ref, "gps": gps_speed}))
        valid = len(faulted) <= 1 and (self._gps_bad < c.debounce_ticks)
        if len(faulted) >= 2:
            faults.append(Fault("WSS_MULTIPLE_FAILED", Severity.CRITICAL, c.source, t, "two or more wheel speeds implausible"))
        # TODO(vehicle): direction (sign) detection, wheel-lock/spin vs sensor-fault discrimination via IMU accel.
        return WheelSpeedReport(t, raw, comp, ref, resid, outliers, faulted, gps_speed, gps_ok,
                                self.scale_factor, valid, faults)
