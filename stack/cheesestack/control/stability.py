"""Stability limiting: keep curvature requests inside the friction / comfort envelope.

    |kappa| <= a_lat_max / v^2        (lateral acceleration)
    |kappa| <= r_max / v              (yaw rate)
    |kappa| <= kappa_mech             (full lock)
Also `max_speed_for_curvature` (inverse, used for curve speed planning; the app's
LongitudinalModel uses v = sqrt(comfort / |kappa|) with comfort ~ 2.2..4.2 m/s^2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .vehicle_params import VehicleParams


@dataclass
class StabilityConfig:
    max_lateral_accel: float = 3.0  # m/s^2 (comfort; friction limit ~ 0.8 g)
    max_yaw_rate: float = 0.6  # rad/s
    min_speed_mps: float = 0.5  # below this only the mechanical limit applies


@dataclass
class StabilityResult:
    curvature: float
    limit: float
    limited_by: str = ""  # "", "lateral_accel", "yaw_rate", "mechanical"


class StabilityLimiter:
    def __init__(self, params: VehicleParams | None = None, config: StabilityConfig | None = None) -> None:
        self.params = params or VehicleParams()
        self.config = config or StabilityConfig()

    def max_curvature(self, v: float) -> tuple[float, str]:
        c = self.config
        limits = {"mechanical": self.params.max_curvature}
        va = abs(v)
        if va > c.min_speed_mps:
            limits["lateral_accel"] = c.max_lateral_accel / (va * va)
            limits["yaw_rate"] = c.max_yaw_rate / va
        name = min(limits, key=limits.get)
        return limits[name], name

    def limit(self, kappa: float, v: float) -> StabilityResult:
        lim, name = self.max_curvature(v)
        if abs(kappa) > lim:
            return StabilityResult(math.copysign(lim, kappa), lim, name)
        return StabilityResult(kappa, lim, "")

    def max_speed_for_curvature(self, kappa: float, v_cap: float = 40.0) -> float:
        k = abs(kappa)
        if k < 1e-6:
            return v_cap
        c = self.config
        return min(v_cap, math.sqrt(c.max_lateral_accel / k), c.max_yaw_rate / k)

    # TODO(control): friction-estimate-aware limits (wet road), banked/graded roads,
    # and a sideslip-based stability check using measured yaw rate vs v*kappa.
