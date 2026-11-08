"""Steering state estimation: handwheel <-> road-wheel, rate, backlash/deadband.

Model: road_wheel = backlash(handwheel - offset) / steering_ratio, where
`backlash` is a play (hysteresis) operator of total width `backlash_rad`
(handwheel rad): the output only follows once the input has moved through the
play. A small symmetric deadband around center suppresses sensor jitter.
Rates are a finite difference low-passed by a first-order filter.

Limits: constant steering ratio (no variable-ratio racks), no compliance or
Ackermann per-wheel split (the bicycle "road-wheel" angle is the mean).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .params import VehicleParams


@dataclass
class SteeringConfig:
    backlash_rad: float = math.radians(1.0)  # total handwheel play
    deadband_rad: float = math.radians(0.3)  # handwheel jitter suppression around center
    offset_rad: float = 0.0  # handwheel zero offset (calibration)
    rate_tau_s: float = 0.05  # low-pass time constant for rate estimate
    clamp_to_lock: bool = True


@dataclass
class SteeringState:
    stamp: float
    handwheel: float  # rad, measured (offset removed)
    handwheel_effective: float  # rad, after backlash/deadband
    road_wheel: float  # rad
    handwheel_rate: float  # rad/s
    road_wheel_rate: float  # rad/s
    curvature: float  # 1/m, kinematic tan(delta)/L
    saturated: bool = False


def handwheel_to_road_wheel(handwheel: float, params: VehicleParams) -> float:
    return handwheel / params.steering_ratio


def road_wheel_to_handwheel(road_wheel: float, params: VehicleParams) -> float:
    return road_wheel * params.steering_ratio


def curvature_to_road_wheel(curvature: float, params: VehicleParams) -> float:
    """Kinematic inverse: delta = atan(kappa * L) (matches BicycleModel.swift)."""
    return math.atan(curvature * params.wheelbase)


def road_wheel_to_curvature(road_wheel: float, params: VehicleParams) -> float:
    return math.tan(road_wheel) / params.wheelbase


def curvature_to_handwheel_deg(curvature: float, params: VehicleParams, clamp: bool = True) -> float:
    """Handwheel degrees for a path curvature, clamped to +/- maxSteerWheelDeg."""
    hw = math.degrees(road_wheel_to_handwheel(curvature_to_road_wheel(curvature, params), params))
    if clamp:
        hw = max(-params.max_steer_wheel_deg, min(params.max_steer_wheel_deg, hw))
    return hw


class SteeringEstimator:
    """Stateful handwheel -> road-wheel estimator with backlash and rate filtering."""

    def __init__(self, params: VehicleParams, config: Optional[SteeringConfig] = None) -> None:
        self.params = params
        self.config = config or SteeringConfig()
        self.reset()

    def reset(self) -> None:
        self._effective: Optional[float] = None
        self._last_t: Optional[float] = None
        self._last_eff: float = 0.0
        self._rate: float = 0.0

    def _backlash(self, x: float) -> float:
        half = 0.5 * self.config.backlash_rad
        if self._effective is None:
            self._effective = x
        elif x > self._effective + half:
            self._effective = x - half
        elif x < self._effective - half:
            self._effective = x + half
        return self._effective

    def update(self, t: float, handwheel_rad: float) -> SteeringState:
        cfg = self.config
        hw = handwheel_rad - cfg.offset_rad
        eff = self._backlash(hw)
        if abs(eff) < cfg.deadband_rad:
            eff_out = 0.0
        else:
            eff_out = eff - math.copysign(cfg.deadband_rad, eff)
        lock = math.radians(self.params.max_steer_wheel_deg)
        saturated = abs(eff_out) > lock
        if cfg.clamp_to_lock:
            eff_out = max(-lock, min(lock, eff_out))

        if self._last_t is not None and t > self._last_t:
            dt = t - self._last_t
            raw = (eff_out - self._last_eff) / dt
            alpha = dt / (cfg.rate_tau_s + dt)
            self._rate += alpha * (raw - self._rate)
        self._last_t = t
        self._last_eff = eff_out

        # TODO(vehicle): online offset calibration (straight-driving yaw-rate ~ 0 => handwheel bias).
        # TODO(vehicle): speed-dependent ratio / EPS compliance model; per-wheel Ackermann angles.
        delta = handwheel_to_road_wheel(eff_out, self.params)
        return SteeringState(
            stamp=t,
            handwheel=hw,
            handwheel_effective=eff_out,
            road_wheel=delta,
            handwheel_rate=self._rate,
            road_wheel_rate=self._rate / self.params.steering_ratio,
            curvature=road_wheel_to_curvature(delta, self.params),
            saturated=saturated,
        )
