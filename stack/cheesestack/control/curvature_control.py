"""Curvature control: curvature <-> road-wheel angle <-> handwheel degrees.

Kinematic bicycle (rear-axle reference), same as BicycleModel.swift:
    road_wheel = atan(kappa * L)
    handwheel_deg = road_wheel * steering_ratio * 180/pi, clamped to +-max_steer_wheel_deg
Small-angle gain L * ratio * 180/pi deg per 1/m (`theoretical_gain`).

`CurvatureController` adds a curvature rate limit (1/m/s) and magnitude limit,
then `make_control_command` builds a core.ControlCommand.

SIGN: inputs here are stack convention (+ = left). `make_control_command`
flips curvature and handwheel to the app's DriveCommand convention (+ = right,
compass bearings) when `app_sign_convention=True` (the default).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cheesestack.core import ControlCommand, Header

from .saturation import RateLimiter, RateLimiterConfig, clamp
from .vehicle_params import VehicleParams


@dataclass
class SteerSolution:
    steer_wheel_deg: float
    curvature: float
    road_wheel_deg: float
    saturated: bool = False


def curvature_to_road_wheel(kappa: float, params: VehicleParams) -> float:
    return math.atan(kappa * params.wheelbase_m)


def road_wheel_to_curvature(delta_rad: float, params: VehicleParams) -> float:
    return math.tan(delta_rad) / params.wheelbase_m


def road_wheel_to_handwheel_deg(delta_rad: float, params: VehicleParams) -> float:
    return math.degrees(delta_rad) * params.steering_ratio


def handwheel_deg_to_road_wheel(hw_deg: float, params: VehicleParams) -> float:
    return math.radians(hw_deg / params.steering_ratio)


def theoretical_gain(params: VehicleParams) -> float:
    """Small-angle handwheel deg per 1/m of curvature (BicycleModel.theoreticalGain)."""
    return params.wheelbase_m * params.steering_ratio * 180.0 / math.pi


def curvature_to_steer(kappa: float, params: VehicleParams, extra_road_wheel_rad: float = 0.0) -> SteerSolution:
    """Port of BicycleModel.steer's conversion. `extra_road_wheel_rad` carries
    dynamic feedforward (understeer) added on top of the kinematic angle."""
    delta = curvature_to_road_wheel(kappa, params) + extra_road_wheel_rad
    hw = road_wheel_to_handwheel_deg(delta, params)
    lim = params.max_steer_wheel_deg
    hw_c = clamp(hw, -lim, lim)
    return SteerSolution(hw_c, kappa, math.degrees(delta), hw_c != hw)


@dataclass
class CurvatureControlConfig:
    max_curvature_rate: float = 0.1  # 1/m per s (~ comfortable steering rate at speed)
    max_curvature: float | None = None  # defaults to vehicle full-lock curvature
    app_sign_convention: bool = True


class CurvatureController:
    """Rate- and magnitude-limits a curvature request and solves the handwheel angle."""

    def __init__(self, params: VehicleParams | None = None, config: CurvatureControlConfig | None = None) -> None:
        self.params = params or VehicleParams()
        self.config = config or CurvatureControlConfig()
        kmax = self.config.max_curvature or self.params.max_curvature
        self.limiter = RateLimiter(RateLimiterConfig(self.config.max_curvature_rate, None, -kmax, kmax))

    def reset(self, kappa: float = 0.0) -> None:
        self.limiter.reset(kappa)

    def update(self, kappa_request: float, dt: float, extra_road_wheel_rad: float = 0.0) -> SteerSolution:
        k = self.limiter.update(kappa_request, dt)
        sol = curvature_to_steer(k, self.params, extra_road_wheel_rad)
        sol.saturated = sol.saturated or self.limiter.saturated
        return sol


def make_control_command(
    header: Header,
    steer: SteerSolution,
    throttle: float,
    brake: float,
    target_speed_mps: float,
    app_sign_convention: bool = True,
    source: str = "control",
) -> ControlCommand:
    """Build a core.ControlCommand. With app_sign_convention, +curvature/+steer = RIGHT
    (DriveCommand.swift); otherwise stack convention (+ = left)."""
    sgn = -1.0 if app_sign_convention else 1.0
    return ControlCommand(
        header=header,
        steer_wheel_deg=sgn * steer.steer_wheel_deg,
        throttle=clamp(throttle, 0.0, 1.0),
        brake=clamp(brake, 0.0, 1.0),
        curvature=sgn * steer.curvature,
        target_speed_mps=max(0.0, target_speed_mps),
        source=source,
    )
