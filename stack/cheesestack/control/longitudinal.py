"""Longitudinal trajectory tracking: speed PID + acceleration feedforward -> pedals.

    a_cmd = a_ref + PID(v_ref - v)          (anti-windup clamp on a_cmd limits)
    pedals = inverse of the app's LongitudinalModel.acceleration:
        throttle: F = m a + drag + roll, t = F / min(torque*9/r, P/max(v,3))
        brake:    if a_cmd below coast decel, b = (a_coast - a_cmd) / max_brake_decel

`target_speed_for_curvature` and `simple_pedals` are direct ports of
LongitudinalModel.targetSpeed / .pedals for parity with the app.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .saturation import PID, PIDConfig, clamp
from .vehicle_params import VehicleParams

RHO_AIR = 1.225
G = 9.81
ROLL_COEFF = 0.013


def target_speed_for_curvature(params: VehicleParams, path_curvature: float) -> float:
    """Port of LongitudinalModel.targetSpeed (m/s)."""
    comfort = max(2.2, 4.2 - params.mass_kg / 1800.0)
    k = abs(path_curvature)
    curve_cap = 16.0 if k < 0.002 else math.sqrt(comfort / k)
    hp_per_kg = params.horsepower / max(400.0, params.mass_kg)
    power_cap = 8.0 + hp_per_kg * 40.0
    return min(13.5, curve_cap, power_cap)


def simple_pedals(params: VehicleParams, speed: float, target: float) -> tuple[float, float]:
    """Port of LongitudinalModel.pedals: proportional throttle/brake on speed error."""
    err = target - speed
    if err >= 0:
        hp_per_tonne = params.horsepower / (params.mass_kg / 1000.0)
        authority = min(1.0, hp_per_tonne / 220.0)
        return min(1.0, (err / 3.5) * (0.35 + 0.65 * authority)), 0.0
    return 0.0, min(1.0, -err / 3.5)


def resistive_force(params: VehicleParams, v: float) -> float:
    m = max(400.0, params.mass_kg)
    return 0.5 * RHO_AIR * params.drag_coefficient * params.frontal_area_m2 * v * v + ROLL_COEFF * m * G


def accel_to_pedals(params: VehicleParams, a_cmd: float, v: float) -> tuple[float, float]:
    """Invert the app's longitudinal plant. Returns (throttle, brake) in [0, 1]."""
    m = max(400.0, params.mass_kg)
    f_res = resistive_force(params, v)
    a_coast = -f_res / m
    if a_cmd >= a_coast:
        f_needed = m * a_cmd + f_res
        per_throttle = min(
            params.torque_nm * 9.0 / max(0.2, params.tire_radius_m),
            params.horsepower * 745.7 / max(v, 3.0),
        )
        t = f_needed / max(per_throttle, 1e-6)
        t = clamp(t, 0.0, 1.0)
        # Plant ignores throttle <= 0.02 (coast). Round to the nearer of coast / minimum effective throttle.
        if t <= 0.02:
            t = 0.021 if t > 0.0105 else 0.0
        return t, 0.0
    return 0.0, clamp((a_coast - a_cmd) / params.max_brake_decel, 0.0, 1.0)


@dataclass
class LongitudinalConfig:
    pid: PIDConfig = field(default_factory=lambda: PIDConfig(kp=0.8, ki=0.25, kd=0.0, out_min=-6.0, out_max=3.0,
                                                               anti_windup="clamp", i_limit=2.0))
    max_accel: float = 2.5
    max_decel: float = 6.0
    use_feedforward: bool = True
    standstill_speed_mps: float = 0.2
    standstill_brake: float = 0.3  # hold brake when stopped with v_ref ~ 0


@dataclass
class LongitudinalOutput:
    accel_cmd: float
    throttle: float
    brake: float
    speed_error: float


class LongitudinalController:
    def __init__(self, params: VehicleParams | None = None, config: LongitudinalConfig | None = None) -> None:
        self.params = params or VehicleParams()
        self.config = config or LongitudinalConfig()
        self.pid = PID(self.config.pid)

    def reset(self) -> None:
        self.pid.reset()

    def update(self, v: float, v_ref: float, a_ref: float, dt: float) -> LongitudinalOutput:
        c = self.config
        err = v_ref - v
        ff = a_ref if c.use_feedforward else 0.0
        a = self.pid.update(err, dt, feedforward=ff)
        a = clamp(a, -c.max_decel, c.max_accel)
        if v_ref < c.standstill_speed_mps and v < c.standstill_speed_mps:
            self.pid.reset()
            return LongitudinalOutput(0.0, 0.0, c.standstill_brake, err)
        thr, brk = accel_to_pedals(self.params, a, v)
        # TODO(control): grade compensation from pitch, gear/creep model, pedal-map calibration table.
        return LongitudinalOutput(a, thr, brk, err)
