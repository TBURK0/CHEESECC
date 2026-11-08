"""Slip estimation: per-wheel longitudinal slip ratio and body sideslip angle.

Longitudinal slip (SAE-like, symmetric for drive and brake):
    s = (omega * R - v_wheel) / max(|omega * R|, |v_wheel|, v_eps)
where v_wheel is the ground speed of the wheel center along the wheel heading,
computed from body (vx, vy, r) at the CG and the steer angle for front wheels.

Sideslip: Kalman filter on the linear dynamic bicycle lateral states [vy, r]
(see `DynamicBicycle.lateral_state_space`) with gyro yaw rate and lateral
accelerometer as measurements. beta = atan2(vy, vx). Below `min_speed` the
filter is held at the kinematic solution (beta = atan(lr tan(delta) / L)).
Limits: linear tires, flat road (no bank/roll correction of a_y).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .dynamics import DynamicBicycle, DynamicBicycleConfig
from .params import VehicleParams

WHEELS = ("fl", "fr", "rl", "rr")


@dataclass
class SlipConfig:
    v_eps: float = 0.5  # m/s, denominator floor for slip ratio
    min_speed: float = 2.0  # m/s, below: kinematic sideslip
    q_vy: float = 0.3  # m/s/sqrt(s) process noise
    q_r: float = 0.1  # rad/s/sqrt(s)
    gyro_sigma: float = 0.01  # rad/s
    ay_sigma: float = 0.3  # m/s^2
    slip_warn: float = 0.15  # |s| above => wheel slipping/locking


@dataclass
class SlipEstimate:
    stamp: float
    slip_ratio: dict = field(default_factory=dict)  # wheel -> s
    wheel_ground_speed: dict = field(default_factory=dict)  # wheel -> m/s
    sideslip: float = 0.0  # rad, at CG
    vy: float = 0.0  # m/s, at CG
    yaw_rate: float = 0.0
    alpha_front: float = 0.0
    alpha_rear: float = 0.0
    slipping_wheels: tuple = ()


def wheel_positions(params: VehicleParams) -> dict[str, tuple[float, float]]:
    """Wheel centers relative to the CG (x fwd, y left)."""
    return {
        "fl": (params.lf, 0.5 * params.track_front),
        "fr": (params.lf, -0.5 * params.track_front),
        "rl": (-params.lr, 0.5 * params.track_rear),
        "rr": (-params.lr, -0.5 * params.track_rear),
    }


def wheel_ground_speeds(params: VehicleParams, vx: float, vy: float, r: float, road_wheel: float) -> dict[str, float]:
    """Ground speed of each wheel center along its rolling direction [m/s]."""
    out = {}
    cd, sd = math.cos(road_wheel), math.sin(road_wheel)
    for w, (px, py) in wheel_positions(params).items():
        ux, uy = vx - r * py, vy + r * px
        out[w] = ux * cd + uy * sd if w.startswith("f") else ux
    return out


def slip_ratio(wheel_linear_speed: float, ground_speed: float, v_eps: float = 0.5) -> float:
    den = max(abs(wheel_linear_speed), abs(ground_speed), v_eps)
    return (wheel_linear_speed - ground_speed) / den


class SideslipEstimator:
    """Linear KF over [vy, r] using the dynamic bicycle model."""

    def __init__(self, params: VehicleParams, config: Optional[SlipConfig] = None) -> None:
        self.params = params
        self.config = config or SlipConfig()
        self.model = DynamicBicycle(params, DynamicBicycleConfig(min_speed=self.config.min_speed))
        self.reset()

    def reset(self) -> None:
        self.x = np.zeros(2)
        self.P = np.diag([1.0, 0.1])
        self._t: Optional[float] = None
        self._delta = 0.0

    def update(self, t: float, vx: float, road_wheel: float,
               gyro_z: Optional[float] = None, lat_accel: Optional[float] = None) -> tuple[float, float, float]:
        """Returns (beta, vy, r)."""
        c, p = self.config, self.params
        dt = 0.0 if self._t is None else max(0.0, t - self._t)
        self._t = t
        if abs(vx) < c.min_speed:
            beta = math.atan(p.lr * math.tan(road_wheel) / p.wheelbase)
            r = vx * math.cos(beta) * math.tan(road_wheel) / p.wheelbase
            self.x = np.array([vx * math.tan(beta), r if gyro_z is None else gyro_z])
            self.P = np.diag([0.1, 0.01])
            self._delta = road_wheel
            return beta, float(self.x[0]), float(self.x[1])

        A, B, C, D = self.model.lateral_state_space(vx)
        if dt > 0:
            Adt = A * dt
            Ad = np.eye(2) + Adt + 0.5 * Adt @ Adt
            Bd = (np.eye(2) * dt + 0.5 * A * dt * dt) @ B
            self.x = Ad @ self.x + Bd * self._delta  # ZOH on previous input
            self.P = Ad @ self.P @ Ad.T + np.diag([c.q_vy ** 2 * dt, c.q_r ** 2 * dt])
        self._delta = road_wheel
        if gyro_z is not None:
            self._kf(gyro_z, np.array([0.0, 1.0]), c.gyro_sigma ** 2)
        if lat_accel is not None:
            self._kf(lat_accel - D * road_wheel, C, c.ay_sigma ** 2)
        # TODO(vehicle): bank-angle / roll compensation of a_y; adapt Cf/Cr online (RLS).
        return math.atan2(self.x[0], vx), float(self.x[0]), float(self.x[1])

    def _kf(self, z: float, H: np.ndarray, R: float) -> None:
        S = float(H @ self.P @ H) + R
        K = self.P @ H / S
        self.x = self.x + K * (z - float(H @ self.x))
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P


class SlipEstimator:
    """Combines the sideslip filter with per-wheel longitudinal slip ratios."""

    def __init__(self, params: VehicleParams, config: Optional[SlipConfig] = None) -> None:
        self.params = params
        self.config = config or SlipConfig()
        self.sideslip = SideslipEstimator(params, self.config)

    def update(self, t: float, vx: float, road_wheel: float,
               wheel_omega: Optional[Sequence[float]] = None,
               gyro_z: Optional[float] = None, lat_accel: Optional[float] = None) -> SlipEstimate:
        """`wheel_omega`: angular speeds [rad/s] ordered fl, fr, rl, rr. `vx`: reference body speed."""
        beta, vy, r = self.sideslip.update(t, vx, road_wheel, gyro_z, lat_accel)
        af, ar = self.sideslip.model.slip_angles(vx, vy, r, road_wheel)
        est = SlipEstimate(stamp=t, sideslip=beta, vy=vy, yaw_rate=r, alpha_front=af, alpha_rear=ar)
        ground = wheel_ground_speeds(self.params, vx, vy, r, road_wheel)
        est.wheel_ground_speed = ground
        if wheel_omega is not None:
            if len(wheel_omega) != 4:
                raise ValueError("wheel_omega must have 4 entries (fl, fr, rl, rr)")
            R = self.params.tire_radius
            est.slip_ratio = {w: slip_ratio(om * R, ground[w], self.config.v_eps) for w, om in zip(WHEELS, wheel_omega)}
            est.slipping_wheels = tuple(w for w, s in est.slip_ratio.items() if abs(s) > self.config.slip_warn)
        # TODO(vehicle): reference speed should come from a fused (GPS+IMU) estimator, not driven wheels.
        return est
