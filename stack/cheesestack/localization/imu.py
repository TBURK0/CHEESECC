"""IMU integration: standstill detection + bias estimation, gravity removal, planar strapdown.

IMU frame follows the vehicle convention (x forward, y left, z up). Accelerometers
measure specific force f = a - g_world, so a level IMU at rest reads (0, 0, +g).

What works now:
- `StandstillDetector`: sliding-window per-axis std test on accel and gyro
  (optionally AND-ed with a wheel-speed zero flag).
- `ImuBiasEstimator`: running mean of gyro (bias) and accel (gravity direction
  -> roll/pitch mount tilt) while stationary.
- `remove_gravity`: rotate specific force to the level frame and subtract g.
- `PlanarStrapdown`: integrates bias-corrected yaw rate (gyro projected on the
  gravity axis) and level-frame horizontal acceleration into x, y, yaw, vx, vy.

Limits: pure strapdown drifts quadratically; use it only as a short-horizon
prediction inside the state estimator.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cheesestack.core import Pose2D, rot2, wrap_angle

GRAVITY = 9.80665


@dataclass
class ImuSample:
    stamp: float
    accel: np.ndarray  # (3,) m/s^2 specific force, imu frame
    gyro: np.ndarray  # (3,) rad/s, imu frame


def tilt_from_accel(accel) -> tuple[float, float]:
    """Roll, pitch (rad) from a static specific-force vector. Pitch positive = nose down for x fwd, z up.

    Uses R = Ry(pitch) Rx(roll) such that R^T [0,0,g] = accel.
    """
    ax, ay, az = (float(v) for v in accel)
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.hypot(ay, az))
    return roll, pitch


def level_rotation(roll: float, pitch: float) -> np.ndarray:
    """Rotation from imu frame to the gravity-aligned (level) frame with the same heading."""
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    return ry @ rx


def remove_gravity(accel, roll: float, pitch: float, g: float = GRAVITY) -> np.ndarray:
    """Linear acceleration in the level frame: R_level * f - [0, 0, g]."""
    return level_rotation(roll, pitch) @ np.asarray(accel, dtype=float) - np.array([0.0, 0.0, g])


@dataclass
class StandstillConfig:
    window: int = 50  # samples (~0.5 s at 100 Hz)
    accel_std_max: float = 0.05  # m/s^2
    gyro_std_max: float = 0.01  # rad/s
    gyro_mean_max: float = 0.05  # rad/s, rejects slow constant turns


class StandstillDetector:
    def __init__(self, config: Optional[StandstillConfig] = None) -> None:
        self.config = config or StandstillConfig()
        self._acc: deque = deque(maxlen=self.config.window)
        self._gyr: deque = deque(maxlen=self.config.window)

    def update(self, sample: ImuSample, wheels_stopped: Optional[bool] = None) -> bool:
        self._acc.append(np.asarray(sample.accel, dtype=float))
        self._gyr.append(np.asarray(sample.gyro, dtype=float))
        if len(self._acc) < self.config.window:
            return False
        if wheels_stopped is False:
            return False
        g = np.array(self._gyr)
        a = np.array(self._acc)
        still = (float(np.max(np.std(a, axis=0))) < self.config.accel_std_max
                 and float(np.max(np.std(g, axis=0))) < self.config.gyro_std_max
                 and float(np.linalg.norm(g.mean(axis=0))) < self.config.gyro_mean_max)
        return bool(still)


@dataclass
class ImuBiasConfig:
    min_samples: int = 50
    max_samples: int = 2000  # running mean saturates into an EMA after this
    standstill: StandstillConfig = field(default_factory=StandstillConfig)


class ImuBiasEstimator:
    """Gyro bias and mount tilt from stationary periods.

    Accel bias along gravity is not separable from tilt using statics alone;
    we estimate only the gravity direction (roll/pitch) and the norm error.
    """

    def __init__(self, config: Optional[ImuBiasConfig] = None) -> None:
        self.config = config or ImuBiasConfig()
        self.detector = StandstillDetector(self.config.standstill)
        self.gyro_bias = np.zeros(3)
        self.mean_accel = np.array([0.0, 0.0, GRAVITY])
        self.n = 0

    @property
    def ready(self) -> bool:
        return self.n >= self.config.min_samples

    @property
    def roll_pitch(self) -> tuple[float, float]:
        return tilt_from_accel(self.mean_accel)

    @property
    def accel_scale_error(self) -> float:
        return float(np.linalg.norm(self.mean_accel) / GRAVITY - 1.0)

    def update(self, sample: ImuSample, wheels_stopped: Optional[bool] = None) -> bool:
        still = self.detector.update(sample, wheels_stopped)
        if still:
            self.n += 1
            k = 1.0 / min(self.n, self.config.max_samples)
            self.gyro_bias += k * (np.asarray(sample.gyro, float) - self.gyro_bias)
            self.mean_accel += k * (np.asarray(sample.accel, float) - self.mean_accel)
        return still
    # TODO(localization): online bias random-walk states in the EKF, temperature compensation,
    # accel bias via multi-orientation calibration, and full mount extrinsic (yaw) from driving data.


@dataclass
class StrapdownConfig:
    max_dt: float = 0.1  # larger gaps are skipped (dt clamped to 0)
    zupt: bool = True  # zero-velocity update when the bias estimator reports standstill


class PlanarStrapdown:
    """2D inertial dead reckoning in the `odom` frame.

    yaw_dot = (R_level (gyro - bias))_z ; [vx, vy]_world += R(yaw) a_level_xy * dt.
    """

    def __init__(self, config: Optional[StrapdownConfig] = None, bias: Optional[ImuBiasEstimator] = None,
                 pose: Pose2D = Pose2D(), velocity=(0.0, 0.0)) -> None:
        self.config = config or StrapdownConfig()
        self.bias = bias
        self.x, self.y, self.yaw = pose.x, pose.y, pose.yaw
        self.vel = np.array(velocity, dtype=float)
        self.last_stamp: Optional[float] = None
        self.yaw_rate = 0.0
        self.accel_body = np.zeros(2)

    @property
    def pose(self) -> Pose2D:
        return Pose2D(self.x, self.y, self.yaw)

    @property
    def speed(self) -> float:
        """Signed speed along heading."""
        return float(self.vel @ np.array([math.cos(self.yaw), math.sin(self.yaw)]))

    def update(self, sample: ImuSample) -> Pose2D:
        if self.last_stamp is None:
            self.last_stamp = sample.stamp
            return self.pose
        dt = sample.stamp - self.last_stamp
        self.last_stamp = sample.stamp
        if dt <= 0 or dt > self.config.max_dt:
            return self.pose
        gyro = np.asarray(sample.gyro, float)
        roll = pitch = 0.0
        still = False
        if self.bias is not None:
            still = self.bias.update(sample)
            gyro = gyro - self.bias.gyro_bias
            if self.bias.ready:
                roll, pitch = self.bias.roll_pitch
        R = level_rotation(roll, pitch)
        self.yaw_rate = float((R @ gyro)[2])
        a_lin = remove_gravity(sample.accel, roll, pitch)[:2]
        self.accel_body = a_lin
        yaw_mid = self.yaw + 0.5 * self.yaw_rate * dt
        a_world = rot2(yaw_mid) @ a_lin
        if still and self.config.zupt:
            self.vel[:] = 0.0
            a_world = np.zeros(2)
        self.x += self.vel[0] * dt + 0.5 * a_world[0] * dt * dt
        self.y += self.vel[1] * dt + 0.5 * a_world[1] * dt * dt
        self.vel += a_world * dt
        self.yaw = wrap_angle(self.yaw + self.yaw_rate * dt)
        return self.pose
    # TODO(localization): coning/sculling correction for high-rate IMUs, non-holonomic constraint
    # (vy_body ~ 0) as a pseudo-measurement, lever arm imu -> base_link.
