"""Heading estimation: complementary filter of gyro yaw rate and GPS course over ground.

yaw_k = yaw_{k-1} + (gyro_z - b) dt                      (prediction)
yaw  += k_p * wrap(course - yaw)                          (correction, only when moving)
b    -= k_i * wrap(course - yaw) * dt                     (slow gyro-bias integral, Mahony-style)

GPS course is only meaningful when moving forward above `min_speed_mps`; a
reverse gear flips it by pi (`reversing=True`). Gain is scaled by course
accuracy when one is provided.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from cheesestack.core import wrap_angle


@dataclass
class HeadingConfig:
    time_constant_s: float = 2.0  # crossover: gyro dominates below, GPS above
    bias_gain: float = 0.02  # 1/s^2 integral gain
    min_speed_mps: float = 2.0
    max_bias_rad_s: float = 0.05
    nominal_course_sigma_rad: float = math.radians(3.0)
    init_on_first_course: bool = True


class HeadingFilter:
    def __init__(self, config: Optional[HeadingConfig] = None, yaw0: float = 0.0) -> None:
        self.config = config or HeadingConfig()
        self.yaw = yaw0
        self.bias = 0.0
        self.initialized = not self.config.init_on_first_course
        self._last_t: Optional[float] = None
        self._last_course_t: Optional[float] = None

    def predict(self, t: float, gyro_z: float) -> float:
        if self._last_t is not None and t > self._last_t:
            self.yaw = wrap_angle(self.yaw + (gyro_z - self.bias) * (t - self._last_t))
        self._last_t = t
        return self.yaw

    def correct(self, t: float, course_yaw: float, speed_mps: float, course_sigma_rad: Optional[float] = None,
                reversing: bool = False) -> bool:
        """Fuse a GPS course (stack yaw convention). Returns True if applied."""
        c = self.config
        if speed_mps < c.min_speed_mps or not math.isfinite(course_yaw):
            return False
        if reversing:
            course_yaw = wrap_angle(course_yaw + math.pi)
        if not self.initialized:
            self.yaw, self.initialized, self._last_course_t = course_yaw, True, t
            return True
        dt = 0.1 if self._last_course_t is None else max(1e-3, t - self._last_course_t)
        self._last_course_t = t
        sig = course_sigma_rad or c.nominal_course_sigma_rad
        k = min(1.0, dt / c.time_constant_s * (c.nominal_course_sigma_rad / sig) ** 2)
        e = wrap_angle(course_yaw - self.yaw)
        self.yaw = wrap_angle(self.yaw + k * e)
        self.bias = float(np.clip(self.bias - c.bias_gain * e * dt, -c.max_bias_rad_s, c.max_bias_rad_s))
        return True
    # TODO(localization): dual-antenna GPS heading, magnetometer fallback at standstill,
    # course-accuracy model from RMC/phone `courseAccuracy`, slip-angle compensation in turns.


def course_from_positions(p0, p1, min_dist_m: float = 0.5) -> Optional[float]:
    """Course (stack yaw) from two ENU positions; None if too close."""
    d = np.asarray(p1, float) - np.asarray(p0, float)
    if float(np.hypot(*d)) < min_dist_m:
        return None
    return math.atan2(d[1], d[0])
