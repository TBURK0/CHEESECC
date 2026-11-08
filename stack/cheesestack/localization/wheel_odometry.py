"""Wheel-speed processing: ticks/rpm -> speed, per-wheel -> body speed + yaw rate, slip-aware filtering.

Body frame is `base_link` (rear-axle center). For a vehicle with rear-axle
center speed v and yaw rate r, the ideal (no-slip) wheel speeds are

    rear  left/right:  v -/+ r * track_rear / 2
    front left/right:  wheel velocity magnitude at the front contact point,
                       projected onto the wheel heading ~ (v -/+ r*track_front/2) / cos(steer)

What works now:
- Encoder tick and RPM conversion with wrap-around handling.
- Rear-differential speed and yaw rate; front pair / bicycle-model fallbacks.
- Slip detection: per-wheel residual vs. a consensus speed (median of
  center-equivalent speeds) and a wheel-acceleration plausibility limit.
  Slipping wheels are excluded; a quality score in [0, 1] is reported.
- First-order low-pass on outputs and pose-delta integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cheesestack.core import Pose2D, wrap_angle

WHEELS = ("fl", "fr", "rl", "rr")


def ticks_to_speed(delta_ticks: int, dt: float, ticks_per_rev: int, wheel_radius_m: float) -> float:
    if dt <= 0:
        return 0.0
    return 2.0 * math.pi * wheel_radius_m * (delta_ticks / ticks_per_rev) / dt


def tick_delta(curr: int, prev: int, counter_bits: int = 16) -> int:
    """Signed difference of a wrapping unsigned counter."""
    mod = 1 << counter_bits
    d = (curr - prev) % mod
    return d - mod if d >= mod // 2 else d


def rpm_to_speed(rpm: float, wheel_radius_m: float) -> float:
    return rpm * 2.0 * math.pi / 60.0 * wheel_radius_m


@dataclass
class WheelSpeeds:
    """Signed wheel surface speeds (m/s). Missing wheels are NaN."""

    stamp: float
    fl: float = float("nan")
    fr: float = float("nan")
    rl: float = float("nan")
    rr: float = float("nan")

    def as_array(self) -> np.ndarray:
        return np.array([self.fl, self.fr, self.rl, self.rr], dtype=float)


@dataclass
class WheelOdomConfig:
    wheelbase_m: float = 2.7
    track_front_m: float = 1.55
    track_rear_m: float = 1.55
    slip_abs_tol_mps: float = 0.3
    slip_rel_tol: float = 0.08
    max_wheel_accel_mps2: float = 12.0  # beyond this a wheel is spinning/locking
    lowpass_tau_s: float = 0.05  # 0 disables filtering
    speed_var_floor: float = 0.01  # (m/s)^2 reported for a healthy measurement


@dataclass
class WheelOdomEstimate:
    stamp: float
    v: float
    yaw_rate: float
    slipping: dict = field(default_factory=dict)  # wheel -> bool
    quality: float = 1.0  # 1 = all wheels consistent
    v_var: float = 0.01
    yaw_rate_var: float = 0.001
    yaw_rate_source: str = "rear"


class WheelOdometry:
    """Stateful wheel odometry with slip-aware wheel selection."""

    def __init__(self, config: Optional[WheelOdomConfig] = None) -> None:
        self.config = config or WheelOdomConfig()
        self._prev: Optional[WheelSpeeds] = None
        self._v = 0.0
        self._r = 0.0
        self._last_stamp: Optional[float] = None
        self.pose = Pose2D()

    def _center_equivalent(self, w: np.ndarray, r: float, steer: float) -> np.ndarray:
        c = self.config
        cs = max(math.cos(steer), 0.2)
        return np.array([
            w[0] * cs + r * c.track_front_m / 2.0,
            w[1] * cs - r * c.track_front_m / 2.0,
            w[2] + r * c.track_rear_m / 2.0,
            w[3] - r * c.track_rear_m / 2.0,
        ])

    def update(self, ws: WheelSpeeds, steer: float = 0.0, yaw_rate_ref: Optional[float] = None) -> WheelOdomEstimate:
        """`steer` is the road-wheel angle (rad); `yaw_rate_ref` (e.g. gyro) improves slip detection in turns."""
        c = self.config
        w = ws.as_array()
        valid = np.isfinite(w)
        r_ref = self._r if yaw_rate_ref is None else yaw_rate_ref
        slipping = np.zeros(4, dtype=bool)

        # Acceleration plausibility.
        if self._prev is not None:
            dt = ws.stamp - self._prev.stamp
            if dt > 0:
                acc = (w - self._prev.as_array()) / dt
                slipping |= np.isfinite(acc) & (np.abs(acc) > c.max_wheel_accel_mps2)

        # Consensus residual.
        ce = self._center_equivalent(w, r_ref, steer)
        ok = valid & ~slipping
        if ok.sum() >= 3:
            med = float(np.median(ce[ok]))
            tol = c.slip_abs_tol_mps + c.slip_rel_tol * abs(med)
            slipping |= valid & (np.abs(ce - med) > tol)
        ok = valid & ~slipping

        # Speed: prefer rear (non-steered) wheels.
        rear_ok = ok[2:]
        if rear_ok.any():
            v = float(np.mean(ce[2:][rear_ok]))
        elif ok[:2].any():
            v = float(np.mean(ce[:2][ok[:2]]))
        else:
            v = self._v  # hold
        # Yaw rate.
        if ok[2] and ok[3]:
            r, src = (w[3] - w[2]) / c.track_rear_m, "rear"
        elif ok[0] and ok[1]:
            cs = max(math.cos(steer), 0.2)
            r, src = (w[1] - w[0]) * cs / c.track_front_m, "front"
        else:
            r, src = v * math.tan(steer) / c.wheelbase_m, "bicycle"

        # Low-pass.
        if self._last_stamp is not None and c.lowpass_tau_s > 0:
            dt = max(0.0, ws.stamp - self._last_stamp)
            alpha = dt / (c.lowpass_tau_s + dt) if dt > 0 else 1.0
            v = self._v + alpha * (v - self._v)
            r = self._r + alpha * (r - self._r)
        # Integrate pose (midpoint).
        if self._last_stamp is not None:
            dt = ws.stamp - self._last_stamp
            if dt > 0:
                yaw_mid = self.pose.yaw + 0.5 * r * dt
                self.pose = Pose2D(self.pose.x + v * math.cos(yaw_mid) * dt,
                                   self.pose.y + v * math.sin(yaw_mid) * dt,
                                   wrap_angle(self.pose.yaw + r * dt))
        self._v, self._r, self._prev, self._last_stamp = v, r, ws, ws.stamp

        n_valid = int(valid.sum())
        quality = float(ok.sum() / n_valid) if n_valid else 0.0
        inflate = 1.0 / max(quality, 0.1) ** 2
        return WheelOdomEstimate(
            stamp=ws.stamp, v=v, yaw_rate=r,
            slipping={k: bool(s) for k, s in zip(WHEELS, slipping)},
            quality=quality,
            v_var=c.speed_var_floor * inflate,
            yaw_rate_var=(0.01 if src == "rear" else 0.03 if src == "front" else 0.1) * inflate,
            yaw_rate_source=src,
        )
    # TODO(localization): tire radius auto-calibration against GPS distance, per-wheel scale factors,
    # direction-of-travel from gear/encoder quadrature, CAN wheel-speed decoding per vehicle.
