"""Velocity and curvature estimation.

- `VelocityEstimator`: alpha-beta (g-h) filter on 2D position, fused with an
  optional direct speed measurement (wheels), producing world velocity,
  signed longitudinal speed and longitudinal acceleration.
- `CurvatureEstimator`: kappa = yaw_rate / v with low-speed regularization
  kappa = r v / max(v^2, v_min^2), blending to the steer-based bicycle
  curvature tan(steer)/L when slow, clamped to the vehicle's max curvature.
- `curvature_from_path`: signed Menger curvature (CCW / left turn positive,
  the stack convention; NOTE the app's GeoMath.signedCurvature is right-positive).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VelocityConfig:
    alpha: float = 0.5
    beta: float = 0.1
    speed_meas_weight: float = 0.7  # blend weight for direct speed measurements
    accel_tau_s: float = 0.3


class VelocityEstimator:
    def __init__(self, config: Optional[VelocityConfig] = None) -> None:
        self.config = config or VelocityConfig()
        self.pos: Optional[np.ndarray] = None
        self.vel = np.zeros(2)
        self.accel = 0.0
        self._t: Optional[float] = None
        self._speed_prev: Optional[float] = None

    def update_position(self, t: float, xy) -> np.ndarray:
        z = np.asarray(xy, float)
        if self.pos is None or self._t is None:
            self.pos, self._t = z.copy(), t
            return self.vel
        dt = t - self._t
        if dt <= 0:
            return self.vel
        pred = self.pos + self.vel * dt
        r = z - pred
        self.pos = pred + self.config.alpha * r
        self.vel = self.vel + self.config.beta / dt * r
        self._update_accel(t, dt)
        self._t = t
        return self.vel

    def update_speed(self, t: float, speed: float, yaw: float) -> np.ndarray:
        """Blend a measured signed speed along `yaw` into the velocity estimate."""
        w = self.config.speed_meas_weight
        u = np.array([math.cos(yaw), math.sin(yaw)])
        along = float(self.vel @ u)
        self.vel = self.vel + w * (speed - along) * u
        return self.vel

    def _update_accel(self, t: float, dt: float) -> None:
        s = float(np.linalg.norm(self.vel))
        if self._speed_prev is not None:
            a_raw = (s - self._speed_prev) / dt
            k = dt / (self.config.accel_tau_s + dt)
            self.accel += k * (a_raw - self.accel)
        self._speed_prev = s

    def speed(self, yaw: Optional[float] = None) -> float:
        if yaw is None:
            return float(np.linalg.norm(self.vel))
        return float(self.vel @ np.array([math.cos(yaw), math.sin(yaw)]))


@dataclass
class CurvatureConfig:
    v_min_mps: float = 1.0
    wheelbase_m: float = 2.7
    max_curvature: float = 0.2  # 1/m (~5 m turning radius)
    lowpass_tau_s: float = 0.1


class CurvatureEstimator:
    def __init__(self, config: Optional[CurvatureConfig] = None) -> None:
        self.config = config or CurvatureConfig()
        self.kappa = 0.0
        self._t: Optional[float] = None

    def instantaneous(self, v: float, yaw_rate: float, steer: Optional[float] = None) -> float:
        c = self.config
        k_dyn = yaw_rate * v / max(v * v, c.v_min_mps**2)
        if steer is not None:
            k_steer = math.tan(steer) / c.wheelbase_m
            w = min(1.0, abs(v) / c.v_min_mps)  # trust dynamics more as speed rises
            k = w * k_dyn + (1 - w) * k_steer
        else:
            k = k_dyn
        return float(np.clip(k, -c.max_curvature, c.max_curvature))

    def update(self, t: float, v: float, yaw_rate: float, steer: Optional[float] = None) -> float:
        k = self.instantaneous(v, yaw_rate, steer)
        if self._t is None or self.config.lowpass_tau_s <= 0:
            self.kappa = k
        else:
            dt = max(0.0, t - self._t)
            a = dt / (self.config.lowpass_tau_s + dt) if dt > 0 else 0.0
            self.kappa += a * (k - self.kappa)
        self._t = t
        return self.kappa


def curvature_from_path(points) -> np.ndarray:
    """Signed Menger curvature at each point of an (N,2) polyline (ends copied from neighbours).

    Positive = turning left (CCW). Degenerate (repeated/collinear) triples give 0.
    """
    p = np.asarray(points, float).reshape(-1, 2)
    n = len(p)
    k = np.zeros(n)
    if n < 3:
        return k
    a, b, c = p[:-2], p[1:-1], p[2:]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    denom = np.linalg.norm(b - a, axis=1) * np.linalg.norm(c - b, axis=1) * np.linalg.norm(c - a, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        k[1:-1] = np.where(denom > 1e-9, 2.0 * cross / denom, 0.0)
    k[0], k[-1] = k[1], k[-2]
    return k
# TODO(localization): curvature from map/lane geometry, Savitzky-Golay smoothing of path curvature,
# understeer gradient in the steer->curvature model.
