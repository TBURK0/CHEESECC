"""Yaw-rate estimation: bicycle-model prediction fused with a (biased) gyro.

Two-state linear Kalman filter over x = [r, b_gyro]:
- process: r random walk (driven by `yaw_accel_sigma`), bias slow random walk;
- measurement 1: gyro z_g = r + b + n_g;
- measurement 2 (pseudo): model r_model = v * delta / (L + K v^2), whose
  variance grows with |a_y| (tire nonlinearity) and with speed (slip
  sensitivity), so the gyro dominates when the model is untrustworthy.

Because the model measurement does not see the bias, the bias becomes
observable whenever the car drives in the model's linear regime (e.g. straight
or gentle curves). A large normalized model residual is a cheap
"car is sliding / model invalid" indicator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .params import VehicleParams


@dataclass
class YawRateConfig:
    yaw_accel_sigma: float = 3.0  # rad/s^2, process noise on r
    bias_sigma: float = 1e-3  # rad/s/sqrt(s), bias random walk
    gyro_sigma: float = 0.01  # rad/s
    model_sigma_base: float = 0.02  # rad/s
    model_sigma_per_ay: float = 0.03  # rad/s per m/s^2 of |a_y|
    model_sigma_per_speed: float = 0.002  # rad/s per m/s
    init_bias_sigma: float = 0.05
    residual_gate: float = 4.0  # sigma; model residual above => model_valid False
    use_steady_state_model: bool = True  # False => kinematic v*tan(delta)/L


@dataclass
class YawRateEstimate:
    stamp: float
    yaw_rate: float
    gyro_bias: float
    variance: float
    model_yaw_rate: float
    model_residual: float  # gyro (bias-corrected) - model
    model_valid: bool


class YawRateEstimator:
    def __init__(self, params: VehicleParams, config: Optional[YawRateConfig] = None) -> None:
        self.params = params
        self.config = config or YawRateConfig()
        self.reset()

    def reset(self, yaw_rate: float = 0.0, bias: float = 0.0) -> None:
        c = self.config
        self.x = np.array([yaw_rate, bias], dtype=float)
        self.P = np.diag([1.0, c.init_bias_sigma ** 2])
        self._t: Optional[float] = None

    def model_yaw_rate(self, v: float, road_wheel: float) -> float:
        if self.config.use_steady_state_model:
            return self.params.steady_state_yaw_rate(v, road_wheel)
        return v * np.tan(road_wheel) / self.params.wheelbase

    def _update(self, z: float, H: np.ndarray, R: float) -> float:
        y = z - float(H @ self.x)
        S = float(H @ self.P @ H) + R
        K = (self.P @ H) / S
        self.x = self.x + K * y
        I_KH = np.eye(2) - np.outer(K, H)
        self.P = I_KH @ self.P @ I_KH.T + np.outer(K, K) * R  # Joseph form
        return y / np.sqrt(S)

    def update(self, t: float, speed: float, road_wheel: float,
               gyro_z: Optional[float] = None, lat_accel: Optional[float] = None) -> YawRateEstimate:
        c = self.config
        if self._t is not None and t > self._t:
            dt = t - self._t
            self.P = self.P + np.diag([(c.yaw_accel_sigma * dt) ** 2, c.bias_sigma ** 2 * dt])
        self._t = t

        r_model = self.model_yaw_rate(speed, road_wheel)
        ay = abs(lat_accel) if lat_accel is not None else abs(speed * r_model)
        model_sigma = c.model_sigma_base + c.model_sigma_per_ay * ay + c.model_sigma_per_speed * abs(speed)

        residual = 0.0
        valid = True
        if gyro_z is not None:
            # Gate the model against the gyro before fusing it.
            r_gyro = gyro_z - self.x[1]
            residual = r_gyro - r_model
            gate_sigma = np.sqrt(model_sigma ** 2 + c.gyro_sigma ** 2 + self.P[1, 1])
            valid = abs(residual) <= c.residual_gate * gate_sigma
            self._update(gyro_z, np.array([1.0, 1.0]), c.gyro_sigma ** 2)
        if valid:
            self._update(r_model, np.array([1.0, 0.0]), model_sigma ** 2)
        # TODO(vehicle): include roll/bank-induced gyro cross-coupling; temperature bias model.
        # TODO(vehicle): learned residual model for nonlinear regime (interface: f(v, delta, ay) -> r).
        return YawRateEstimate(t, float(self.x[0]), float(self.x[1]), float(self.P[0, 0]),
                               float(r_model), float(residual), bool(valid))
