"""Kalman-filter variants (numpy): linear KF, EKF, UKF, plus planar process models.

All filters store `x` (n,) and `P` (n,n), expose `predict(...)` and `update(...)`,
and return the normalized innovation squared (NIS) from `update` for gating /
consistency monitoring. Angle states are handled through `angle_indices`
(wrapped after every step and in residuals).

Process models:
- `CTRVModel`: state [x, y, yaw, v, yaw_rate], constant turn rate & velocity.
- `KinematicBicycleModel`: state [x, y, yaw, v], control [accel, steer], rear-axle reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from cheesestack.core import wrap_angle


def _wrap_idx(v: np.ndarray, idx: Sequence[int]) -> np.ndarray:
    v = np.array(v, dtype=float, copy=True)
    for i in idx:
        v[..., i] = (v[..., i] + math.pi) % (2 * math.pi) - math.pi
    return v


def numerical_jacobian(f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float = 1e-6,
                       angle_out: Sequence[int] = ()) -> np.ndarray:
    x = np.asarray(x, float)
    fx = np.asarray(f(x), float)
    J = np.zeros((fx.size, x.size))
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = eps
        d = (np.asarray(f(x + dx), float) - np.asarray(f(x - dx), float))
        d = _wrap_idx(d, angle_out)
        J[:, i] = d / (2 * eps)
    return J


def _joseph(P, K, H, R):
    I_KH = np.eye(P.shape[0]) - K @ H
    P = I_KH @ P @ I_KH.T + K @ R @ K.T
    return 0.5 * (P + P.T)


class KalmanFilter:
    """Linear KF: x' = F x + B u + w,  z = H x + v."""

    def __init__(self, x0, P0, F=None, H=None, Q=None, R=None, B=None, angle_indices: Sequence[int] = ()) -> None:
        self.x = np.asarray(x0, float).copy()
        self.P = np.asarray(P0, float).copy()
        n = self.x.size
        self.F = np.eye(n) if F is None else np.asarray(F, float)
        self.H = H if H is None else np.asarray(H, float)
        self.Q = np.zeros((n, n)) if Q is None else np.asarray(Q, float)
        self.R = R if R is None else np.asarray(R, float)
        self.B = B
        self.angle_indices = tuple(angle_indices)

    def predict(self, u=None, F=None, Q=None, B=None) -> None:
        F = self.F if F is None else np.asarray(F, float)
        Q = self.Q if Q is None else np.asarray(Q, float)
        B = self.B if B is None else B
        self.x = F @ self.x
        if u is not None and B is not None:
            self.x = self.x + np.asarray(B, float) @ np.asarray(u, float)
        self.x = _wrap_idx(self.x, self.angle_indices)
        self.P = F @ self.P @ F.T + Q

    def update(self, z, H=None, R=None, residual_angles: Sequence[int] = ()) -> float:
        H = self.H if H is None else np.asarray(H, float)
        R = self.R if R is None else np.asarray(R, float)
        y = _wrap_idx(np.atleast_1d(np.asarray(z, float)) - H @ self.x, residual_angles)
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        self.x = _wrap_idx(self.x + K @ y, self.angle_indices)
        self.P = _joseph(self.P, K, H, R)
        return float(y @ np.linalg.solve(S, y))


class ExtendedKalmanFilter:
    """EKF with user-supplied f/h and (optional) Jacobians; numeric Jacobians otherwise."""

    def __init__(self, x0, P0, angle_indices: Sequence[int] = ()) -> None:
        self.x = np.asarray(x0, float).copy()
        self.P = np.asarray(P0, float).copy()
        self.angle_indices = tuple(angle_indices)

    def predict(self, f: Callable[[np.ndarray], np.ndarray], Q, F: Optional[np.ndarray] = None) -> None:
        if F is None:
            F = numerical_jacobian(f, self.x, angle_out=self.angle_indices)
        self.x = _wrap_idx(f(self.x), self.angle_indices)
        self.P = F @ self.P @ F.T + np.asarray(Q, float)
        self.P = 0.5 * (self.P + self.P.T)

    def innovation(self, z, h, H=None, R=None, residual_angles: Sequence[int] = ()):
        z = np.atleast_1d(np.asarray(z, float))
        if H is None:
            H = numerical_jacobian(lambda s: np.atleast_1d(h(s)), self.x, angle_out=residual_angles)
        H = np.atleast_2d(H)
        y = _wrap_idx(z - np.atleast_1d(h(self.x)), residual_angles)
        S = H @ self.P @ H.T + np.atleast_2d(R)
        return y, S, H

    def update(self, z, h, R, H=None, residual_angles: Sequence[int] = (), gate: Optional[float] = None) -> float:
        """Returns NIS. If `gate` is set and NIS > gate, the update is skipped (NIS still returned)."""
        R = np.atleast_2d(np.asarray(R, float))
        y, S, H = self.innovation(z, h, H, R, residual_angles)
        nis = float(y @ np.linalg.solve(S, y))
        if gate is not None and nis > gate:
            return nis
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        self.x = _wrap_idx(self.x + K @ y, self.angle_indices)
        self.P = _joseph(self.P, K, H, R)
        return nis


class UnscentedKalmanFilter:
    """UKF with Van der Merwe scaled sigma points and angle-aware mean/residuals."""

    def __init__(self, x0, P0, alpha: float = 1e-1, beta: float = 2.0, kappa: float = 0.0,
                 angle_indices: Sequence[int] = ()) -> None:
        self.x = np.asarray(x0, float).copy()
        self.P = np.asarray(P0, float).copy()
        self.n = self.x.size
        self.angle_indices = tuple(angle_indices)
        n = self.n
        lam = alpha**2 * (n + kappa) - n
        self._c = n + lam
        self.Wm = np.full(2 * n + 1, 1.0 / (2 * self._c))
        self.Wc = self.Wm.copy()
        self.Wm[0] = lam / self._c
        self.Wc[0] = lam / self._c + (1 - alpha**2 + beta)

    def sigma_points(self) -> np.ndarray:
        P = 0.5 * (self.P + self.P.T)
        try:
            L = np.linalg.cholesky(self._c * P)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(self._c * P)
            L = V @ np.diag(np.sqrt(np.clip(w, 0, None)))
        pts = np.empty((2 * self.n + 1, self.n))
        pts[0] = self.x
        pts[1:self.n + 1] = self.x + L.T
        pts[self.n + 1:] = self.x - L.T
        return pts

    def _mean(self, pts: np.ndarray, angles: Sequence[int]) -> np.ndarray:
        m = self.Wm @ pts
        for i in angles:
            m[i] = math.atan2(self.Wm @ np.sin(pts[:, i]), self.Wm @ np.cos(pts[:, i]))
        return m

    def predict(self, f: Callable[[np.ndarray], np.ndarray], Q) -> None:
        pts = np.array([f(p) for p in self.sigma_points()])
        self.x = self._mean(pts, self.angle_indices)
        d = _wrap_idx(pts - self.x, self.angle_indices)
        self.P = (self.Wc[:, None] * d).T @ d + np.asarray(Q, float)

    def update(self, z, h, R, residual_angles: Sequence[int] = (), gate: Optional[float] = None) -> float:
        pts = self.sigma_points()
        zs = np.array([np.atleast_1d(h(p)) for p in pts])
        zm = self._mean(zs, residual_angles)
        dz = _wrap_idx(zs - zm, residual_angles)
        dx = _wrap_idx(pts - self.x, self.angle_indices)
        S = (self.Wc[:, None] * dz).T @ dz + np.atleast_2d(R)
        Pxz = (self.Wc[:, None] * dx).T @ dz
        y = _wrap_idx(np.atleast_1d(np.asarray(z, float)) - zm, residual_angles)
        nis = float(y @ np.linalg.solve(S, y))
        if gate is not None and nis > gate:
            return nis
        K = np.linalg.solve(S.T, Pxz.T).T
        self.x = _wrap_idx(self.x + K @ y, self.angle_indices)
        self.P = self.P - K @ S @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        return nis


# --- process models -----------------------------------------------------------
@dataclass
class CTRVModel:
    """State [x, y, yaw, v, yaw_rate]. Noise: white longitudinal accel and yaw accel."""

    sigma_accel: float = 1.0  # m/s^2
    sigma_yaw_accel: float = 0.5  # rad/s^2
    eps_yaw_rate: float = 1e-4

    angle_indices = (2,)
    dim = 5

    def f(self, s: np.ndarray, dt: float) -> np.ndarray:
        x, y, th, v, w = s
        if abs(w) > self.eps_yaw_rate:
            nx = x + v / w * (math.sin(th + w * dt) - math.sin(th))
            ny = y + v / w * (math.cos(th) - math.cos(th + w * dt))
        else:
            # Second-order expansion in w so the model matches its Jacobian across the branch.
            nx = x + v * math.cos(th) * dt - 0.5 * v * math.sin(th) * w * dt * dt
            ny = y + v * math.sin(th) * dt + 0.5 * v * math.cos(th) * w * dt * dt
        return np.array([nx, ny, wrap_angle(th + w * dt), v, w])

    def jacobian(self, s: np.ndarray, dt: float) -> np.ndarray:
        x, y, th, v, w = s
        F = np.eye(5)
        if abs(w) > self.eps_yaw_rate:
            s0, c0 = math.sin(th), math.cos(th)
            s1, c1 = math.sin(th + w * dt), math.cos(th + w * dt)
            F[0, 2] = v / w * (c1 - c0)
            F[0, 3] = (s1 - s0) / w
            F[0, 4] = v * dt * c1 / w - v * (s1 - s0) / w**2
            F[1, 2] = v / w * (s1 - s0)
            F[1, 3] = (c0 - c1) / w
            F[1, 4] = v * dt * s1 / w - v * (c0 - c1) / w**2
        else:
            c, s_ = math.cos(th), math.sin(th)
            F[0, 2] = -v * s_ * dt
            F[0, 3] = c * dt
            F[0, 4] = -0.5 * v * s_ * dt * dt
            F[1, 2] = v * c * dt
            F[1, 3] = s_ * dt
            F[1, 4] = 0.5 * v * c * dt * dt
        F[2, 4] = dt
        return F

    def Q(self, s: np.ndarray, dt: float) -> np.ndarray:
        th = s[2]
        c, sn = math.cos(th), math.sin(th)
        G = np.array([
            [0.5 * dt * dt * c, 0.0],
            [0.5 * dt * dt * sn, 0.0],
            [0.0, 0.5 * dt * dt],
            [dt, 0.0],
            [0.0, dt],
        ])
        return G @ np.diag([self.sigma_accel**2, self.sigma_yaw_accel**2]) @ G.T


@dataclass
class KinematicBicycleModel:
    """State [x, y, yaw, v]; control [accel, steer]. Reference point: rear axle (base_link)."""

    wheelbase_m: float = 2.7
    sigma_accel: float = 0.5
    sigma_steer: float = 0.02

    angle_indices = (2,)
    dim = 4

    def f(self, s: np.ndarray, dt: float, u=(0.0, 0.0)) -> np.ndarray:
        x, y, th, v = s
        a, d = u
        w = v * math.tan(d) / self.wheelbase_m
        th_mid = th + 0.5 * w * dt
        return np.array([
            x + v * math.cos(th_mid) * dt,
            y + v * math.sin(th_mid) * dt,
            wrap_angle(th + w * dt),
            v + a * dt,
        ])

    def jacobian(self, s: np.ndarray, dt: float, u=(0.0, 0.0)) -> np.ndarray:
        return numerical_jacobian(lambda q: self.f(q, dt, u), np.asarray(s, float), angle_out=(2,))

    def Q(self, s: np.ndarray, dt: float, u=(0.0, 0.0)) -> np.ndarray:
        # Map control noise through the control Jacobian.
        Gu = numerical_jacobian(lambda uu: self.f(np.asarray(s, float), dt, uu), np.asarray(u, float), angle_out=(2,))
        Q = Gu @ np.diag([self.sigma_accel**2, self.sigma_steer**2]) @ Gu.T
        return Q + np.eye(4) * 1e-9
# TODO(localization): square-root UKF for numerical robustness, IMM over CTRV/CV for manoeuvres,
# error-state (indirect) EKF with IMU bias states once a real IMU is mounted.
