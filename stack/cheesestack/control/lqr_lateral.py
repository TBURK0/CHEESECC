"""LQR lateral controller on the dynamic-bicycle error model (Rajamani, ch. 2.5).

State x = [e_y, de_y, e_psi, de_psi] (front-axle-agnostic, CG reference), input
delta (road-wheel rad). Continuous model A(v), B; discretised by ZOH using a
numpy matrix exponential (scaling-and-squaring Taylor). The discrete algebraic
Riccati equation is solved by fixed-point iteration

    P <- Q + A'PA - A'PB (R + B'PB)^-1 B'PA

and K = (R + B'PB)^-1 B'PA. Gains are cached per speed bucket.

Steering = -K x + delta_ff, with the standard steady-state feedforward
    delta_ff = kappa (L + K_v v^2) - k3 * e_psi_ss  (Rajamani eq. 3.12)
Output is converted to curvature via the kinematic relation for the shared
curvature pipeline.

Sign: e_y + = vehicle left of path, e_psi = vehicle yaw - path yaw; delta + = left.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from cheesestack.core import Pose2D

from .reference_path import ReferencePath
from .vehicle_params import VehicleParams


def expm(M: np.ndarray, order: int = 12) -> np.ndarray:
    """Matrix exponential by scaling and squaring + Taylor series."""
    norm = np.linalg.norm(M, ord=np.inf)
    j = max(0, int(math.ceil(math.log2(norm))) + 1) if norm > 0 else 0
    A = M / (2 ** j)
    E = np.eye(M.shape[0])
    term = np.eye(M.shape[0])
    for k in range(1, order + 1):
        term = term @ A / k
        E = E + term
    for _ in range(j):
        E = E @ E
    return E


def c2d_zoh(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    E = expm(M * dt)
    return E[:n, :n], E[:n, n:]


def solve_dare(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray,
               max_iter: int = 10000, tol: float = 1e-9) -> np.ndarray:
    """Iterative DARE solution. Raises RuntimeError if it does not converge."""
    P = Q.copy()
    for _ in range(max_iter):
        BtP = B.T @ P
        S = R + BtP @ B
        Pn = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(S, BtP @ A)
        if np.max(np.abs(Pn - P)) < tol * max(1.0, np.max(np.abs(Pn))):
            return 0.5 * (Pn + Pn.T)
        P = Pn
    raise RuntimeError("DARE did not converge")


def dlqr(A, B, Q, R) -> tuple[np.ndarray, np.ndarray]:
    P = solve_dare(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


def error_dynamics(params: VehicleParams, v: float) -> tuple[np.ndarray, np.ndarray]:
    """Continuous lateral error model at speed v (> 0)."""
    v = max(v, 1.0)
    m, Iz = params.mass_kg, params.yaw_inertia_kgm2
    cf, cr = params.cornering_stiffness_front, params.cornering_stiffness_rear
    lf, lr = params.cg_to_front_m, params.cg_to_rear_m
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, -(cf + cr) / (m * v), (cf + cr) / m, (-cf * lf + cr * lr) / (m * v)],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, -(cf * lf - cr * lr) / (Iz * v), (cf * lf - cr * lr) / Iz, -(cf * lf ** 2 + cr * lr ** 2) / (Iz * v)],
    ])
    B = np.array([[0.0], [cf / m], [0.0], [cf * lf / Iz]])
    return A, B


@dataclass
class LQRConfig:
    q_diag: tuple[float, float, float, float] = (1.0, 0.0, 1.0, 0.0)
    r: float = 10.0
    dt: float = 0.05
    speed_bucket_mps: float = 0.5
    max_steer_rad: float = 0.6
    use_feedforward: bool = True


@dataclass
class LQROutput:
    delta_rad: float
    curvature: float
    state: np.ndarray
    gain: np.ndarray


class LQRLateralController:
    def __init__(self, params: VehicleParams | None = None, config: LQRConfig | None = None) -> None:
        self.params = params or VehicleParams()
        self.config = config or LQRConfig()
        self._cache: dict[int, np.ndarray] = {}
        self._prev: tuple[float, float] | None = None  # (e_y, e_psi)

    def reset(self) -> None:
        self._prev = None

    def gain(self, v: float) -> np.ndarray:
        c = self.config
        key = int(round(max(v, 1.0) / c.speed_bucket_mps))
        if key not in self._cache:
            A, B = error_dynamics(self.params, key * c.speed_bucket_mps)
            Ad, Bd = c2d_zoh(A, B, c.dt)
            K, _ = dlqr(Ad, Bd, np.diag(c.q_diag), np.array([[c.r]]))
            self._cache[key] = K.ravel()
        return self._cache[key]

    def feedforward(self, kappa: float, v: float, K: np.ndarray) -> float:
        """Rajamani eq. 3.12 with per-axle cornering stiffness (C = 2 C_alpha):
        delta_ff = L k + K_v a_y - k3 (lr k - lf m v^2 k / (C_r L)),  K_v = understeer gradient."""
        p = self.params
        m, L, lf, lr = p.mass_kg, p.wheelbase_m, p.cg_to_front_m, p.cg_to_rear_m
        cr = p.cornering_stiffness_rear
        ay = v * v * kappa
        return L * kappa + p.understeer_gradient * ay - K[2] * (lr * kappa - lf * m * v * v * kappa / (cr * L))

    def update(self, pose: Pose2D, v: float, yaw_rate: float, path: ReferencePath, dt: float) -> LQROutput:
        c = self.config
        proj = path.project(pose)
        e_y, e_psi = proj.cross_track, proj.heading_error
        if self._prev is None or dt <= 0:
            de_y = v * math.sin(e_psi)
        else:
            de_y = (e_y - self._prev[0]) / dt
        de_psi = yaw_rate - v * proj.curvature
        self._prev = (e_y, e_psi)
        x = np.array([e_y, de_y, e_psi, de_psi])
        K = self.gain(v)
        delta = -float(K @ x)
        if c.use_feedforward:
            delta += self.feedforward(proj.curvature, v, K)
        delta = max(-c.max_steer_rad, min(c.max_steer_rad, delta))
        # TODO(control): preview (feedforward from look-ahead curvature) and actuator delay augmentation.
        return LQROutput(delta, math.tan(delta) / self.params.wheelbase_m, x, K)
