"""Path optimization: quadratic reference smoothing and Frenet lateral sampling.

1. `smooth_path` solves the unconstrained QP
       min_x  w_ref ||x - x_ref||^2 + w_smooth ||D2 x||^2 + w_length ||D1 x||^2
   independently per coordinate, with the endpoints pinned by a large weight.
   That's one sparse-structured (dense here) linear solve: (W + w_s D2'D2 + w_l D1'D1) x = W x_ref.
2. `LateralPathSampler` samples quintic lateral offsets d(s) in the Frenet frame of
   a reference path from the current (d, d', d'') to a set of end offsets and
   lengths, scores each for offset, smoothness and obstacle clearance, and returns
   the best as a Cartesian path. Spatial (not temporal) — the time-parameterised
   version is in `trajectory.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .common import QuinticPolynomial, ReferencePath


@dataclass
class SmoothingConfig:
    w_ref: float = 1.0
    w_smooth: float = 50.0
    w_length: float = 0.0
    w_anchor: float = 1e6  # endpoint pinning weight


def _diff_matrix(n: int, order: int) -> np.ndarray:
    D = np.eye(n)
    for _ in range(order):
        D = np.diff(D, axis=0)
    return D


def smooth_path(points, config: SmoothingConfig = SmoothingConfig()) -> np.ndarray:
    """Quadratic smoothing of an (N,2) polyline (assumes roughly uniform spacing)."""
    X = np.asarray(points, dtype=float).reshape(-1, 2)
    n = len(X)
    if n < 3:
        return X.copy()
    w = np.full(n, config.w_ref)
    w[0] = w[-1] = config.w_anchor
    D2 = _diff_matrix(n, 2)
    D1 = _diff_matrix(n, 1)
    H = np.diag(w) + config.w_smooth * D2.T @ D2 + config.w_length * D1.T @ D1
    return np.linalg.solve(H, w[:, None] * X)
    # TODO(planning): banded solver, curvature/corridor inequality constraints (needs a QP
    # solver such as OSQP), non-uniform spacing.


def resample_polyline(points, step: float) -> np.ndarray:
    P = np.asarray(points, dtype=float).reshape(-1, 2)
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    s = np.arange(0.0, cum[-1] + 1e-9, step)
    if s[-1] < cum[-1] - 1e-6:
        s = np.r_[s, cum[-1]]
    return np.stack([np.interp(s, cum, P[:, 0]), np.interp(s, cum, P[:, 1])], axis=1)


def path_curvature(points) -> np.ndarray:
    """Discrete curvature of a polyline via finite differences."""
    P = np.asarray(points, dtype=float)
    dx, dy = np.gradient(P[:, 0]), np.gradient(P[:, 1])
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    return (dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2) ** 1.5, 1e-9)


@dataclass
class LateralSamplingConfig:
    d_samples: tuple = (-3.6, -2.7, -1.8, -0.9, 0.0, 0.9, 1.8, 2.7, 3.6)
    lengths_m: tuple = (20.0, 30.0, 45.0)
    ds: float = 1.0
    d_max: float = 4.0  # drivable corridor half-width
    vehicle_radius: float = 1.2
    safety_margin: float = 0.5
    w_offset: float = 1.0
    w_smooth: float = 50.0
    w_clearance: float = 2.0
    w_length: float = 0.02  # prefer longer, gentler transitions


@dataclass
class LateralPath:
    s: np.ndarray
    d: np.ndarray
    xy: np.ndarray
    cost: float
    d_end: float
    length: float
    min_clearance: float


class LateralPathSampler:
    def __init__(self, ref: ReferencePath, config: LateralSamplingConfig = LateralSamplingConfig()) -> None:
        self.ref = ref
        self.config = config

    def sample(self, s0: float, d0: float, dd0: float = 0.0, ddd0: float = 0.0,
               obstacles_sd: Optional[np.ndarray] = None, preferred_d: float = 0.0) -> list[LateralPath]:
        """All candidate paths (feasible ones only), sorted by cost.

        obstacles_sd: (M, 3) array of (s, d, radius) static obstacles in Frenet.
        """
        c = self.config
        obs = np.zeros((0, 3)) if obstacles_sd is None else np.asarray(obstacles_sd, float).reshape(-1, 3)
        out = []
        for L in c.lengths_m:
            s_rel = np.arange(0.0, L + 1e-9, c.ds)
            for d1 in c.d_samples:
                q = QuinticPolynomial(d0, dd0, ddd0, d1, 0.0, 0.0, L)
                d = q.x(s_rel)
                if np.abs(d).max() > c.d_max:
                    continue
                s = s0 + s_rel
                clear = float("inf")
                if len(obs):
                    dist = np.hypot(s[:, None] - obs[None, :, 0], d[:, None] - obs[None, :, 1]) - obs[None, :, 2]
                    clear = float(dist.min()) - c.vehicle_radius
                    # Obstacles beyond the candidate end: the path holds d1.
                    beyond = obs[obs[:, 0] > s[-1]]
                    if len(beyond):
                        clear = min(clear, float((np.abs(beyond[:, 1] - d1) - beyond[:, 2]).min()) - c.vehicle_radius)
                if clear < c.safety_margin:
                    continue
                dpp = q.acc(s_rel)
                cost = (c.w_offset * abs(d1 - preferred_d)
                        + c.w_smooth * float(np.mean(dpp**2))
                        + (c.w_clearance / max(clear, 0.1) if np.isfinite(clear) else 0.0)
                        - c.w_length * L)
                x, y = self.ref.to_cartesian(s, d)
                out.append(LateralPath(s, d, np.stack([x, y], 1), cost, d1, L, clear))
        return sorted(out, key=lambda p: p.cost)

    def best(self, *args, **kwargs) -> Optional[LateralPath]:
        paths = self.sample(*args, **kwargs)
        return paths[0] if paths else None
