"""Collision prediction between predicted trajectories using oriented boxes.

* ``OrientedBox`` + ``boxes_overlap``: separating-axis test (SAT) for two
  rectangles; also returns minimum penetration depth (0 if separated).
* ``CollisionChecker.check``: resamples two ``PredictedTrajectory`` onto a
  common time grid and returns the earliest time their footprints overlap
  (circle broad-phase, SAT narrow-phase).
* ``CollisionChecker.probability``: Monte-Carlo collision probability using
  each trajectory's per-step position covariance (one standard-normal draw per
  sample, scaled by every step's Cholesky factor, i.e. errors fully correlated
  in time - a reasonable model for track-state error). Seeded, deterministic.
* ``CollisionChecker.check_prediction``: over a multi-hypothesis object
  prediction, ``P = Σ_h p_h · P(collision | h)``.

Limits: yaw uncertainty is ignored in sampling; boxes are rigid rectangles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from cheesestack.core import BoundingBox3D

from .motion_models import PredictedTrajectory


@dataclass
class OrientedBox:
    x: float
    y: float
    yaw: float
    length: float
    width: float

    @classmethod
    def from_bbox3d(cls, b: BoundingBox3D) -> "OrientedBox":
        return cls(b.pose.x, b.pose.y, b.pose.yaw, b.length, b.width)

    def axes(self) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([[c, s], [-s, c]])

    def corners(self) -> np.ndarray:
        """(4, 2) corners, CCW starting front-left."""
        hl, hw = 0.5 * self.length, 0.5 * self.width
        local = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        R = np.array([[c, -s], [s, c]])
        return local @ R.T + np.array([self.x, self.y])

    @property
    def radius(self) -> float:
        return 0.5 * math.hypot(self.length, self.width)


def boxes_overlap(a: OrientedBox, b: OrientedBox, margin: float = 0.0):
    """SAT test. Returns (overlap: bool, penetration_depth: float). ``margin`` inflates both boxes."""
    if math.hypot(a.x - b.x, a.y - b.y) > a.radius + b.radius + 2 * margin:
        return False, 0.0
    ca, cb = a.corners(), b.corners()
    depth = math.inf
    for axis in np.vstack([a.axes(), b.axes()]):
        pa, pb = ca @ axis, cb @ axis
        overlap = min(pa.max(), pb.max()) - max(pa.min(), pb.min()) + 2 * margin
        if overlap <= 0.0:
            return False, 0.0
        depth = min(depth, overlap)
    return True, float(depth)


@dataclass
class CollisionCheckerConfig:
    dt: float = 0.1
    horizon: Optional[float] = None  # None: min of the two trajectories' horizons
    safety_margin: float = 0.0  # m, inflates both footprints
    n_samples: int = 200
    seed: int = 0


@dataclass
class CollisionResult:
    collides: bool
    time: float = math.inf  # earliest relative time of overlap
    index: int = -1
    penetration: float = 0.0
    probability: float = 0.0
    min_distance: float = math.inf  # min center distance over horizon


class CollisionChecker:
    def __init__(self, config: Optional[CollisionCheckerConfig] = None):
        self.cfg = config or CollisionCheckerConfig()

    def _grid(self, ta: PredictedTrajectory, tb: PredictedTrajectory) -> np.ndarray:
        h = min(ta.times[-1], tb.times[-1])
        if self.cfg.horizon is not None:
            h = min(h, self.cfg.horizon)
        n = int(math.floor(h / self.cfg.dt + 1e-9)) + 1
        return np.arange(n) * self.cfg.dt

    def _first_hit(self, pa, ya, pb, yb, dims_a, dims_b):
        for i in range(len(pa)):
            hit, depth = boxes_overlap(OrientedBox(pa[i, 0], pa[i, 1], ya[i], *dims_a),
                                       OrientedBox(pb[i, 0], pb[i, 1], yb[i], *dims_b), self.cfg.safety_margin)
            if hit:
                return i, depth
        return -1, 0.0

    def check(self, ta: PredictedTrajectory, dims_a, tb: PredictedTrajectory, dims_b,
              with_probability: bool = False) -> CollisionResult:
        """``dims_*`` = (length, width). Deterministic (mean) check, optional MC probability."""
        times = self._grid(ta, tb)
        ra, rb = ta.resample(times), tb.resample(times)
        i, depth = self._first_hit(ra.positions, ra.states[:, 2], rb.positions, rb.states[:, 2], dims_a, dims_b)
        dmin = float(np.min(np.linalg.norm(ra.positions - rb.positions, axis=1)))
        res = CollisionResult(i >= 0, float(times[i]) if i >= 0 else math.inf, i, depth,
                              1.0 if i >= 0 else 0.0, dmin)
        if with_probability:
            res.probability = self._mc_probability(ra, dims_a, rb, dims_b)
        return res

    def _mc_probability(self, ra: PredictedTrajectory, dims_a, rb: PredictedTrajectory, dims_b) -> float:
        rng = np.random.default_rng(self.cfg.seed)
        n = len(ra.times)
        # Relative position covariance (independent agents) -> one Cholesky per step.
        cov = ra.pos_cov + rb.pos_cov + 1e-9 * np.eye(2)
        L = np.linalg.cholesky(cov)  # (n, 2, 2)
        z = rng.standard_normal((self.cfg.n_samples, 2))
        rel_mean = rb.positions - ra.positions
        ra_r = max(dims_a[0], dims_a[1]) * 0.5 * math.sqrt(2)
        rb_r = max(dims_b[0], dims_b[1]) * 0.5 * math.sqrt(2)
        hits = 0
        for k in range(self.cfg.n_samples):
            rel = rel_mean + np.einsum("nij,j->ni", L, z[k])
            near = np.nonzero(np.linalg.norm(rel, axis=1) <= ra_r + rb_r + 2 * self.cfg.safety_margin)[0]
            for i in near:
                a = OrientedBox(0.0, 0.0, ra.states[i, 2], *dims_a)
                b = OrientedBox(rel[i, 0], rel[i, 1], rb.states[i, 2], *dims_b)
                if boxes_overlap(a, b, self.cfg.safety_margin)[0]:
                    hits += 1
                    break
        # TODO(prediction): importance sampling / analytic (Gaussian-over-Minkowski-sum) bound
        # for low-probability events; include yaw uncertainty.
        return hits / self.cfg.n_samples

    def check_prediction(self, ego: PredictedTrajectory, ego_dims, hypotheses, obj_dims) -> CollisionResult:
        """Aggregate over multi-hypothesis ``hypotheses`` (list of PredictedTrajectory)."""
        p_total, t_first, best = 0.0, math.inf, None
        dmin = math.inf
        for h in hypotheses:
            r = self.check(ego, ego_dims, h, obj_dims, with_probability=True)
            p_total += h.probability * r.probability
            dmin = min(dmin, r.min_distance)
            if r.collides and r.time < t_first:
                t_first, best = r.time, r
        if best is None:
            return CollisionResult(False, probability=float(min(p_total, 1.0)), min_distance=dmin)
        return CollisionResult(True, t_first, best.index, best.penetration, float(min(p_total, 1.0)), dmin)
