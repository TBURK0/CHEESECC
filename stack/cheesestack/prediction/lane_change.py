"""Lane-change prediction from lateral offset / lateral velocity features.

Lanes are plain centerline polylines (``Lane``) since prediction may not import
the mapping layer. The pipeline:

* ``project_to_lane``: Frenet projection -> (s, d, lane heading); d > 0 = left.
* ``extract_lane_change_features``: d normalised by half-width, lateral velocity
  ``d_dot = v * sin(yaw - lane_heading)``, relative heading, and inverse
  time-to-line-crossing.
* ``LaneChangeModel``: interface returning [p_keep, p_left, p_right].
  ``LogisticLaneChangeModel`` is a 3-class softmax-logistic baseline with
  hand-set symmetric weights and a numpy gradient-descent ``fit``.
* ``LaneChangePredictor``: finds the current lane and its neighbours among the
  given lanes and moves probability mass off sides that have no lane.

Limits: single-frame features (no temporal filtering), no lane-marking types
(solid lines), no traffic context (gap acceptance).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from cheesestack.core import TrackedObject, wrap_angle

KEEP, LEFT, RIGHT = 0, 1, 2
FEATURE_NAMES = ("d_norm", "d_dot", "rel_heading", "inv_tlc")


@dataclass
class Lane:
    """A lane as a centerline polyline (N, 2) in the map frame, ordered along travel."""

    centerline: np.ndarray
    width: float = 3.5
    lane_id: str = ""

    def __post_init__(self):
        self.centerline = np.asarray(self.centerline, dtype=float).reshape(-1, 2)
        if len(self.centerline) < 2:
            raise ValueError("Lane centerline needs at least 2 points")


@dataclass
class LaneProjection:
    s: float  # arc length along centerline
    d: float  # signed lateral offset, + left
    heading: float  # centerline heading at projection
    segment: int


def project_to_lane(lane: Lane, xy) -> LaneProjection:
    """Closest-point projection onto the polyline (extrapolates past both ends)."""
    p = np.asarray(xy, dtype=float)
    a, b = lane.centerline[:-1], lane.centerline[1:]
    ab = b - a
    L2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12)
    t = np.einsum("ij,ij->i", p - a, ab) / L2
    n = len(ab)
    lo = np.zeros(n)
    hi = np.ones(n)
    lo[0], hi[-1] = -np.inf, np.inf  # extrapolate before the start and past the end
    t_clamped = np.clip(t, lo, hi)
    closest = a + ab * t_clamped[:, None]
    dist = np.linalg.norm(closest - p, axis=1)
    i = int(np.argmin(dist))
    seg_len = np.sqrt(L2)
    s = float(np.sum(seg_len[:i]) + t_clamped[i] * seg_len[i])
    heading = math.atan2(ab[i, 1], ab[i, 0])
    rel = p - a[i]
    cross = ab[i, 0] * rel[1] - ab[i, 1] * rel[0]
    d = float(cross / seg_len[i])
    return LaneProjection(s, d, heading, i)


@dataclass
class LaneChangeFeatures:
    d: float
    d_norm: float
    d_dot: float
    rel_heading: float
    inv_tlc: float  # 1 / time-to-line-crossing toward the side we drift to, signed (+ left)

    def as_array(self) -> np.ndarray:
        return np.array([self.d_norm, self.d_dot, self.rel_heading, self.inv_tlc])


def extract_lane_change_features(lane: Lane, x: float, y: float, yaw: float, v: float,
                                 tlc_cap: float = 10.0) -> LaneChangeFeatures:
    proj = project_to_lane(lane, (x, y))
    half = 0.5 * lane.width
    rel = wrap_angle(yaw - proj.heading)
    d_dot = v * math.sin(rel)
    if abs(d_dot) > 1e-3:
        dist_to_line = (half - proj.d) if d_dot > 0 else (half + proj.d)
        tlc = max(dist_to_line, 0.0) / abs(d_dot)
        inv_tlc = math.copysign(1.0 / max(tlc, 1.0 / tlc_cap), d_dot)
    else:
        inv_tlc = 0.0
    return LaneChangeFeatures(proj.d, proj.d / half, d_dot, rel, inv_tlc)


class LaneChangeModel(ABC):
    """Maps a (F,) feature vector to class probabilities [keep, left, right]."""

    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        ...


@dataclass
class LogisticLaneChangeConfig:
    # Weights over (d_norm, d_dot, rel_heading, inv_tlc) for the LEFT logit; RIGHT mirrors them.
    weights: tuple = (2.5, 2.0, 4.0, 1.0)
    bias: float = -3.0


class LogisticLaneChangeModel(LaneChangeModel):
    """Multinomial logistic (softmax) model; KEEP logit fixed at 0.

    Weight matrix ``W`` is (3, F+1) with last column the bias. Default weights
    are symmetric: LEFT uses +features, RIGHT uses -features.
    """

    def __init__(self, config: Optional[LogisticLaneChangeConfig] = None):
        self.cfg = config or LogisticLaneChangeConfig()
        w = np.asarray(self.cfg.weights, dtype=float)
        self.W = np.zeros((3, len(w) + 1))
        self.W[LEFT, :-1], self.W[LEFT, -1] = w, self.cfg.bias
        self.W[RIGHT, :-1], self.W[RIGHT, -1] = -w, self.cfg.bias

    def logits(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        Xb = np.hstack([X, np.ones((len(X), 1))])
        return Xb @ self.W.T

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        z = self.logits(features)
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return p[0] if np.ndim(features) == 1 else p

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.5, iters: int = 500, l2: float = 1e-3) -> "LogisticLaneChangeModel":
        """Batch gradient descent on cross-entropy. ``y`` holds class ids 0/1/2."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=int)
        Y = np.eye(3)[y]
        Xb = np.hstack([X, np.ones((len(X), 1))])
        for _ in range(iters):
            z = Xb @ self.W.T
            z -= z.max(axis=1, keepdims=True)
            P = np.exp(z)
            P /= P.sum(axis=1, keepdims=True)
            G = (P - Y).T @ Xb / len(X) + l2 * self.W
            G[KEEP] = 0.0  # keep KEEP logit as the 0 reference
            self.W -= lr * G
        return self


@dataclass
class LaneChangePrediction:
    current_lane: Optional[str]
    p_keep: float
    p_left: float
    p_right: float
    left_lane: Optional[str] = None
    right_lane: Optional[str] = None
    features: Optional[LaneChangeFeatures] = None

    @property
    def most_likely(self) -> str:
        return ("keep", "left", "right")[int(np.argmax([self.p_keep, self.p_left, self.p_right]))]


@dataclass
class LaneChangePredictorConfig:
    max_lane_offset_factor: float = 1.0  # object must be within factor*width of a lane center
    neighbor_tolerance: float = 0.5  # |center spacing - width| / width tolerance for neighbours
    heading_gate: float = math.radians(60)  # ignore lanes going a different direction


class LaneChangePredictor:
    """Lane-change probabilities for an object given candidate lanes."""

    def __init__(self, config: Optional[LaneChangePredictorConfig] = None,
                 model: Optional[LaneChangeModel] = None):
        self.cfg = config or LaneChangePredictorConfig()
        self.model = model or LogisticLaneChangeModel()

    def _neighbors(self, lanes: Sequence[Lane], cur: int, x: float, y: float, yaw: float):
        cur_lane = lanes[cur]
        cur_proj = project_to_lane(cur_lane, (x, y))
        left = right = None
        best_l = best_r = math.inf
        for j, ln in enumerate(lanes):
            if j == cur:
                continue
            pr = project_to_lane(ln, (x, y))
            if abs(wrap_angle(pr.heading - cur_proj.heading)) > self.cfg.heading_gate:
                continue
            offset = cur_proj.d - pr.d  # lane j center relative to current center, + = left
            spacing = 0.5 * (cur_lane.width + ln.width)
            err = abs(abs(offset) - spacing) / spacing
            if err > self.cfg.neighbor_tolerance:
                continue
            if offset > 0 and err < best_l:
                left, best_l = j, err
            elif offset < 0 and err < best_r:
                right, best_r = j, err
        return left, right

    def predict_state(self, lanes: Sequence[Lane], x: float, y: float, yaw: float, v: float) -> LaneChangePrediction:
        if not lanes:
            return LaneChangePrediction(None, 1.0, 0.0, 0.0)
        cands = []
        for i, ln in enumerate(lanes):
            pr = project_to_lane(ln, (x, y))
            if abs(wrap_angle(yaw - pr.heading)) > self.cfg.heading_gate and v > 0.5:
                continue
            if abs(pr.d) <= self.cfg.max_lane_offset_factor * ln.width:
                cands.append((abs(pr.d), i))
        if not cands:
            return LaneChangePrediction(None, 1.0, 0.0, 0.0)
        cur = min(cands)[1]
        feats = extract_lane_change_features(lanes[cur], x, y, yaw, v)
        p = np.asarray(self.model.predict_proba(feats.as_array()), dtype=float).copy()
        li, ri = self._neighbors(lanes, cur, x, y, yaw)
        if li is None:
            p[KEEP] += p[LEFT]
            p[LEFT] = 0.0
        if ri is None:
            p[KEEP] += p[RIGHT]
            p[RIGHT] = 0.0
        # TODO(prediction): temporal smoothing (HMM over frames), lane-marking legality,
        # and gap-acceptance features from surrounding traffic.
        return LaneChangePrediction(lanes[cur].lane_id, float(p[KEEP]), float(p[LEFT]), float(p[RIGHT]),
                                    lanes[li].lane_id if li is not None else None,
                                    lanes[ri].lane_id if ri is not None else None, feats)

    def predict(self, obj: TrackedObject, lanes: Sequence[Lane]) -> LaneChangePrediction:
        vx, vy = float(obj.velocity[0]), float(obj.velocity[1])
        v = math.hypot(vx, vy)
        yaw = math.atan2(vy, vx) if v > 0.3 else obj.pose.yaw
        return self.predict_state(lanes, obj.pose.x, obj.pose.y, yaw, v)
