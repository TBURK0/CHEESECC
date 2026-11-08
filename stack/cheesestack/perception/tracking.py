"""Multi-object tracking: constant-velocity Kalman tracks + Hungarian assignment.

Each track filters state (x, y, vx, vy) in a planar metric frame with a
constant-velocity model driven by white acceleration noise. Every cycle:

1. predict all tracks to the measurement time,
2. build a cost matrix of squared Mahalanobis distances (innovation
   covariance S = H P H^T + R), gated by a chi-square threshold,
3. solve the assignment with the Hungarian algorithm (:func:`hungarian`,
   pure numpy, rectangular OK),
4. update matched tracks, age unmatched ones, spawn tracks from unmatched
   measurements.

Lifecycle: TENTATIVE --(n_init hits)--> CONFIRMED --(miss)--> LOST
--(hit)--> CONFIRMED; LOST tracks are deleted after `max_misses`, a
TENTATIVE track is deleted on its first miss.

Input measurements are core.Detection with `box3d` (pose in the tracking
frame), optionally with `attributes["covariance"]` (2x2). Output is a list
of core.TrackedObject.
"""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from cheesestack.core import Detection, Header, ObjectClass, Pose2D, TrackedObject, wrap_angle

INF_COST = 1e9


def hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost assignment (Kuhn-Munkres, O(n^3)). Works on rectangular matrices.

    Returns (row, col) pairs; each row/col used at most once and
    min(n_rows, n_cols) pairs are returned. Use a large finite cost for
    forbidden pairs and filter them afterwards.
    """
    cost = np.asarray(cost, dtype=float)
    if cost.size == 0:
        return []
    transposed = cost.shape[0] > cost.shape[1]
    a = cost.T if transposed else cost
    n, m = a.shape  # n <= m
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)  # p[j] = row (1-based) assigned to col j
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = a[i0 - 1, :] - u[i0] - v[1:]
            free = ~used[1:]
            upd = free & (cur < minv[1:])
            minv[1:][upd] = cur[upd]
            way[1:][upd] = j0
            cand = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            used_idx = np.nonzero(used)[0]
            u[p[used_idx]] += delta
            v[used_idx] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]
    if transposed:
        pairs = [(c, r) for r, c in pairs]
    return sorted(pairs)


class TrackState(enum.Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"


@dataclass
class TrackerConfig:
    accel_noise: float = 2.0  # m/s^2 std of white acceleration
    meas_std: float = 0.5  # default position measurement std (m)
    init_vel_std: float = 5.0
    gate_chi2: float = 9.21  # 2 dof, 99%
    n_init: int = 3
    max_misses: int = 5
    class_mismatch_penalty: float = 4.0  # added to cost when classes differ (both known)
    frame_id: str = "odom"


@dataclass
class Track:
    track_id: int
    x: np.ndarray  # (4,)
    P: np.ndarray  # (4,4)
    stamp: float
    cls: ObjectClass = ObjectClass.UNKNOWN
    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    misses: int = 0
    age: int = 0
    score: float = 0.5
    length: float = 4.5
    width: float = 1.8
    yaw: float = 0.0
    class_votes: dict = field(default_factory=dict)


def cv_matrices(dt: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Constant-velocity transition F and discrete white-accel process noise Q."""
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
    q1 = np.array([[dt4 / 4, dt3 / 2], [dt3 / 2, dt2]]) * q * q
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = q1
    Q[np.ix_([1, 3], [1, 3])] = q1
    return F, Q


_H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])


class MultiObjectTracker:
    def __init__(self, config: Optional[TrackerConfig] = None) -> None:
        self.config = config or TrackerConfig()
        self.tracks: list[Track] = []
        self._ids = itertools.count(1)
        self.stamp: Optional[float] = None

    # -- helpers --------------------------------------------------------
    def _meas(self, d: Detection) -> tuple[np.ndarray, np.ndarray]:
        if d.box3d is None:
            raise ValueError("tracker needs Detection.box3d (metric pose)")
        z = np.array([d.box3d.pose.x, d.box3d.pose.y])
        R = d.attributes.get("covariance")
        R = np.asarray(R, float)[:2, :2] if R is not None else np.eye(2) * self.config.meas_std ** 2
        return z, R

    def predict(self, stamp: float) -> None:
        for t in self.tracks:
            dt = stamp - t.stamp
            if dt <= 0:
                continue
            F, Q = cv_matrices(dt, self.config.accel_noise)
            t.x = F @ t.x
            t.P = F @ t.P @ F.T + Q
            t.stamp = stamp

    def _new_track(self, d: Detection, stamp: float) -> Track:
        z, R = self._meas(d)
        P = np.diag([R[0, 0], R[1, 1], self.config.init_vel_std ** 2, self.config.init_vel_std ** 2])
        P[:2, :2] = R
        b = d.box3d
        tr = Track(next(self._ids), np.array([z[0], z[1], 0.0, 0.0]), P, stamp, d.cls,
                   score=d.score, length=b.length, width=b.width, yaw=b.pose.yaw)
        tr.class_votes[d.cls] = d.score
        return tr

    def _update(self, t: Track, d: Detection) -> None:
        z, R = self._meas(d)
        y = z - _H @ t.x
        S = _H @ t.P @ _H.T + R
        K = t.P @ _H.T @ np.linalg.inv(S)
        t.x = t.x + K @ y
        I_KH = np.eye(4) - K @ _H
        t.P = I_KH @ t.P @ I_KH.T + K @ R @ K.T  # Joseph form
        t.hits += 1
        t.misses = 0
        t.score = 0.7 * t.score + 0.3 * d.score
        t.class_votes[d.cls] = t.class_votes.get(d.cls, 0.0) + d.score
        known = {c: s for c, s in t.class_votes.items() if c != ObjectClass.UNKNOWN}
        t.cls = max(known, key=known.get) if known else ObjectClass.UNKNOWN
        b = d.box3d
        t.length = 0.8 * t.length + 0.2 * b.length
        t.width = 0.8 * t.width + 0.2 * b.width
        t.yaw = b.pose.yaw
        if t.state == TrackState.TENTATIVE and t.hits >= self.config.n_init:
            t.state = TrackState.CONFIRMED
        elif t.state == TrackState.LOST:
            t.state = TrackState.CONFIRMED

    def cost_matrix(self, dets: Sequence[Detection]) -> np.ndarray:
        cfg = self.config
        C = np.full((len(self.tracks), len(dets)), INF_COST)
        meas = [self._meas(d) for d in dets]
        for i, t in enumerate(self.tracks):
            for j, (z, R) in enumerate(meas):
                S = _H @ t.P @ _H.T + R
                y = z - _H @ t.x
                d2 = float(y @ np.linalg.solve(S, y))
                if d2 > cfg.gate_chi2:
                    continue
                c = dets[j].cls
                if c != ObjectClass.UNKNOWN and t.cls != ObjectClass.UNKNOWN and c != t.cls:
                    d2 += cfg.class_mismatch_penalty
                C[i, j] = d2
        return C

    # -- main -----------------------------------------------------------
    def update(self, detections: Sequence[Detection], stamp: float) -> list[TrackedObject]:
        """Run one tracking cycle; returns confirmed (and lost-but-coasting) tracks."""
        cfg = self.config
        dets = [d for d in detections if d.box3d is not None]
        self.predict(stamp)
        C = self.cost_matrix(dets)
        matched_t, matched_d = set(), set()
        for i, j in hungarian(C):
            if C[i, j] >= INF_COST:
                continue
            self._update(self.tracks[i], dets[j])
            matched_t.add(i)
            matched_d.add(j)
        survivors = []
        for i, t in enumerate(self.tracks):
            t.age += 1
            if i not in matched_t:
                t.misses += 1
                if t.state == TrackState.TENTATIVE:
                    continue
                t.state = TrackState.LOST
                if t.misses > cfg.max_misses:
                    continue
            survivors.append(t)
        for j, d in enumerate(dets):
            if j not in matched_d:
                survivors.append(self._new_track(d, stamp))
        self.tracks = survivors
        self.stamp = stamp
        return self.outputs()

    def outputs(self, include_lost: bool = True) -> list[TrackedObject]:
        out = []
        for t in self.tracks:
            if t.state == TrackState.TENTATIVE or (t.state == TrackState.LOST and not include_lost):
                continue
            speed = float(np.hypot(t.x[2], t.x[3]))
            yaw = float(np.arctan2(t.x[3], t.x[2])) if speed > 0.5 else t.yaw
            conf = float(np.clip(t.score * (0.5 ** t.misses), 0.0, 1.0))
            out.append(TrackedObject(
                track_id=t.track_id,
                header=Header(stamp=t.stamp, frame_id=self.config.frame_id, source="perception.tracking"),
                cls=t.cls, pose=Pose2D(float(t.x[0]), float(t.x[1]), wrap_angle(yaw)),
                velocity=t.x[2:4].copy(), length=t.length, width=t.width, confidence=conf,
                covariance=t.P.copy(), age=t.age))
        return out
        # TODO(perception): ego-motion compensation when tracking in base_link,
        # TODO(perception): CTRV/IMM models, appearance embeddings in the cost, track-score (log-likelihood) lifecycle.
