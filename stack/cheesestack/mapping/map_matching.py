"""Map matching: snap ego poses onto lanes.

* `MapMatcher`     single-pose: project onto every nearby lane, gate on distance and
                   heading, score with a Gaussian cost on (lateral offset, heading error).
* `HMMMapMatcher`  sequence: Viterbi over lane candidates per pose (Newson & Krumm 2009
                   style). Emission = single-pose cost; transition penalizes the
                   mismatch between along-map travel distance and Euclidean pose
                   displacement, plus a lane-change penalty.

Confidence for single-pose matches is a softmax over candidates that includes an
"off-map" null hypothesis at the gating boundary, so a lone far candidate is not
reported with confidence 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from cheesestack.core import Pose2D, wrap_angle

from .topology import LaneMap


@dataclass
class MapMatcherConfig:
    max_distance_m: float = 5.0
    max_heading_diff_rad: float = math.radians(60.0)
    sigma_lateral_m: float = 1.0
    sigma_heading_rad: float = math.radians(15.0)
    max_candidates: int = 8


@dataclass(frozen=True)
class MatchResult:
    lane_id: str
    s: float  # station along the lane
    d: float  # lateral offset, + left
    distance: float
    heading_error: float  # pose.yaw - lane heading, wrapped
    cost: float  # Mahalanobis-squared over (d, heading)
    confidence: float = 0.0


class MapMatcher:
    """Nearest-lane matching with heading gating.

    # TODO(mapping): spatial index for candidate lookup; use pose covariance
    # instead of fixed sigmas; lane-marking evidence from perception.
    """

    def __init__(self, lane_map: LaneMap, config: Optional[MapMatcherConfig] = None) -> None:
        self.lane_map = lane_map
        self.config = config or MapMatcherConfig()

    def candidates(self, pose: Pose2D) -> list[MatchResult]:
        cfg = self.config
        raw = []
        for lane in self.lane_map:
            pr = lane.centerline.project((pose.x, pose.y))
            if pr.distance > cfg.max_distance_m:
                continue
            he = wrap_angle(pose.yaw - pr.heading)
            if abs(he) > cfg.max_heading_diff_rad:
                continue
            cost = (pr.distance / cfg.sigma_lateral_m) ** 2 + (he / cfg.sigma_heading_rad) ** 2
            raw.append((cost, lane.id, pr, he))
        raw.sort(key=lambda r: r[0])
        raw = raw[: cfg.max_candidates]
        null_cost = (cfg.max_distance_m / cfg.sigma_lateral_m) ** 2
        if not raw:
            return []
        costs = np.array([r[0] for r in raw] + [null_cost])
        w = np.exp(-0.5 * (costs - costs.min()))
        conf = w[:-1] / w.sum()
        return [MatchResult(lid, pr.s, pr.d, pr.distance, he, float(c), float(cf))
                for (c, lid, pr, he), cf in zip(raw, conf)]

    def match(self, pose: Pose2D) -> Optional[MatchResult]:
        c = self.candidates(pose)
        return c[0] if c else None


@dataclass
class HMMMatcherConfig:
    matcher: MapMatcherConfig = field(default_factory=MapMatcherConfig)
    beta_m: float = 2.0  # transition scale: |route_dist - euclid| / beta (neg-log)
    lane_change_cost: float = 3.0  # neg-log penalty per lane change
    max_hops: int = 3  # successor search depth between consecutive poses
    backward_tolerance_m: float = 1.0  # allow small negative progress (noise)


class HMMMapMatcher:
    """Offline Viterbi map matching over a pose sequence.

    Poses with no candidates yield None and break the chain (decoding restarts
    after them).
    # TODO(mapping): online fixed-lag decoding, candidate transitions through
    # neighbor-then-successor combinations, speed-dependent beta.
    """

    def __init__(self, lane_map: LaneMap, config: Optional[HMMMatcherConfig] = None) -> None:
        self.lane_map = lane_map
        self.config = config or HMMMatcherConfig()
        self.single = MapMatcher(lane_map, self.config.matcher)

    # -------------------------------------------------------------- transitions
    def route_distance(self, a: MatchResult, b: MatchResult) -> Optional[tuple[float, int]]:
        """(along-map distance, lane changes) from candidate a to b, or None if infeasible."""
        cfg = self.config
        la = self.lane_map[a.lane_id]
        if a.lane_id == b.lane_id:
            ds = b.s - a.s
            return (max(ds, 0.0), 0) if ds >= -cfg.backward_tolerance_m else None
        if b.lane_id in (la.left_neighbor, la.right_neighbor):
            lb = self.lane_map[b.lane_id]
            s_on_b = lb.centerline.project(la.centerline.interpolate(a.s)).s
            ds = b.s - s_on_b
            if ds < -cfg.backward_tolerance_m:
                return None
            return math.hypot(max(ds, 0.0), 0.5 * (la.width + lb.width)), 1
        # bounded successor search (Dijkstra on small frontier)
        best: Optional[float] = None
        frontier = [(la.length - a.s, a.lane_id, 0)]
        while frontier:
            dist, lid, depth = frontier.pop()
            if depth >= cfg.max_hops:
                continue
            for sid in self.lane_map[lid].successors:
                if sid not in self.lane_map:
                    continue
                if sid == b.lane_id:
                    cand = dist + b.s
                    best = cand if best is None else min(best, cand)
                else:
                    frontier.append((dist + self.lane_map[sid].length, sid, depth + 1))
        return (best, 0) if best is not None else None

    def _transition_cost(self, a: MatchResult, b: MatchResult, euclid: float) -> float:
        rd = self.route_distance(a, b)
        if rd is None:
            return math.inf
        dist, lc = rd
        return abs(dist - euclid) / self.config.beta_m + lc * self.config.lane_change_cost

    # ---------------------------------------------------------------- decoding
    def match_sequence(self, poses: Sequence[Pose2D]) -> list[Optional[MatchResult]]:
        cands = [self.single.candidates(p) for p in poses]
        out: list[Optional[MatchResult]] = [None] * len(poses)
        t = 0
        while t < len(poses):
            if not cands[t]:
                t += 1
                continue
            # decode a maximal chain starting at t
            costs = [np.array([0.5 * c.cost for c in cands[t]])]
            back: list[np.ndarray] = [np.full(len(cands[t]), -1)]
            idx = [t]
            k = t + 1
            while k < len(poses) and cands[k]:
                prev, cur = cands[idx[-1]], cands[k]
                pp, pc = poses[idx[-1]], poses[k]
                euclid = math.hypot(pc.x - pp.x, pc.y - pp.y)
                T = np.array([[self._transition_cost(a, b, euclid) for b in cur] for a in prev])
                tot = costs[-1][:, None] + T
                bi = np.argmin(tot, axis=0)
                best = tot[bi, np.arange(len(cur))]
                if not np.isfinite(best).any():
                    break  # chain broken: restart decoding at k
                costs.append(best + np.array([0.5 * c.cost for c in cur]))
                back.append(bi)
                idx.append(k)
                k += 1
            j = int(np.argmin(costs[-1]))
            for step in range(len(idx) - 1, -1, -1):
                out[idx[step]] = cands[idx[step]][j]
                j = int(back[step][j])
            t = k
        return out
