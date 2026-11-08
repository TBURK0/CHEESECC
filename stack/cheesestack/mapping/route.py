"""Route representation: an ordered lane sequence with a stitched reference line and
Frenet (station s / lateral d) conversion.

A `Route` is a list of `RouteSegment(lane_id, s_start, s_end)`. Consecutive
segments are either successor lanes (segment ends at the lane end) or a lane
change (the source segment ends part-way, the target segment starts at the
projection of that point). The reference line is the concatenation of segment
centerline pieces; lane changes appear as a short lateral hop in it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from cheesestack.core import Pose2D, wrap_angle

from .topology import LaneMap, Polyline


@dataclass(frozen=True)
class RouteSegment:
    lane_id: str
    s_start: float
    s_end: float
    is_lane_change_from_prev: bool = False

    @property
    def length(self) -> float:
        return self.s_end - self.s_start


@dataclass
class RouteConfig:
    lane_change_fraction: float = 0.5  # where along the source lane a lane change happens
    min_segment_m: float = 0.5  # minimum kept length of a lane-change source segment
    frenet_window_m: float = 30.0  # +- search window around s_hint for projection


class Route:
    """Ordered lane sequence with Frenet conversion.

    # TODO(mapping): smooth lane-change transitions (spline blend instead of a
    # lateral hop), multi-lane corridor routes (any lane of a road), re-routing
    # hooks when the overlay version changes.
    """

    def __init__(self, segments: Sequence[RouteSegment], lane_map: LaneMap,
                 config: Optional[RouteConfig] = None) -> None:
        if not segments:
            raise ValueError("route needs at least one segment")
        self.config = config or RouteConfig()
        self.lane_map = lane_map
        self.segments: list[RouteSegment] = list(segments)
        pts: list[np.ndarray] = []
        first_idx: list[int] = []
        for seg in self.segments:
            piece = lane_map[seg.lane_id].centerline.sub(seg.s_start, seg.s_end).points
            if pts and np.linalg.norm(pts[-1] - piece[0]) < 1e-6:
                piece = piece[1:]
                first_idx.append(len(pts) - 1)
            else:
                first_idx.append(len(pts))
            pts.extend(piece)
        self.reference = Polyline(np.asarray(pts))
        self._seg_offset = np.array([self.reference.s[min(i, len(self.reference.s) - 1)] for i in first_idx])

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_lane_ids(cls, lane_map: LaneMap, lane_ids: Sequence[str], start_s: float = 0.0,
                      end_s: Optional[float] = None, config: Optional[RouteConfig] = None) -> "Route":
        """Build from a lane-id path (e.g. `RoadGraph.shortest_path(...).lanes`).

        A pair (u, v) where v is u's left/right neighbor is a lane change; any
        other pair must be a successor link.
        """
        cfg = config or RouteConfig()
        if not lane_ids:
            raise ValueError("empty lane path")
        segs: list[RouteSegment] = []
        cur_start = start_s
        is_lc = False
        for k, lid in enumerate(lane_ids):
            lane = lane_map[lid]
            last = k == len(lane_ids) - 1
            seg_end = lane.length if end_s is None or not last else min(end_s, lane.length)
            next_start = 0.0
            next_lc = False
            if not last:
                nxt = lane_ids[k + 1]
                if nxt in (lane.left_neighbor, lane.right_neighbor):
                    remaining = lane.length - cur_start
                    seg_end = cur_start + max(cfg.min_segment_m, cfg.lane_change_fraction * remaining)
                    seg_end = min(seg_end, lane.length)
                    next_start = lane_map[nxt].centerline.project(lane.centerline.interpolate(seg_end)).s
                    next_lc = True
                elif nxt not in lane.successors:
                    raise ValueError(f"{lid} -> {nxt} is neither a successor nor a neighbor")
            if seg_end - cur_start > 1e-6:
                segs.append(RouteSegment(lid, cur_start, seg_end, is_lc))
            cur_start, is_lc = next_start, next_lc
        return cls(segs, lane_map, cfg)

    # ----------------------------------------------------------------- queries
    @property
    def length(self) -> float:
        return self.reference.length

    @property
    def lane_ids(self) -> list[str]:
        return [s.lane_id for s in self.segments]

    def segment_index_at(self, s: float) -> int:
        return int(np.clip(np.searchsorted(self._seg_offset, s, side="right") - 1, 0, len(self.segments) - 1))

    def lane_at(self, s: float) -> tuple[str, float]:
        """(lane_id, station along that lane) for route station s."""
        i = self.segment_index_at(s)
        seg = self.segments[i]
        lane_s = float(np.clip(seg.s_start + (s - self._seg_offset[i]), seg.s_start, seg.s_end))
        return seg.lane_id, lane_s

    def segment_start_station(self, index: int) -> float:
        return float(self._seg_offset[index])

    def speed_limit_at(self, s: float) -> float:
        return self.lane_map[self.lane_at(s)[0]].speed_limit

    def heading_at(self, s: float) -> float:
        return float(self.reference.heading_at(s))

    def to_frenet(self, x: float, y: float, s_hint: Optional[float] = None) -> tuple[float, float]:
        """(s, d) of a map point; d > 0 left of the route. s extrapolates past the ends."""
        if s_hint is None:
            pr = self.reference.project((x, y), extrapolate=True)
        else:
            w = self.config.frenet_window_m
            pr = self.reference.project((x, y), s_min=s_hint - w, s_max=s_hint + w, extrapolate=True)
        return pr.s, pr.d

    def from_frenet(self, s: float, d: float) -> np.ndarray:
        """Map point for (s, d). Beyond the ends, extends along the end tangent."""
        L = self.length
        base = self.reference.interpolate(s)
        h = self.reference.heading_at(s)
        t = np.array([np.cos(h), np.sin(h)])
        if s < 0:
            base = base + t * s
        elif s > L:
            base = base + t * (s - L)
        return base + d * np.array([-t[1], t[0]])

    def pose_to_frenet(self, pose: Pose2D, s_hint: Optional[float] = None) -> tuple[float, float, float]:
        """(s, d, heading error) for a pose."""
        s, d = self.to_frenet(pose.x, pose.y, s_hint)
        return s, d, wrap_angle(pose.yaw - self.heading_at(s))

    def remaining_distance(self, pose_or_s) -> float:
        s = pose_or_s if isinstance(pose_or_s, (int, float)) else self.to_frenet(pose_or_s.x, pose_or_s.y)[0]
        return float(max(0.0, self.length - s))

    def distance_to_next_lane_change(self, s: float) -> Optional[float]:
        """Station distance to the next lane-change point ahead of s, or None."""
        for i in range(1, len(self.segments)):
            if self.segments[i].is_lane_change_from_prev and self._seg_offset[i] > s:
                return float(self._seg_offset[i] - s)
        return None

    def sample(self, ds: float = 1.0) -> np.ndarray:
        """(N, 4) array: s, x, y, heading along the reference line."""
        n = max(2, int(np.ceil(self.length / ds)) + 1)
        s = np.linspace(0.0, self.length, n)
        xy = self.reference.interpolate(s)
        return np.column_stack([s, xy, self.reference.heading_at(s)])
