"""Lane topology: polylines with arc-length parameterization, lanes, boundaries, and a lane map.

A `Lane` is a directed centerline (`Polyline`) plus width, speed limit and
topological links (successors / predecessors / same-direction left/right
neighbors). `LaneMap` is the container every other mapping module operates on.

Frames: all geometry is in the `map` frame (ENU, metres). Lateral offsets are
positive to the LEFT of the direction of travel.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

from cheesestack.core import wrap_angle


# --------------------------------------------------------------------------- polyline


@dataclass(frozen=True)
class Projection:
    """Result of projecting a point onto a polyline."""

    s: float  # arc length of the foot point (may be <0 or >length when extrapolating)
    d: float  # signed lateral offset, + left of travel direction
    distance: float  # unsigned Euclidean distance to the foot point
    point: np.ndarray  # foot point (2,)
    segment: int  # index of the segment containing the foot point
    heading: float  # tangent heading at the foot point, rad


class Polyline:
    """Piecewise-linear 2D curve parameterized by arc length `s` in [0, length].

    Consecutive duplicate vertices are dropped. Interpolation is linear between
    vertices; heading is piecewise constant per segment (use `curvature` module
    for smooth derivatives).
    """

    def __init__(self, points) -> None:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(pts) >= 2:
            keep = np.ones(len(pts), dtype=bool)
            keep[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9
            pts = pts[keep]
        if len(pts) < 2:
            raise ValueError("Polyline needs at least 2 distinct points")
        self.points: np.ndarray = pts
        self._seg = np.diff(pts, axis=0)
        self._seg_len = np.hypot(self._seg[:, 0], self._seg[:, 1])
        self.s: np.ndarray = np.concatenate([[0.0], np.cumsum(self._seg_len)])

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:
        return f"Polyline(n={len(self.points)}, length={self.length:.2f})"

    @property
    def length(self) -> float:
        return float(self.s[-1])

    @property
    def start(self) -> np.ndarray:
        return self.points[0]

    @property
    def end(self) -> np.ndarray:
        return self.points[-1]

    def _locate(self, s):
        s = np.clip(np.asarray(s, dtype=float), 0.0, self.length)
        i = np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self._seg) - 1)
        t = (s - self.s[i]) / self._seg_len[i]
        return i, t

    def interpolate(self, s) -> np.ndarray:
        """Point(s) at arc length s (clamped). Scalar -> (2,), array -> (N, 2)."""
        i, t = self._locate(s)
        out = self.points[i] + self._seg[i] * np.asarray(t)[..., None]
        return out

    def heading_at(self, s) -> float | np.ndarray:
        i, _ = self._locate(s)
        h = np.arctan2(self._seg[i, 1], self._seg[i, 0])
        return float(h) if np.ndim(h) == 0 else h

    def vertex_headings(self) -> np.ndarray:
        """Heading per vertex (segment heading, last vertex repeats last segment)."""
        h = np.arctan2(self._seg[:, 1], self._seg[:, 0])
        return np.concatenate([h, h[-1:]])

    # ------------------------------------------------------------- projection
    def project(self, point, s_min: Optional[float] = None, s_max: Optional[float] = None,
                extrapolate: bool = False) -> Projection:
        """Closest point on the polyline to `point`.

        `s_min`/`s_max` restrict the search to segments overlapping that window
        (use for loops / self-approaching routes). With `extrapolate=True`, a point
        before the start or past the end gets s<0 / s>length along the end segment's
        tangent instead of being clamped (needed for Frenet conversion).
        """
        p = np.asarray(point, dtype=float).reshape(2)
        a = self.points[:-1]
        ap = p - a
        L2 = self._seg_len ** 2
        t = np.clip(np.einsum("ij,ij->i", ap, self._seg) / L2, 0.0, 1.0)
        foot = a + self._seg * t[:, None]
        dist = np.hypot(*(p - foot).T)
        if s_min is not None or s_max is not None:
            lo = -np.inf if s_min is None else s_min
            hi = np.inf if s_max is None else s_max
            mask = (self.s[1:] >= lo) & (self.s[:-1] <= hi)
            if mask.any():
                dist = np.where(mask, dist, np.inf)
        i = int(np.argmin(dist))
        ti = float(t[i])
        if extrapolate and ((i == 0 and ti <= 0.0) or (i == len(self._seg) - 1 and ti >= 1.0)):
            ti = float(np.dot(ap[i], self._seg[i]) / L2[i])
        fp = a[i] + self._seg[i] * ti
        seg = self._seg[i]
        cross = seg[0] * ap[i, 1] - seg[1] * ap[i, 0]
        lateral = abs(cross) / self._seg_len[i] if extrapolate else float(np.hypot(*(p - fp)))
        sign = 1.0 if cross >= 0 else -1.0
        return Projection(
            s=float(self.s[i] + ti * self._seg_len[i]),
            d=sign * float(lateral),
            distance=float(np.hypot(*(p - fp))),
            point=fp,
            segment=i,
            heading=float(math.atan2(seg[1], seg[0])),
        )

    # --------------------------------------------------------------- derived
    def resample(self, ds: float) -> "Polyline":
        """Uniformly resampled copy (spacing <= ds, endpoints kept)."""
        n = max(2, int(math.ceil(self.length / ds)) + 1)
        return Polyline(self.interpolate(np.linspace(0.0, self.length, n)))

    def sub(self, s0: float, s1: float) -> "Polyline":
        """Section between arc lengths s0 < s1 (clamped)."""
        s0, s1 = max(0.0, s0), min(self.length, s1)
        if s1 - s0 < 1e-9:
            raise ValueError("empty sub-polyline")
        inner = self.points[(self.s > s0) & (self.s < s1)]
        return Polyline(np.vstack([self.interpolate(s0), inner, self.interpolate(s1)]))

    def reversed(self) -> "Polyline":
        return Polyline(self.points[::-1])

    def to_list(self, ndigits: int = 4) -> list[list[float]]:
        return [[round(float(x), ndigits), round(float(y), ndigits)] for x, y in self.points]


# ------------------------------------------------------------------------ lanes


class BoundaryType(enum.Enum):
    SOLID = "solid"
    DASHED = "dashed"
    DOUBLE_SOLID = "double_solid"
    CURB = "curb"
    VIRTUAL = "virtual"  # no physical marking (e.g. inside intersections)

    @property
    def allows_crossing(self) -> bool:
        return self in (BoundaryType.DASHED, BoundaryType.VIRTUAL)


@dataclass
class LaneBoundary:
    polyline: Polyline
    type: BoundaryType = BoundaryType.SOLID

    @property
    def allows_crossing(self) -> bool:
        return self.type.allows_crossing


@dataclass
class Lane:
    """One directed lane.

    Neighbors are same-direction lanes only (lane-change targets).
    # TODO(mapping): opposing-lane neighbors, per-vertex width, lane type
    # (bike/bus/shoulder), turn restrictions, stop lines and signal associations.
    """

    id: str
    centerline: Polyline
    width: float = 3.5
    speed_limit: float = 13.89  # m/s (50 km/h)
    successors: list[str] = field(default_factory=list)
    predecessors: list[str] = field(default_factory=list)
    left_neighbor: Optional[str] = None
    right_neighbor: Optional[str] = None
    left_boundary: Optional[LaneBoundary] = None
    right_boundary: Optional[LaneBoundary] = None
    road_id: str = ""
    attributes: dict = field(default_factory=dict)

    @property
    def length(self) -> float:
        return self.centerline.length

    def can_change_left(self) -> bool:
        return self.left_neighbor is not None and (self.left_boundary is None or self.left_boundary.allows_crossing)

    def can_change_right(self) -> bool:
        return self.right_neighbor is not None and (self.right_boundary is None or self.right_boundary.allows_crossing)


@dataclass
class LaneMapConfig:
    connect_tolerance_m: float = 0.5  # max gap between a lane end and its successor start
    neighbor_tolerance_m: float = 1.5  # |lateral spacing - mean width| tolerance in validate()


class LaneMap:
    """Container of lanes with topology queries.

    Spatial queries are brute force over all lanes.
    # TODO(mapping): R-tree / grid spatial index for city-scale maps; tiling and
    # lazy loading by region.
    """

    def __init__(self, lanes: Optional[list[Lane]] = None, config: Optional[LaneMapConfig] = None,
                 name: str = "") -> None:
        self.config = config or LaneMapConfig()
        self.name = name
        self.lanes: dict[str, Lane] = {}
        for lane in lanes or []:
            self.add_lane(lane)
        if lanes:
            self.rebuild_predecessors()

    # ------------------------------------------------------------------ dict-ish
    def add_lane(self, lane: Lane) -> Lane:
        if lane.id in self.lanes:
            raise KeyError(f"duplicate lane id {lane.id!r}")
        self.lanes[lane.id] = lane
        return lane

    def remove_lane(self, lane_id: str) -> Lane:
        lane = self.lanes.pop(lane_id)
        for other in self.lanes.values():
            if lane_id in other.successors:
                other.successors.remove(lane_id)
            if lane_id in other.predecessors:
                other.predecessors.remove(lane_id)
            if other.left_neighbor == lane_id:
                other.left_neighbor = None
            if other.right_neighbor == lane_id:
                other.right_neighbor = None
        return lane

    def __getitem__(self, lane_id: str) -> Lane:
        return self.lanes[lane_id]

    def get(self, lane_id: str) -> Optional[Lane]:
        return self.lanes.get(lane_id)

    def __contains__(self, lane_id: object) -> bool:
        return lane_id in self.lanes

    def __iter__(self) -> Iterator[Lane]:
        return iter(self.lanes.values())

    def __len__(self) -> int:
        return len(self.lanes)

    def ids(self) -> list[str]:
        return list(self.lanes.keys())

    # ------------------------------------------------------------------ topology
    def link(self, from_id: str, to_id: str) -> None:
        a, b = self.lanes[from_id], self.lanes[to_id]
        if to_id not in a.successors:
            a.successors.append(to_id)
        if from_id not in b.predecessors:
            b.predecessors.append(from_id)

    def set_neighbors(self, left_id: str, right_id: str) -> None:
        """Declare `left_id` to be the left neighbor of `right_id` (and vice versa)."""
        self.lanes[right_id].left_neighbor = left_id
        self.lanes[left_id].right_neighbor = right_id

    def rebuild_predecessors(self) -> None:
        """Recompute predecessor lists from successor lists (successors are authoritative)."""
        for lane in self.lanes.values():
            lane.predecessors = []
        for lane in self.lanes.values():
            for sid in lane.successors:
                if sid in self.lanes and lane.id not in self.lanes[sid].predecessors:
                    self.lanes[sid].predecessors.append(lane.id)

    def successors(self, lane_id: str) -> list[Lane]:
        return [self.lanes[i] for i in self.lanes[lane_id].successors if i in self.lanes]

    def predecessors(self, lane_id: str) -> list[Lane]:
        return [self.lanes[i] for i in self.lanes[lane_id].predecessors if i in self.lanes]

    def left_of(self, lane_id: str) -> Optional[Lane]:
        n = self.lanes[lane_id].left_neighbor
        return self.lanes.get(n) if n else None

    def right_of(self, lane_id: str) -> Optional[Lane]:
        n = self.lanes[lane_id].right_neighbor
        return self.lanes.get(n) if n else None

    def leftmost(self, lane_id: str) -> Lane:
        lane, seen = self.lanes[lane_id], {lane_id}
        while lane.left_neighbor and lane.left_neighbor in self.lanes and lane.left_neighbor not in seen:
            seen.add(lane.left_neighbor)
            lane = self.lanes[lane.left_neighbor]
        return lane

    def rightmost(self, lane_id: str) -> Lane:
        lane, seen = self.lanes[lane_id], {lane_id}
        while lane.right_neighbor and lane.right_neighbor in self.lanes and lane.right_neighbor not in seen:
            seen.add(lane.right_neighbor)
            lane = self.lanes[lane.right_neighbor]
        return lane

    # ------------------------------------------------------------------ spatial
    def nearest_lanes(self, point, radius: float = 10.0) -> list[tuple[Lane, Projection]]:
        """Lanes whose centerline passes within `radius` of `point`, nearest first."""
        out = []
        for lane in self.lanes.values():
            pr = lane.centerline.project(point)
            if pr.distance <= radius:
                out.append((lane, pr))
        out.sort(key=lambda lp: lp[1].distance)
        return out

    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) over all centerline vertices."""
        pts = np.vstack([lane.centerline.points for lane in self.lanes.values()])
        return (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))

    # ---------------------------------------------------------------- validation
    def validate(self) -> list[str]:
        """Return a list of human-readable problems (empty list == consistent)."""
        errs: list[str] = []
        tol = self.config.connect_tolerance_m
        for lane in self.lanes.values():
            for sid in lane.successors:
                if sid not in self.lanes:
                    errs.append(f"{lane.id}: unknown successor {sid}")
                    continue
                gap = float(np.linalg.norm(lane.centerline.end - self.lanes[sid].centerline.start))
                if gap > tol:
                    errs.append(f"{lane.id}->{sid}: endpoint gap {gap:.2f} m > {tol} m")
                if lane.id not in self.lanes[sid].predecessors:
                    errs.append(f"{sid}: missing predecessor {lane.id}")
            for pid in lane.predecessors:
                if pid not in self.lanes:
                    errs.append(f"{lane.id}: unknown predecessor {pid}")
            for side, nid, back in (("left", lane.left_neighbor, "right_neighbor"),
                                    ("right", lane.right_neighbor, "left_neighbor")):
                if nid is None:
                    continue
                if nid not in self.lanes:
                    errs.append(f"{lane.id}: unknown {side} neighbor {nid}")
                    continue
                other = self.lanes[nid]
                if getattr(other, back) != lane.id:
                    errs.append(f"{lane.id}: {side} neighbor {nid} is not reciprocal")
                mid = lane.centerline.interpolate(lane.length / 2)
                pr = other.centerline.project(mid)
                expect = 0.5 * (lane.width + other.width)
                d = -pr.d if side == "left" else pr.d  # neighbor on left => we are on its right (d<0)
                if abs(d - expect) > self.config.neighbor_tolerance_m:
                    errs.append(f"{lane.id}: {side} neighbor {nid} at lateral {d:.2f} m, expected ~{expect:.2f} m")
                if abs(wrap_angle(pr.heading - lane.centerline.heading_at(lane.length / 2))) > math.pi / 2:
                    errs.append(f"{lane.id}: {side} neighbor {nid} runs in the opposite direction")
        return errs
