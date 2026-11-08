"""Road-boundary generation: offset centerlines, lane boundaries, drivable-corridor polygons.

* `offset_polyline`         miter offset of a polyline (+ left), with a miter limit.
* `lane_boundary_polylines` explicit lane boundaries if present, else centerline +- width/2.
* `generate_lane_boundaries` fills Lane.left/right_boundary for a whole map, typing them
  DASHED between same-direction neighbors and SOLID/CURB on the outside.
* `Corridor` / `build_corridor` polygon of the drivable area along a lane sequence or route.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .route import Route
from .topology import BoundaryType, Lane, LaneBoundary, LaneMap, Polyline


@dataclass
class BoundaryConfig:
    miter_limit: float = 3.0  # max offset stretch factor at sharp corners
    inner_type: BoundaryType = BoundaryType.DASHED  # between same-direction neighbors
    left_outer_type: BoundaryType = BoundaryType.SOLID  # e.g. center line (keep-right traffic)
    right_outer_type: BoundaryType = BoundaryType.CURB
    overwrite: bool = False  # replace boundaries already present on lanes


def offset_polyline(points, offset: float, miter_limit: float = 3.0) -> np.ndarray:
    """Offset an (N, 2) polyline laterally by `offset` (+ left) using miter joins.

    Interior vertices move along the bisector of adjacent segment normals by
    offset / cos(half turn angle), capped at `miter_limit * |offset|`.
    # TODO(mapping): remove self-intersections (loops) produced on the inside of
    # curves whose radius < |offset|; round joins.
    """
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(p) < 2:
        raise ValueError("need >= 2 points")
    seg = np.diff(p, axis=0)
    seg /= np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-12)
    nrm = np.column_stack([-seg[:, 1], seg[:, 0]])  # left normals per segment
    vn = np.empty_like(p)
    vn[0], vn[-1] = nrm[0], nrm[-1]
    if len(p) > 2:
        bis = nrm[:-1] + nrm[1:]
        bl = np.linalg.norm(bis, axis=1, keepdims=True)
        bis = np.where(bl > 1e-9, bis / np.maximum(bl, 1e-12), nrm[:-1])
        cos_half = np.einsum("ij,ij->i", bis, nrm[:-1])
        scale = 1.0 / np.maximum(cos_half, 1.0 / miter_limit)
        vn[1:-1] = bis * scale[:, None]
    return p + offset * vn


def lane_boundary_polylines(lane: Lane, miter_limit: float = 3.0) -> tuple[Polyline, Polyline]:
    """(left, right) boundary polylines of a lane, in the lane's travel direction."""
    c = lane.centerline.points
    left = lane.left_boundary.polyline if lane.left_boundary else Polyline(offset_polyline(c, lane.width / 2, miter_limit))
    right = lane.right_boundary.polyline if lane.right_boundary else Polyline(offset_polyline(c, -lane.width / 2, miter_limit))
    return left, right


def generate_lane_boundaries(lane_map: LaneMap, config: Optional[BoundaryConfig] = None) -> int:
    """Populate missing lane boundaries in-place. Returns number of boundaries created."""
    cfg = config or BoundaryConfig()
    created = 0
    for lane in lane_map:
        c = lane.centerline.points
        if lane.left_boundary is None or cfg.overwrite:
            t = cfg.inner_type if lane.left_neighbor in lane_map else cfg.left_outer_type
            lane.left_boundary = LaneBoundary(Polyline(offset_polyline(c, lane.width / 2, cfg.miter_limit)), t)
            created += 1
        if lane.right_boundary is None or cfg.overwrite:
            t = cfg.inner_type if lane.right_neighbor in lane_map else cfg.right_outer_type
            lane.right_boundary = LaneBoundary(Polyline(offset_polyline(c, -lane.width / 2, cfg.miter_limit)), t)
            created += 1
    return created


# ------------------------------------------------------------------- polygons


def polygon_area(poly) -> float:
    """Signed shoelace area (+ for CCW)."""
    p = np.asarray(poly, dtype=float).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def point_in_polygon(point, poly) -> bool:
    """Even-odd ray casting test."""
    x, y = float(point[0]), float(point[1])
    p = np.asarray(poly, dtype=float).reshape(-1, 2)
    xi, yi = p[:, 0], p[:, 1]
    xj, yj = np.roll(xi, 1), np.roll(yi, 1)
    cond = (yi > y) != (yj > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = (xj - xi) * (y - yi) / (yj - yi) + xi
    return bool(np.count_nonzero(cond & (x < xint)) % 2)


@dataclass
class Corridor:
    """Drivable corridor: left/right boundary chains (travel order) and closed polygon."""

    left: np.ndarray  # (N, 2)
    right: np.ndarray  # (M, 2)

    @property
    def polygon(self) -> np.ndarray:
        """CCW polygon: right chain forward, then left chain backward."""
        return np.vstack([self.right, self.left[::-1]])

    @property
    def area(self) -> float:
        return abs(polygon_area(self.polygon))

    def contains(self, point) -> bool:
        return point_in_polygon(point, self.polygon)


def build_corridor(lane_map: LaneMap, segments: Sequence[tuple[str, float, float]] | Sequence[str],
                   include_neighbors: bool = False, miter_limit: float = 3.0) -> Corridor:
    """Corridor along `segments` = [(lane_id, s0, s1), ...] or a list of lane ids (full lanes).

    With include_neighbors, the corridor spans from the leftmost to rightmost
    same-direction neighbor of each lane (the whole carriageway).
    """
    segs = [(s, 0.0, lane_map[s].length) if isinstance(s, str) else tuple(s) for s in segments]
    lefts, rights = [], []
    for lane_id, s0, s1 in segs:
        lane = lane_map[lane_id]
        lmost = lane_map.leftmost(lane_id) if include_neighbors else lane
        rmost = lane_map.rightmost(lane_id) if include_neighbors else lane
        for side_lane, idx, out in ((lmost, 0, lefts), (rmost, 1, rights)):
            bnd = lane_boundary_polylines(side_lane, miter_limit)[idx]
            # map station on the route lane to station on the side lane / boundary
            a = bnd.project(lane.centerline.interpolate(s0)).s
            b = bnd.project(lane.centerline.interpolate(s1)).s
            if b - a > 1e-6:
                out.append(bnd.sub(a, b).points)
    if not lefts or not rights:
        raise ValueError("corridor is empty")
    return Corridor(left=_chain(lefts), right=_chain(rights))


def corridor_from_route(route: Route, include_neighbors: bool = False, miter_limit: float = 3.0) -> Corridor:
    return build_corridor(route.lane_map, [(s.lane_id, s.s_start, s.s_end) for s in route.segments],
                          include_neighbors, miter_limit)


def _chain(pieces: list[np.ndarray]) -> np.ndarray:
    out = [pieces[0]]
    for p in pieces[1:]:
        if np.linalg.norm(out[-1][-1] - p[0]) < 1e-6:
            p = p[1:]
        if len(p):
            out.append(p)
    return np.vstack(out)
