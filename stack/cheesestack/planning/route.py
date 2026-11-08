"""Route planning: A* over a local waypoint graph, plus a route polyline helper.

The graph is defined here (not in mapping) so planning stays independently
buildable; the integration layer converts the map's lane graph into a
`WaypointGraph`. `lookahead_distance` ports the app's LookaheadScheduler
(Domain/Routing/LookaheadScheduler.swift) so short-lookahead pure pursuit
behaves the same way on both sides.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Hashable, Optional

import numpy as np

NodeId = Hashable


@dataclass
class RouteConfig:
    heuristic_weight: float = 1.0  # 1.0 = admissible A*; >1 = weighted A* (faster, suboptimal)
    max_expansions: int = 100_000


@dataclass
class Route:
    nodes: list
    points: np.ndarray  # (N, 2)
    cost: float

    @property
    def length(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    def cumulative(self) -> np.ndarray:
        if len(self.points) < 2:
            return np.zeros(len(self.points))
        return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))]

    def point_at(self, distance: float) -> tuple[np.ndarray, float]:
        """Interpolated point and heading (rad) at arc length `distance` (clamped)."""
        if len(self.points) == 1:
            return self.points[0].copy(), 0.0
        cum = self.cumulative()
        d = min(max(0.0, distance), cum[-1])
        i = int(np.clip(np.searchsorted(cum, d, side="right") - 1, 0, len(cum) - 2))
        span = max(cum[i + 1] - cum[i], 1e-9)
        t = (d - cum[i]) / span
        a, b = self.points[i], self.points[i + 1]
        return a + t * (b - a), math.atan2(b[1] - a[1], b[0] - a[0])

    def closest(self, p) -> tuple[float, float]:
        """(along-track distance, unsigned offset) of point p to the route."""
        p = np.asarray(p, dtype=float)
        cum = self.cumulative()
        best = (0.0, float("inf"))
        for i in range(len(self.points) - 1):
            a, b = self.points[i], self.points[i + 1]
            ab = b - a
            L2 = float(ab @ ab)
            t = 0.0 if L2 == 0 else float(np.clip((p - a) @ ab / L2, 0, 1))
            off = float(np.linalg.norm(p - (a + t * ab)))
            if off < best[1]:
                best = (cum[i] + t * math.sqrt(L2), off)
        return best


def lookahead_distance(speed_mps: float) -> float:
    """Port of the app's LookaheadScheduler: horizon 0.80 s at rest -> 0.25 s at 25 m/s,
    distance = max(8 m, v * horizon)."""
    v = max(0.0, speed_mps)
    horizon = 0.80 - min(1.0, v / 25.0) * 0.55
    return max(8.0, v * horizon)


class WaypointGraph:
    """Directed weighted graph of 2D waypoints."""

    def __init__(self) -> None:
        self.pos: dict = {}
        self.adj: dict = {}

    def add_node(self, nid: NodeId, x: float, y: float) -> None:
        self.pos[nid] = np.array([x, y], dtype=float)
        self.adj.setdefault(nid, {})

    def add_edge(self, a: NodeId, b: NodeId, cost: Optional[float] = None, bidirectional: bool = True) -> None:
        if a not in self.pos or b not in self.pos:
            raise KeyError("add nodes before edges")
        c = float(np.linalg.norm(self.pos[a] - self.pos[b])) if cost is None else float(cost)
        if c < 0:
            raise ValueError("negative edge cost")
        self.adj[a][b] = c
        if bidirectional:
            self.adj[b][a] = c

    def remove_edge(self, a: NodeId, b: NodeId, bidirectional: bool = True) -> None:
        self.adj.get(a, {}).pop(b, None)
        if bidirectional:
            self.adj.get(b, {}).pop(a, None)

    def nearest_node(self, x: float, y: float) -> NodeId:
        if not self.pos:
            raise ValueError("empty graph")
        ids = list(self.pos)
        P = np.stack([self.pos[i] for i in ids])
        return ids[int(np.argmin(np.linalg.norm(P - np.array([x, y]), axis=1)))]


class RoutePlanner:
    def __init__(self, graph: WaypointGraph, config: RouteConfig = RouteConfig()) -> None:
        self.graph = graph
        self.config = config

    def _h(self, a: NodeId, b: NodeId) -> float:
        return self.config.heuristic_weight * float(np.linalg.norm(self.graph.pos[a] - self.graph.pos[b]))

    def plan_nodes(self, start: NodeId, goal: NodeId) -> Optional[Route]:
        """A* from start to goal. Returns None when unreachable."""
        g = self.graph
        if start not in g.pos or goal not in g.pos:
            raise KeyError("unknown start/goal")
        counter = 0
        open_heap = [(self._h(start, goal), counter, start)]
        came: dict = {}
        cost = {start: 0.0}
        closed = set()
        expansions = 0
        while open_heap:
            _, _, u = heapq.heappop(open_heap)
            if u in closed:
                continue
            if u == goal:
                path = [u]
                while path[-1] in came:
                    path.append(came[path[-1]])
                path.reverse()
                return Route(path, np.stack([g.pos[n] for n in path]), cost[goal])
            closed.add(u)
            expansions += 1
            if expansions > self.config.max_expansions:
                return None
            for v, c in g.adj[u].items():
                nc = cost[u] + c
                if nc < cost.get(v, float("inf")):
                    cost[v] = nc
                    came[v] = u
                    counter += 1
                    heapq.heappush(open_heap, (nc + self._h(v, goal), counter, v))
        return None

    def plan(self, start_xy, goal_xy) -> Optional[Route]:
        """Plan between arbitrary points by snapping to nearest nodes; the endpoints
        are prepended/appended to the polyline."""
        s = self.graph.nearest_node(*start_xy)
        e = self.graph.nearest_node(*goal_xy)
        r = self.plan_nodes(s, e)
        if r is None:
            return None
        pts = [np.asarray(start_xy, float)] + list(r.points) + [np.asarray(goal_xy, float)]
        pts = np.stack(pts)
        keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-6]
        pts = pts[keep]
        extra = float(np.linalg.norm(pts[0] - r.points[0]) + np.linalg.norm(pts[-1] - r.points[-1]))
        return Route(r.nodes, pts, r.cost + extra)

    # TODO(planning): time-dependent edge costs (traffic), turn penalties, lane-level
    # graph with lane-change edges, incremental replanning (D* Lite) on edge closure.
