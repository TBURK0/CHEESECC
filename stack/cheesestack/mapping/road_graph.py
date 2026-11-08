"""Directed lane-level road graph and shortest-path search (Dijkstra / A*, stdlib heapq).

Nodes are lane ids. Edges:
  * SUCCESSOR        u -> v   cost = length(u)            (drive to the end of u)
  * LANE_CHANGE_L/R  u -> v   cost = penalty + |start(u) - start(v)|

Cost is in metres (`cost_mode="distance"`) or seconds (`"time"`, metres divided by
the source lane's speed limit). The A* heuristic is the straight-line distance
from start(u) to start(goal) (divided by the max speed in time mode). Every edge
cost is >= the straight-line hop between lane start points, so the heuristic is
admissible as long as successor lanes are geometrically connected.
"""

from __future__ import annotations

import enum
import heapq
import itertools
import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from .topology import LaneMap


class EdgeType(enum.Enum):
    SUCCESSOR = "successor"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    type: EdgeType
    length_m: float  # cost in metres before time conversion / overrides


@dataclass
class RoadGraphConfig:
    allow_lane_changes: bool = True
    lane_change_penalty_m: float = 30.0  # discourages unnecessary lane changes
    respect_boundary_types: bool = True  # no lane change across solid lines
    cost_mode: str = "distance"  # "distance" | "time"
    min_speed_mps: float = 0.5  # floor for time-mode division


@dataclass
class PathResult:
    lanes: list[str]
    edge_types: list[EdgeType]  # len(lanes) - 1
    cost: float
    expanded: int = 0  # nodes popped from the open set (search effort)

    @property
    def num_lane_changes(self) -> int:
        return sum(1 for e in self.edge_types if e != EdgeType.SUCCESSOR)


class RoadGraph:
    """Lane-level directed graph built from a `LaneMap`.

    # TODO(mapping): turn costs at intersections, traffic-aware time costs,
    # contraction hierarchies for large maps, lane-change edges that land part-way
    # along the target lane (currently modeled lane-start to lane-start).
    """

    def __init__(self, lane_map: LaneMap, config: Optional[RoadGraphConfig] = None) -> None:
        self.config = config or RoadGraphConfig()
        self.lane_map = lane_map
        self._out: dict[str, list[Edge]] = {}
        self._in: dict[str, list[Edge]] = {}
        self.rebuild()

    @classmethod
    def from_lane_map(cls, lane_map: LaneMap, config: Optional[RoadGraphConfig] = None) -> "RoadGraph":
        return cls(lane_map, config)

    def rebuild(self) -> None:
        cfg = self.config
        self._out = {lid: [] for lid in self.lane_map.ids()}
        self._in = {lid: [] for lid in self.lane_map.ids()}
        for lane in self.lane_map:
            for sid in lane.successors:
                if sid in self.lane_map:
                    self._add(Edge(lane.id, sid, EdgeType.SUCCESSOR, lane.length))
            if not cfg.allow_lane_changes:
                continue
            for nid, etype, ok in (
                (lane.left_neighbor, EdgeType.LANE_CHANGE_LEFT, lane.can_change_left()),
                (lane.right_neighbor, EdgeType.LANE_CHANGE_RIGHT, lane.can_change_right()),
            ):
                if nid is None or nid not in self.lane_map:
                    continue
                if cfg.respect_boundary_types and not ok:
                    continue
                hop = float(np.linalg.norm(lane.centerline.start - self.lane_map[nid].centerline.start))
                self._add(Edge(lane.id, nid, etype, cfg.lane_change_penalty_m + hop))

    def _add(self, e: Edge) -> None:
        self._out[e.src].append(e)
        self._in[e.dst].append(e)

    # ----------------------------------------------------------------- queries
    @property
    def nodes(self) -> list[str]:
        return list(self._out.keys())

    def out_edges(self, lane_id: str) -> list[Edge]:
        return list(self._out.get(lane_id, []))

    def in_edges(self, lane_id: str) -> list[Edge]:
        return list(self._in.get(lane_id, []))

    def num_edges(self) -> int:
        return sum(len(v) for v in self._out.values())

    def reachable(self, start: str, blocked: Optional[Iterable[str]] = None) -> set[str]:
        blocked = set(blocked or ())
        seen, stack = {start}, [start]
        while stack:
            u = stack.pop()
            for e in self._out.get(u, []):
                if e.dst not in seen and e.dst not in blocked:
                    seen.add(e.dst)
                    stack.append(e.dst)
        return seen

    def edge_cost(self, e: Edge, speed_limits: Optional[dict[str, float]] = None) -> float:
        if self.config.cost_mode == "time":
            v = (speed_limits or {}).get(e.src, self.lane_map[e.src].speed_limit)
            return e.length_m / max(v, self.config.min_speed_mps)
        return e.length_m

    def _heuristic_fn(self, goal: str, speed_limits: Optional[dict[str, float]]):
        g = self.lane_map[goal].centerline.start
        scale = 1.0
        if self.config.cost_mode == "time":
            speeds = [lane.speed_limit for lane in self.lane_map] + list((speed_limits or {}).values())
            scale = 1.0 / max(max(speeds), self.config.min_speed_mps)
        starts = {lane.id: lane.centerline.start for lane in self.lane_map}

        def h(u: str) -> float:
            p = starts[u]
            return scale * math.hypot(p[0] - g[0], p[1] - g[1])

        return h

    def shortest_path(
        self,
        start: str,
        goal: str,
        algorithm: str = "astar",
        blocked: Optional[Iterable[str]] = None,
        speed_limits: Optional[dict[str, float]] = None,
        cost_scale: Optional[dict[str, float]] = None,
    ) -> Optional[PathResult]:
        """Lane sequence from `start` to `goal` (inclusive), or None if unreachable.

        blocked:      lane ids that may not be entered (closures). `start` is always allowed.
        speed_limits: per-lane overrides used by time-mode costs.
        cost_scale:   per-lane multiplicative cost factors (>= 1 keeps A* admissible).
        """
        if start not in self._out or goal not in self._out:
            raise KeyError(f"unknown lane {start!r} or {goal!r}")
        blocked = set(blocked or ()) - {start}
        if goal in blocked:
            return None
        cost_scale = cost_scale or {}
        if algorithm == "astar":
            h = self._heuristic_fn(goal, speed_limits)
        elif algorithm == "dijkstra":
            h = lambda u: 0.0  # noqa: E731
        else:
            raise ValueError(f"unknown algorithm {algorithm!r}")

        tie = itertools.count()
        g_cost = {start: 0.0}
        parent: dict[str, tuple[str, EdgeType]] = {}
        open_heap = [(h(start), next(tie), start)]
        closed: set[str] = set()
        expanded = 0
        while open_heap:
            _, _, u = heapq.heappop(open_heap)
            if u in closed:
                continue
            closed.add(u)
            expanded += 1
            if u == goal:
                break
            for e in self._out[u]:
                v = e.dst
                if v in blocked or v in closed:
                    continue
                c = self.edge_cost(e, speed_limits) * max(cost_scale.get(u, 1.0), 0.0)
                ng = g_cost[u] + c
                if ng < g_cost.get(v, math.inf) - 1e-12:
                    g_cost[v] = ng
                    parent[v] = (u, e.type)
                    heapq.heappush(open_heap, (ng + h(v), next(tie), v))
        if goal not in closed:
            return None
        lanes, types = [goal], []
        while lanes[-1] != start:
            p, t = parent[lanes[-1]]
            lanes.append(p)
            types.append(t)
        lanes.reverse()
        types.reverse()
        return PathResult(lanes=lanes, edge_types=types, cost=g_cost[goal], expanded=expanded)
