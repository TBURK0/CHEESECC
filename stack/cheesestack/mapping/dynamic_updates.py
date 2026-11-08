"""Dynamic map updates: a versioned overlay on top of a static `LaneMap`.

Updates (closures, speed-limit overrides, observed lane shifts, lane-change
blocks) carry a validity window [created, expires_at). The overlay never mutates
the base map; `materialize(now)` produces a patched copy, and
`graph_constraints(now)` gives inputs for `RoadGraph.shortest_path`.

Every change (add / remove / expire / fused observation) bumps `version`, so
consumers (route planner, matcher) can cheaply detect when to re-plan.
All time arguments are explicit (`now` from a core Clock).
"""

from __future__ import annotations

import copy
import enum
import math
from dataclasses import dataclass, replace
from typing import Optional

from .boundaries import offset_polyline
from .topology import BoundaryType, LaneBoundary, LaneMap, Polyline


class UpdateKind(enum.Enum):
    CLOSURE = "closure"  # value unused; optional s-range for partial closure
    SPEED_LIMIT = "speed_limit"  # value = m/s
    LANE_SHIFT = "lane_shift"  # value = observed lateral offset of the lane, m (+ left)
    LANE_CHANGE_BLOCK = "lane_change_block"  # value: +1 left, -1 right, 0 both


@dataclass
class MapUpdate:
    kind: UpdateKind
    lane_id: str
    value: float = 0.0
    created: float = 0.0
    expires_at: float = math.inf
    s_start: Optional[float] = None
    s_end: Optional[float] = None
    source: str = ""
    confidence: float = 1.0
    update_id: int = -1  # assigned by the overlay

    def is_active(self, now: float) -> bool:
        return self.created <= now < self.expires_at

    def covers(self, s: Optional[float]) -> bool:
        if s is None or (self.s_start is None and self.s_end is None):
            return True
        lo = -math.inf if self.s_start is None else self.s_start
        hi = math.inf if self.s_end is None else self.s_end
        return lo <= s <= hi


@dataclass
class OverlayConfig:
    default_ttl_s: float = 600.0  # applied when an update has expires_at == inf and ttl requested
    observation_ttl_s: float = 120.0  # lifetime of fused observations, refreshed on each one
    shift_gain: float = 0.3  # max EMA gain for observed lane shifts (scaled by confidence)
    min_confidence: float = 0.2  # updates below this are ignored
    max_shift_m: float = 2.0  # clamp on applied lane shift
    block_partial_closures: bool = True  # treat partial closures as blocking for routing


@dataclass(frozen=True)
class ChangeRecord:
    version: int
    action: str  # "add" | "remove" | "expire" | "observe"
    update_id: int
    stamp: float


class MapOverlay:
    """Versioned set of temporary map changes over a base `LaneMap`.

    # TODO(mapping): persistence + sync with a fleet map service (V2X/cloud),
    # conflict resolution between sources, spatial (polygon) closures that
    # resolve to multiple lanes, signed/authenticated updates.
    """

    def __init__(self, base: LaneMap, config: Optional[OverlayConfig] = None) -> None:
        self.base = base
        self.config = config or OverlayConfig()
        self.version = 0
        self._updates: dict[int, MapUpdate] = {}
        self._next_id = 1
        self.history: list[ChangeRecord] = []

    # ------------------------------------------------------------ mutation
    def _bump(self, action: str, uid: int, stamp: float) -> None:
        self.version += 1
        self.history.append(ChangeRecord(self.version, action, uid, stamp))

    def add(self, update: MapUpdate) -> int:
        """Register an update; returns its id (or -1 if rejected)."""
        if update.lane_id not in self.base:
            raise KeyError(f"unknown lane {update.lane_id!r}")
        if update.confidence < self.config.min_confidence or update.expires_at <= update.created:
            return -1
        uid = self._next_id
        self._next_id += 1
        self._updates[uid] = replace(update, update_id=uid)
        self._bump("add", uid, update.created)
        return uid

    def close_lane(self, lane_id: str, now: float, duration_s: Optional[float] = None,
                   s_range: Optional[tuple[float, float]] = None, source: str = "") -> int:
        ttl = self.config.default_ttl_s if duration_s is None else duration_s
        s0, s1 = s_range if s_range else (None, None)
        return self.add(MapUpdate(UpdateKind.CLOSURE, lane_id, 0.0, now, now + ttl, s0, s1, source))

    def set_speed_limit(self, lane_id: str, speed_mps: float, now: float,
                        duration_s: Optional[float] = None, source: str = "") -> int:
        ttl = self.config.default_ttl_s if duration_s is None else duration_s
        return self.add(MapUpdate(UpdateKind.SPEED_LIMIT, lane_id, speed_mps, now, now + ttl, source=source))

    def remove(self, update_id: int, now: float = 0.0) -> bool:
        if self._updates.pop(update_id, None) is None:
            return False
        self._bump("remove", update_id, now)
        return True

    def expire(self, now: float) -> list[int]:
        """Drop updates whose expires_at <= now. Returns removed ids."""
        gone = [uid for uid, u in self._updates.items() if u.expires_at <= now]
        for uid in gone:
            del self._updates[uid]
            self._bump("expire", uid, now)
        return gone

    def observe_lane_shift(self, lane_id: str, offset_m: float, now: float, confidence: float = 1.0,
                           source: str = "perception") -> int:
        """Fuse a perceived lateral offset of a lane (e.g. repainted markings / work-zone
        shift) into a single LANE_SHIFT update per (lane, source), EMA-smoothed and with
        its expiry refreshed. Returns the update id."""
        if confidence < self.config.min_confidence:
            return -1
        for uid, u in self._updates.items():
            if u.kind == UpdateKind.LANE_SHIFT and u.lane_id == lane_id and u.source == source and u.is_active(now):
                g = self.config.shift_gain * min(1.0, confidence)
                u.value = (1.0 - g) * u.value + g * offset_m
                u.confidence = min(1.0, u.confidence + 0.1 * confidence)
                u.expires_at = now + self.config.observation_ttl_s
                self._bump("observe", uid, now)
                return uid
        return self.add(MapUpdate(UpdateKind.LANE_SHIFT, lane_id, offset_m, now,
                                  now + self.config.observation_ttl_s, source=source, confidence=confidence))

    def block_lane_change(self, lane_id: str, side: int, now: float, duration_s: Optional[float] = None,
                          source: str = "") -> int:
        """side: +1 left, -1 right, 0 both."""
        ttl = self.config.default_ttl_s if duration_s is None else duration_s
        return self.add(MapUpdate(UpdateKind.LANE_CHANGE_BLOCK, lane_id, float(side), now, now + ttl, source=source))

    # -------------------------------------------------------------- queries
    def active(self, now: float, lane_id: Optional[str] = None,
               kind: Optional[UpdateKind] = None) -> list[MapUpdate]:
        return [u for u in self._updates.values()
                if u.is_active(now) and (lane_id is None or u.lane_id == lane_id) and (kind is None or u.kind == kind)]

    def is_closed(self, lane_id: str, now: float, s: Optional[float] = None) -> bool:
        return any(u.covers(s) for u in self.active(now, lane_id, UpdateKind.CLOSURE))

    def speed_limit(self, lane_id: str, now: float) -> float:
        """Effective limit = min(base, active overrides)."""
        v = self.base[lane_id].speed_limit
        for u in self.active(now, lane_id, UpdateKind.SPEED_LIMIT):
            v = min(v, u.value)
        return v

    def lane_shift(self, lane_id: str, now: float) -> float:
        shifts = self.active(now, lane_id, UpdateKind.LANE_SHIFT)
        if not shifts:
            return 0.0
        w = sum(u.confidence for u in shifts)
        val = sum(u.value * u.confidence for u in shifts) / max(w, 1e-9)
        return max(-self.config.max_shift_m, min(self.config.max_shift_m, val))

    def lane_change_allowed(self, lane_id: str, side: int, now: float) -> bool:
        for u in self.active(now, lane_id, UpdateKind.LANE_CHANGE_BLOCK):
            if u.value == 0 or int(math.copysign(1, u.value)) == side:
                return False
        return True

    def blocked_lanes(self, now: float) -> set[str]:
        out = set()
        for u in self.active(now, kind=UpdateKind.CLOSURE):
            full = u.s_start is None and u.s_end is None
            if full or self.config.block_partial_closures:
                out.add(u.lane_id)
        return out

    def speed_limits(self, now: float) -> dict[str, float]:
        return {lid: self.speed_limit(lid, now) for lid in {u.lane_id for u in self.active(now, kind=UpdateKind.SPEED_LIMIT)}}

    def graph_constraints(self, now: float) -> dict:
        """kwargs for RoadGraph.shortest_path: blocked lanes and speed overrides."""
        return {"blocked": self.blocked_lanes(now), "speed_limits": self.speed_limits(now)}

    def materialize(self, now: float, drop_closed: bool = True) -> LaneMap:
        """Deep copy of the base map with active updates applied.

        Fully closed lanes are removed (with references cleaned) when drop_closed;
        partially closed lanes keep their geometry and get `attributes["closures"]`.
        """
        lm = copy.deepcopy(self.base)
        for lane in lm:
            lane.speed_limit = self.speed_limit(lane.id, now)
            shift = self.lane_shift(lane.id, now)
            if abs(shift) > 1e-6:
                lane.centerline = Polyline(offset_polyline(lane.centerline.points, shift))
                for side in ("left_boundary", "right_boundary"):
                    b = getattr(lane, side)
                    if b is not None:
                        setattr(lane, side, LaneBoundary(Polyline(offset_polyline(b.polyline.points, shift)), b.type))
            for side, sign in ((1, 1.0), (-1, -1.0)):
                if not self.lane_change_allowed(lane.id, side, now):
                    attr = "left_boundary" if side == 1 else "right_boundary"
                    b = getattr(lane, attr)
                    poly = b.polyline if b else Polyline(offset_polyline(lane.centerline.points, sign * lane.width / 2))
                    setattr(lane, attr, LaneBoundary(poly, BoundaryType.SOLID))
            partial = [(u.s_start, u.s_end) for u in self.active(now, lane.id, UpdateKind.CLOSURE)
                       if not (u.s_start is None and u.s_end is None)]
            if partial:
                lane.attributes["closures"] = partial
        if drop_closed:
            for u in self.active(now, kind=UpdateKind.CLOSURE):
                if u.s_start is None and u.s_end is None and u.lane_id in lm:
                    lm.remove_lane(u.lane_id)
        return lm

    def __len__(self) -> int:
        return len(self._updates)
