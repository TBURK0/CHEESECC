"""Gap assessment in a target lane.

Objects are reduced to (s, d, v_s) along the ego lane (straight-road assumption
in base_link, or via a `ReferencePath`). Objects whose lateral offset lies in the
target lane are sorted by s; every consecutive pair (plus open-ended gaps at the
front/back) is a candidate. A gap is acceptable for the ego to slot in *now* at
its current position if:

  * the ego's longitudinal extent fits with a static margin,
  * time headway to the front vehicle >= `min_front_headway_s` and TTC to it
    (if closing) >= `min_ttc_s`,
  * the rear vehicle's headway to the ego >= `min_rear_headway_s` and its TTC
    (if closing) >= `min_ttc_s`.

Non-current gaps (ahead or behind) are still reported with a score so a merge
planner can try to reach them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from cheesestack.core import TrackedObject

from .common import ReferencePath, object_frenet


@dataclass
class GapConfig:
    lane_width: float = 3.6
    ego_length: float = 4.7
    min_static_margin_m: float = 2.0
    min_front_headway_s: float = 1.0
    min_rear_headway_s: float = 1.0
    min_ttc_s: float = 4.0
    max_range_m: float = 150.0  # open-ended gap length


@dataclass
class LaneObject:
    track_id: int
    s: float  # centre, m
    d: float
    v: float
    length: float = 4.5


@dataclass
class Gap:
    rear: Optional[LaneObject]
    front: Optional[LaneObject]
    s_start: float  # rear bumper of gap (front of rear vehicle)
    s_end: float  # front bound (rear bumper of front vehicle)
    v_start: float  # speed of gap's rear boundary
    v_end: float
    contains_ego: bool = False
    acceptable: bool = False
    front_headway: float = float("inf")
    rear_headway: float = float("inf")
    front_ttc: float = float("inf")
    rear_ttc: float = float("inf")
    reasons: list = field(default_factory=list)
    score: float = 0.0

    @property
    def length(self) -> float:
        return self.s_end - self.s_start

    @property
    def center(self) -> float:
        return 0.5 * (self.s_start + self.s_end)

    @property
    def speed(self) -> float:
        return 0.5 * (self.v_start + self.v_end)


def _ttc(gap: float, closing: float) -> float:
    return gap / closing if closing > 1e-6 and gap > 0 else (0.0 if gap <= 0 else float("inf"))


class GapAssessor:
    def __init__(self, config: GapConfig = GapConfig()) -> None:
        self.config = config

    def lane_objects(self, objects: list[TrackedObject], target_offset: float, ref: Optional[ReferencePath] = None) -> list[LaneObject]:
        out = []
        for o in objects:
            s, d, v = object_frenet(o, ref)
            if abs(d - target_offset) <= self.config.lane_width / 2:
                out.append(LaneObject(o.track_id, s, d, v, o.length))
        return sorted(out, key=lambda lo: lo.s)

    def assess(self, ego_s: float, ego_v: float, lane_objs: list[LaneObject]) -> list[Gap]:
        """Evaluate all gaps in a target lane. `ego_s` is ego centre along the lane."""
        c = self.config
        objs = sorted(lane_objs, key=lambda o: o.s)
        bounds: list[tuple[Optional[LaneObject], Optional[LaneObject]]] = []
        prev = None
        for o in objs:
            bounds.append((prev, o))
            prev = o
        bounds.append((prev, None))
        gaps = []
        e_rear, e_front = ego_s - c.ego_length / 2, ego_s + c.ego_length / 2
        for rear, front in bounds:
            s0 = rear.s + rear.length / 2 if rear else ego_s - c.max_range_m
            s1 = front.s - front.length / 2 if front else ego_s + c.max_range_m
            v0 = rear.v if rear else ego_v
            v1 = front.v if front else ego_v
            g = Gap(rear, front, s0, s1, v0, v1)
            g.contains_ego = s0 < ego_s < s1
            if g.length < c.ego_length + 2 * c.min_static_margin_m:
                g.reasons.append("too short")
            if g.contains_ego:
                fgap, rgap = s1 - e_front, e_rear - s0
                if fgap < c.min_static_margin_m or rgap < c.min_static_margin_m:
                    g.reasons.append("static margin")
                if front:
                    g.front_headway = fgap / max(ego_v, 0.1)
                    g.front_ttc = _ttc(fgap, ego_v - front.v)
                    if g.front_headway < c.min_front_headway_s:
                        g.reasons.append("front headway")
                    if g.front_ttc < c.min_ttc_s:
                        g.reasons.append("front ttc")
                if rear:
                    g.rear_headway = rgap / max(rear.v, 0.1)
                    g.rear_ttc = _ttc(rgap, rear.v - ego_v)
                    if g.rear_headway < c.min_rear_headway_s:
                        g.reasons.append("rear headway")
                    if g.rear_ttc < c.min_ttc_s:
                        g.reasons.append("rear ttc")
            else:
                g.reasons.append("not alongside")
            g.acceptable = not g.reasons
            # Score: prefer long gaps, near the ego, moving near ego speed.
            dist = 0.0 if g.contains_ego else min(abs(s0 - ego_s), abs(s1 - ego_s))
            g.score = min(g.length, 60.0) - 0.5 * dist - 2.0 * abs(g.speed - ego_v)
            gaps.append(g)
        return gaps

    def best_current_gap(self, ego_s: float, ego_v: float, lane_objs: list[LaneObject]) -> Optional[Gap]:
        for g in self.assess(ego_s, ego_v, lane_objs):
            if g.contains_ego and g.acceptable:
                return g
        return None

    # TODO(planning): use predicted trajectories (not constant velocity) and track
    # covariance; account for vehicles changing into the same target lane.
