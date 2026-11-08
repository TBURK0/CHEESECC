"""Overtake decision: kinematic feasibility + cost-benefit.

Manoeuvre model: ego (speed v0, behind a slower lead at speed v_l) moves to the
passing lane, accelerates at `a_pass` up to `v_pass_max`, and returns once it is
`return_gap_m` ahead of the lead. The pass time T solves the relative distance

    D = gap_behind + L_lead + L_ego + return_gap

under the accel-then-cruise profile relative to the lead. Checks:
  * oncoming traffic (two-lane road): closing distance during T plus a time
    margin must be less than the oncoming vehicle's distance;
  * adjacent same-direction traffic in the passing lane: no vehicle ahead in the
    passing lane within the space the ego needs, no fast vehicle approaching
    from behind that would reach the ego within T + margin;
  * sight distance must cover the oncoming check range.
Benefit is time saved over a planning horizon versus following; cost is pass
time, risk from the smallest safety margin, and comfort (accel).
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Optional


class OvertakeDecision(enum.Enum):
    OVERTAKE = "overtake"
    FOLLOW = "follow"


@dataclass
class OvertakeConfig:
    ego_length: float = 4.7
    return_gap_m: float = 15.0
    a_pass: float = 1.5
    v_pass_max: float = 30.0
    oncoming_margin_s: float = 3.0
    rear_margin_s: float = 2.0
    min_speed_gain: float = 2.0  # do not bother overtaking for less, m/s
    horizon_m: float = 2000.0  # distance over which time saving is evaluated
    w_time_saved: float = 1.0
    w_pass_time: float = 0.5
    w_risk: float = 20.0
    w_accel: float = 1.0
    max_pass_time_s: float = 20.0


@dataclass
class OtherVehicle:
    distance: float  # longitudinal, m (signed: + ahead of ego front, - behind ego rear)
    speed: float  # m/s, in its own travel direction (positive)
    length: float = 4.5


@dataclass
class OvertakeContext:
    ego_speed: float
    desired_speed: float
    lead: OtherVehicle  # in ego lane, distance = bumper gap ahead
    oncoming: list = field(default_factory=list)  # OtherVehicle, distance ahead
    passing_lane: list = field(default_factory=list)  # same-direction in passing lane
    sight_distance_m: float = float("inf")
    passing_allowed: bool = True  # lane markings / regulation


@dataclass
class OvertakeResult:
    decision: OvertakeDecision
    pass_time_s: float
    pass_distance_m: float
    min_margin_s: float
    benefit: float
    cost: float
    reasons: list = field(default_factory=list)


def pass_time(rel_distance: float, v0: float, v_lead: float, a: float, v_max: float) -> float:
    """Time for the ego to gain `rel_distance` on a lead at constant speed, accelerating
    at `a` from v0 up to v_max, then cruising. inf if never."""
    if rel_distance <= 0:
        return 0.0
    v_max = max(v_max, v0)
    t_acc = (v_max - v0) / a if a > 0 else 0.0
    # Relative gain during accel phase.
    gain_acc = (v0 - v_lead) * t_acc + 0.5 * a * t_acc**2
    if gain_acc >= rel_distance:
        # solve 0.5 a t^2 + (v0 - vl) t - D = 0
        b = v0 - v_lead
        return (-b + math.sqrt(b * b + 2 * a * rel_distance)) / a
    dv = v_max - v_lead
    if dv <= 1e-6:
        return float("inf")
    return t_acc + (rel_distance - gain_acc) / dv


class OvertakeEvaluator:
    def __init__(self, config: OvertakeConfig = OvertakeConfig()) -> None:
        self.config = config

    def evaluate(self, ctx: OvertakeContext) -> OvertakeResult:
        c = self.config
        reasons = []
        v0, vl = ctx.ego_speed, ctx.lead.speed
        v_target = min(ctx.desired_speed, c.v_pass_max)
        if not ctx.passing_allowed:
            reasons.append("passing not allowed")
        if v_target - vl < c.min_speed_gain:
            reasons.append("insufficient speed gain")
        D = max(ctx.lead.distance, 0.0) + ctx.lead.length + c.ego_length + c.return_gap_m
        T = pass_time(D, v0, vl, c.a_pass, v_target)
        if T > c.max_pass_time_s:
            reasons.append("pass too long")
            T = min(T, 1e6)
        t_acc = min(T, max(0.0, (v_target - v0) / c.a_pass))
        ego_dist = v0 * t_acc + 0.5 * c.a_pass * t_acc**2 + max(0.0, T - t_acc) * max(v_target, v0)

        margins = [float("inf")]
        # Oncoming: they close at their speed; we need the passing lane for T.
        for o in ctx.oncoming:
            closing = ego_dist + o.speed * T
            spare = o.distance - closing
            rel_v = max(v_target, v0) + o.speed
            m = spare / rel_v if rel_v > 0 else float("inf")
            margins.append(m)
            if m < c.oncoming_margin_s:
                reasons.append("oncoming traffic")
        if ctx.oncoming or math.isfinite(ctx.sight_distance_m):
            needed_sight = ego_dist + c.v_pass_max * (T + c.oncoming_margin_s)
            if ctx.sight_distance_m < needed_sight:
                reasons.append("insufficient sight distance")
        # Same-direction traffic in the passing lane.
        for p in ctx.passing_lane:
            if p.distance >= -c.ego_length:  # ahead/alongside: is it out of our way in time?
                their_pos_T = p.distance + p.speed * T
                if their_pos_T < ego_dist + c.return_gap_m:
                    reasons.append("passing lane blocked ahead")
                    margins.append((their_pos_T - ego_dist) / max(v_target, 1.0))
            else:  # behind: will it catch us?
                closing = p.speed - v0
                if closing > 0:
                    t_catch = (-p.distance) / closing
                    margins.append(t_catch - T)
                    if t_catch < T + c.rear_margin_s:
                        reasons.append("fast vehicle approaching in passing lane")

        min_margin = min(margins)
        time_follow = c.horizon_m / max(vl, 0.5)
        time_pass = c.horizon_m / max(v_target, 0.5)
        benefit = c.w_time_saved * (time_follow - time_pass)
        risk = c.w_risk / max(min_margin, 0.1) if math.isfinite(min_margin) else 0.0
        cost = c.w_pass_time * T + risk + c.w_accel * c.a_pass
        if not reasons and benefit <= cost:
            reasons.append("cost exceeds benefit")
        decision = OvertakeDecision.FOLLOW if reasons else OvertakeDecision.OVERTAKE
        return OvertakeResult(decision, T, ego_dist, min_margin, benefit, cost, reasons)

    # TODO(planning): abort-overtake logic mid-manoeuvre, uncertainty on oncoming
    # speed estimates (long range radar), multi-vehicle platoon overtakes.
