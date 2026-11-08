"""Time-to-collision estimators.

* ``ttc_longitudinal``: classic 1-D TTC = gap / closing speed.
* ``ttc_constant_accel``: "enhanced" TTC; smallest positive root of
  ``gap - v_rel t - 0.5 a_rel t^2 = 0`` (v_rel, a_rel are closing speed/accel).
  Optionally accounts for the lead vehicle stopping (no reversing).
* ``closest_point_of_approach``: 2-D CPA time and distance for constant velocities.
* ``ttc_2d_circles``: first time two constant-velocity discs come within
  ``r_a + r_b``.
* ``TTCEstimator``: ego ``VehicleState`` vs ``TrackedObject`` -> all of the above,
  with the object expressed in the ego frame and bumper-to-bumper gaps.

All functions return ``math.inf`` when no collision is predicted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from cheesestack.core import TrackedObject, VehicleState, rot2

INF = math.inf


def ttc_longitudinal(gap: float, v_follow: float, v_lead: float) -> float:
    """Gap [m] (bumper-to-bumper) / closing speed. inf if not closing; 0 if already overlapping."""
    if gap <= 0.0:
        return 0.0
    closing = v_follow - v_lead
    return gap / closing if closing > 1e-9 else INF


def _smallest_positive_root(a: float, b: float, c: float) -> float:
    """Smallest t > 0 with a t^2 + b t + c = 0."""
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return INF
        t = -c / b
        return t if t > 0 else INF
    disc = b * b - 4 * a * c
    if disc < 0:
        return INF
    sq = math.sqrt(disc)
    roots = sorted([(-b - sq) / (2 * a), (-b + sq) / (2 * a)])
    for t in roots:
        if t > 1e-9:
            return t
    return INF


def ttc_constant_accel(gap: float, v_follow: float, v_lead: float,
                       a_follow: float = 0.0, a_lead: float = 0.0,
                       stop_aware: bool = True, horizon: float = 30.0) -> float:
    """Enhanced TTC with constant accelerations.

    With ``stop_aware`` neither vehicle reverses: once a vehicle's speed hits 0 it
    stays stopped. Solved piecewise exactly over the (at most 3) intervals.
    """
    if gap <= 0.0:
        return 0.0
    if not stop_aware:
        t = _smallest_positive_root(-0.5 * (a_follow - a_lead), -(v_follow - v_lead), gap)
        return t if t <= horizon else INF

    def stop_t(v, a):
        return v / -a if a < 0 and v > 0 else (0.0 if v <= 0 and a <= 0 else INF)

    ts = sorted({0.0, min(stop_t(v_follow, a_follow), horizon), min(stop_t(v_lead, a_lead), horizon), horizon})
    g, vf, vl = gap, max(v_follow, 0.0), max(v_lead, 0.0)
    af = a_follow if vf > 0 or a_follow > 0 else 0.0
    al = a_lead if vl > 0 or a_lead > 0 else 0.0
    for t0, t1 in zip(ts[:-1], ts[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        # gap(τ) = g - (vf - vl) τ - 0.5 (af - al) τ^2 for τ in [0, dt]
        tau = _smallest_positive_root(-0.5 * (af - al), -(vf - vl), g)
        if tau <= dt + 1e-12:
            return t0 + tau
        g = g - (vf - vl) * dt - 0.5 * (af - al) * dt * dt
        vf, vl = max(vf + af * dt, 0.0), max(vl + al * dt, 0.0)
        if vf <= 1e-9 and af < 0:
            vf, af = 0.0, 0.0
        if vl <= 1e-9 and al < 0:
            vl, al = 0.0, 0.0
    return INF


def closest_point_of_approach(p_a, v_a, p_b, v_b):
    """(t_cpa >= 0, d_cpa) for two constant-velocity points."""
    dp = np.asarray(p_b, float) - np.asarray(p_a, float)
    dv = np.asarray(v_b, float) - np.asarray(v_a, float)
    vv = float(dv @ dv)
    t = 0.0 if vv < 1e-12 else max(0.0, -float(dp @ dv) / vv)
    return t, float(np.linalg.norm(dp + dv * t))


def ttc_2d_circles(p_a, v_a, r_a: float, p_b, v_b, r_b: float) -> float:
    """First t >= 0 at which |p_b - p_a| = r_a + r_b. 0 if already overlapping."""
    dp = np.asarray(p_b, float) - np.asarray(p_a, float)
    dv = np.asarray(v_b, float) - np.asarray(v_a, float)
    R = r_a + r_b
    c = float(dp @ dp) - R * R
    if c <= 0:
        return 0.0
    return _smallest_positive_root(float(dv @ dv), 2.0 * float(dp @ dv), c)


@dataclass
class TTCEstimatorConfig:
    ego_length: float = 4.5
    ego_width: float = 1.8
    ego_rear_overhang: float = 1.0  # base_link (rear axle) to rear bumper
    lateral_margin: float = 0.3  # extra lateral clearance treated as "in path"
    horizon: float = 10.0


@dataclass
class TTCResult:
    in_path: bool
    gap: float  # bumper-to-bumper longitudinal gap (ego frame)
    ttc: float  # 1-D, constant speeds
    ttc_accel: float  # 1-D, constant accelerations, stop-aware
    t_cpa: float
    d_cpa: float
    ttc_2d: float  # disc-based 2-D TTC

    @property
    def min_ttc(self) -> float:
        cands = [self.ttc_2d] + ([self.ttc, self.ttc_accel] if self.in_path else [])
        return min(cands)


class TTCEstimator:
    """Ego vs object TTC. Object velocity is assumed in the same frame as ego pose."""

    def __init__(self, config: Optional[TTCEstimatorConfig] = None):
        self.cfg = config or TTCEstimatorConfig()

    def estimate(self, ego: VehicleState, obj: TrackedObject, obj_accel: float = 0.0) -> TTCResult:
        c = self.cfg
        R = rot2(ego.pose.yaw)
        # Ego geometric center in map frame (base_link is rear axle).
        center_off = 0.5 * c.ego_length - c.ego_rear_overhang
        ego_c = np.array([ego.pose.x, ego.pose.y]) + R @ np.array([center_off, 0.0])
        rel = R.T @ (np.array([obj.pose.x, obj.pose.y]) - ego_c)
        v_obj_e = R.T @ np.asarray(obj.velocity, float)
        rel_yaw = obj.pose.yaw - ego.pose.yaw
        # Object extent projected onto ego axes.
        ext_x = 0.5 * (abs(math.cos(rel_yaw)) * obj.length + abs(math.sin(rel_yaw)) * obj.width)
        ext_y = 0.5 * (abs(math.sin(rel_yaw)) * obj.length + abs(math.cos(rel_yaw)) * obj.width)
        in_path = rel[0] > 0 and abs(rel[1]) < 0.5 * c.ego_width + ext_y + c.lateral_margin
        gap = float(rel[0] - 0.5 * c.ego_length - ext_x)
        v_lead = float(v_obj_e[0])
        if in_path:
            ttc = ttc_longitudinal(gap, ego.v, v_lead)
            ttc_a = ttc_constant_accel(gap, ego.v, v_lead, ego.a, obj_accel, horizon=c.horizon)
        else:
            ttc = ttc_a = INF
        v_ego = np.array([ego.v, 0.0])
        t_cpa, d_cpa = closest_point_of_approach(np.zeros(2), v_ego, rel, v_obj_e)
        r_e = 0.5 * c.ego_width  # TODO(prediction): multi-disc footprint instead of one disc each.
        r_o = 0.5 * min(obj.length, obj.width)
        ttc2 = ttc_2d_circles(np.zeros(2), v_ego, r_e, rel, v_obj_e, r_o)
        if ttc2 > c.horizon:
            ttc2 = INF
        return TTCResult(bool(in_path), gap, ttc, ttc_a, t_cpa, d_cpa, ttc2)
