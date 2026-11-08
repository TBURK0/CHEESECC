"""Merge planning: choose a target-lane gap and a longitudinal profile to reach it.

For each candidate gap and a set of horizons T, the gap is propagated with
constant velocity (each boundary at its own speed), and a quintic longitudinal
profile s(t) is solved from the ego state (s, v, a) to (gap target point at T,
gap speed, 0). Profiles violating speed/accel/jerk limits or arriving in a gap
that has shrunk below the required length are rejected; the remaining one with
the lowest cost (time + jerk + accel + gap score) is returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .common import QuinticPolynomial
from .gaps import Gap, GapConfig


@dataclass
class MergeConfig:
    horizons_s: tuple = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
    dt: float = 0.1
    v_max: float = 33.0
    a_max: float = 2.0
    a_min: float = -3.5
    jerk_max: float = 4.0
    w_time: float = 1.0
    w_jerk: float = 0.1
    w_accel: float = 1.0
    w_score: float = 0.05
    gap: GapConfig = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.gap is None:
            self.gap = GapConfig()


@dataclass
class MergePlan:
    gap: Gap
    T: float
    t: np.ndarray
    s: np.ndarray
    v: np.ndarray
    a: np.ndarray
    cost: float

    @property
    def target_speed_now(self) -> float:
        """Speed command one step ahead, for a simple speed controller."""
        return float(self.v[min(1, len(self.v) - 1)])


class MergePlanner:
    def __init__(self, config: MergeConfig = MergeConfig()) -> None:
        self.config = config

    def _gap_target(self, g: Gap, T: float, ego_s: float) -> tuple[float, float, float]:
        """(target s, target v, gap length) of gap at time T."""
        c = self.config.gap
        s0 = g.s_start + g.v_start * T
        s1 = g.s_end + g.v_end * T
        if g.rear is None and g.front is None:
            return ego_s + g.v_start * T, g.v_start, s1 - s0
        if g.rear is None:  # open behind: slot in a safe headway behind front vehicle
            target = s1 - c.ego_length / 2 - max(c.min_static_margin_m, c.min_front_headway_s * g.v_end)
            return target, g.v_end, s1 - s0
        if g.front is None:  # open ahead: slot a safe headway in front of rear vehicle
            target = s0 + c.ego_length / 2 + max(c.min_static_margin_m, c.min_rear_headway_s * g.v_start)
            return target, g.v_start, s1 - s0
        return 0.5 * (s0 + s1), 0.5 * (g.v_start + g.v_end), s1 - s0

    def plan(self, ego_s: float, ego_v: float, ego_a: float, gaps: list[Gap]) -> Optional[MergePlan]:
        c = self.config
        need = c.gap.ego_length + 2 * c.gap.min_static_margin_m
        best: Optional[MergePlan] = None
        for g in gaps:
            for T in c.horizons_s:
                st, vt, L = self._gap_target(g, T, ego_s)
                if L < need or vt < 0 or vt > c.v_max:
                    continue
                poly = QuinticPolynomial(ego_s, ego_v, ego_a, st, vt, 0.0, T)
                t = np.arange(0.0, T + 1e-9, c.dt)
                s, v, a, j = poly.x(t), poly.v(t), poly.acc(t), poly.jerk(t)
                if v.min() < -1e-3 or v.max() > c.v_max or a.max() > c.a_max or a.min() < c.a_min:
                    continue
                if np.abs(j).max() > c.jerk_max:
                    continue
                cost = c.w_time * T + c.w_jerk * float(np.sum(j**2) * c.dt) + c.w_accel * float(np.abs(a).max()) - c.w_score * g.score
                if best is None or cost < best.cost:
                    best = MergePlan(g, T, t, s, v, a, cost)
        return best

    # TODO(planning): interaction-aware merging (rear vehicle yields/accelerates),
    # end-of-lane deadline constraint, re-plan with commitment/hysteresis on gap choice.
