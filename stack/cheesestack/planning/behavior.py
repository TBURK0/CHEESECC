"""Behavioral planning: a rule-based finite state machine.

States: LANE_FOLLOW, FOLLOW_VEHICLE, PREPARE_LANE_CHANGE, LANE_CHANGE, STOP,
EMERGENCY_STOP. Transitions are driven by a compact `WorldSummary` assembled
by the integration layer from perception/prediction/safety. Priority order:
emergency > stop > lane-change logic > car following. Minimum dwell times and
hysteresis on the following distance keep it from chattering.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional


class BehaviorState(enum.Enum):
    LANE_FOLLOW = "lane_follow"
    FOLLOW_VEHICLE = "follow_vehicle"
    PREPARE_LANE_CHANGE = "prepare_lane_change"
    LANE_CHANGE = "lane_change"
    STOP = "stop"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class WorldSummary:
    ego_speed: float
    desired_speed: float
    lead_distance: Optional[float] = None  # bumper gap to in-lane lead, m
    lead_speed: Optional[float] = None
    lane_change_request: int = 0  # -1 right, 0 none, +1 left
    target_gap_acceptable: bool = False
    lane_change_progress: float = 0.0  # 0 at start, 1 when centred in target lane
    stop_distance: Optional[float] = None  # distance to stop line / stop target, m
    min_ttc: Optional[float] = None  # smallest TTC to any in-path object, s
    emergency: bool = False  # safety-layer request (fault, driver takeover...)


@dataclass
class BehaviorConfig:
    follow_headway_s: float = 3.0  # enter FOLLOW_VEHICLE if lead within this headway
    follow_exit_factor: float = 1.3  # leave only beyond headway * factor (hysteresis)
    follow_min_range_m: float = 15.0
    emergency_ttc_s: float = 1.5
    comfort_decel: float = 2.5  # m/s^2, used to decide when to start stopping
    stop_margin_m: float = 5.0
    stopped_speed: float = 0.2
    stop_dwell_s: float = 1.0
    prepare_timeout_s: float = 8.0
    min_state_dwell_s: float = 0.3
    lane_change_done_progress: float = 0.95
    abort_lane_change_progress: float = 0.3  # abort allowed only before this progress


@dataclass
class BehaviorDecision:
    state: BehaviorState
    target_speed: float
    lane_change_direction: int  # -1/0/+1, nonzero during PREPARE/LANE_CHANGE
    reason: str
    stop_distance: Optional[float] = None


class BehaviorPlanner:
    def __init__(self, config: BehaviorConfig = BehaviorConfig()) -> None:
        self.config = config
        self.state = BehaviorState.LANE_FOLLOW
        self.entered_at = 0.0
        self.lc_direction = 0
        self.stopped_since: Optional[float] = None
        self.history: list[tuple[float, BehaviorState, str]] = []

    def _go(self, t: float, new: BehaviorState, reason: str) -> None:
        if new != self.state:
            self.state = new
            self.entered_at = t
            self.history.append((t, new, reason))

    def _following_needed(self, w: WorldSummary, exiting: bool) -> bool:
        if w.lead_distance is None:
            return False
        rng = max(self.config.follow_min_range_m, self.config.follow_headway_s * max(w.ego_speed, 0.0))
        if exiting:
            rng *= self.config.follow_exit_factor
        return w.lead_distance <= rng

    def _must_stop(self, w: WorldSummary) -> bool:
        if w.stop_distance is None:
            return False
        brake = w.ego_speed**2 / (2 * self.config.comfort_decel)
        return w.stop_distance <= brake + self.config.stop_margin_m

    def update(self, w: WorldSummary, t: float) -> BehaviorDecision:
        c = self.config
        s = self.state
        dwell = t - self.entered_at
        reason = "hold"

        # 1. Emergency has absolute priority (no dwell).
        ttc_bad = w.min_ttc is not None and w.min_ttc < c.emergency_ttc_s
        if w.emergency or ttc_bad:
            self.lc_direction = 0
            self._go(t, BehaviorState.EMERGENCY_STOP, "emergency flag" if w.emergency else f"ttc {w.min_ttc:.2f}s")
            return self._decision(w, "emergency")
        if s == BehaviorState.EMERGENCY_STOP:
            if w.ego_speed <= c.stopped_speed:
                self._go(t, BehaviorState.STOP, "emergency cleared, stopped")
                self.stopped_since = t
            return self._decision(w, "emergency stopping")

        # 2. Stop handling.
        if self._must_stop(w) and s != BehaviorState.STOP:
            if s == BehaviorState.LANE_CHANGE and w.lane_change_progress >= c.abort_lane_change_progress:
                pass  # finish the lane change first; speed is still limited below
            else:
                self.lc_direction = 0
                self.stopped_since = None
                self._go(t, BehaviorState.STOP, "stop required")
                return self._decision(w, "stop required")
        if self.state == BehaviorState.STOP:
            if w.ego_speed <= c.stopped_speed:
                self.stopped_since = self.stopped_since if self.stopped_since is not None else t
            still_needed = w.stop_distance is not None and self._must_stop(w)
            if not still_needed and self.stopped_since is not None and t - self.stopped_since >= c.stop_dwell_s:
                self.stopped_since = None
                self._go(t, BehaviorState.LANE_FOLLOW, "stop released")
            elif not still_needed and w.stop_distance is None and w.ego_speed > c.stopped_speed:
                # stop target vanished while still rolling
                self._go(t, BehaviorState.LANE_FOLLOW, "stop target cleared")
            return self._decision(w, "stopping" if self.state == BehaviorState.STOP else "stop released")

        if dwell < c.min_state_dwell_s:
            return self._decision(w, "dwell")

        # 3. Lane change logic.
        s = self.state
        if s in (BehaviorState.LANE_FOLLOW, BehaviorState.FOLLOW_VEHICLE) and w.lane_change_request != 0:
            self.lc_direction = 1 if w.lane_change_request > 0 else -1
            self._go(t, BehaviorState.PREPARE_LANE_CHANGE, "lane change requested")
            reason = "prepare"
        elif s == BehaviorState.PREPARE_LANE_CHANGE:
            if w.lane_change_request == 0:
                self.lc_direction = 0
                self._go(t, BehaviorState.LANE_FOLLOW, "request cancelled")
            elif dwell > c.prepare_timeout_s:
                self.lc_direction = 0
                self._go(t, BehaviorState.LANE_FOLLOW, "prepare timeout")
            elif w.target_gap_acceptable:
                self._go(t, BehaviorState.LANE_CHANGE, "gap accepted")
            reason = "prepare"
        elif s == BehaviorState.LANE_CHANGE:
            if w.lane_change_progress >= c.lane_change_done_progress:
                self.lc_direction = 0
                self._go(t, BehaviorState.LANE_FOLLOW, "lane change complete")
            elif not w.target_gap_acceptable and w.lane_change_progress < c.abort_lane_change_progress:
                self.lc_direction = 0
                self._go(t, BehaviorState.LANE_FOLLOW, "lane change aborted: gap closed")
            reason = "lane change"

        # 4. Car following with hysteresis.
        s = self.state
        if s == BehaviorState.LANE_FOLLOW and self._following_needed(w, exiting=False):
            self._go(t, BehaviorState.FOLLOW_VEHICLE, "lead vehicle in range")
        elif s == BehaviorState.FOLLOW_VEHICLE and not self._following_needed(w, exiting=True):
            self._go(t, BehaviorState.LANE_FOLLOW, "lead vehicle gone")
        return self._decision(w, reason)

    def _decision(self, w: WorldSummary, reason: str) -> BehaviorDecision:
        c = self.config
        s = self.state
        v = w.desired_speed
        if s in (BehaviorState.STOP, BehaviorState.EMERGENCY_STOP):
            v = 0.0
        elif s == BehaviorState.FOLLOW_VEHICLE and w.lead_speed is not None and w.lead_distance is not None:
            # Simple gap regulation: lead speed plus proportional gap correction.
            desired_gap = max(c.follow_min_range_m * 0.5, c.follow_headway_s * 0.6 * w.ego_speed)
            v = min(w.desired_speed, max(0.0, w.lead_speed + 0.3 * (w.lead_distance - desired_gap)))
        if self._must_stop(w) and w.stop_distance is not None:
            # Speed that allows a comfortable stop at the line.
            v = min(v, max(0.0, 2 * c.comfort_decel * max(0.0, w.stop_distance - c.stop_margin_m)) ** 0.5)
        lc = self.lc_direction if s in (BehaviorState.PREPARE_LANE_CHANGE, BehaviorState.LANE_CHANGE) else 0
        return BehaviorDecision(s, v, lc, reason, w.stop_distance)

    # TODO(planning): traffic-light/sign rules, yielding at intersections, explicit
    # minimum-risk-maneuver (pull over), learned policy behind the same interface.
