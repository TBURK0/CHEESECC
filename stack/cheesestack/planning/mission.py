"""Mission planning: an ordered list of goals and a small lifecycle state machine.

States: IDLE -> ACTIVE -> ARRIVED, or ABORTED from IDLE/ACTIVE. A goal is reached
when the ego is inside its position tolerance (and heading tolerance, if set)
and slow enough. This mirrors the app's "route + destination" notion at the
mission level; the route itself comes from `route.RoutePlanner`.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Optional

from cheesestack.core import Pose2D, wrap_angle


class MissionState(enum.Enum):
    IDLE = "idle"
    ACTIVE = "active"
    ARRIVED = "arrived"
    ABORTED = "aborted"


@dataclass
class Goal:
    name: str
    pose: Pose2D
    tolerance_m: float = 3.0
    yaw_tolerance: Optional[float] = None  # rad; None = ignore heading
    max_speed_mps: Optional[float] = None  # must be at or below this to count as reached


@dataclass
class MissionConfig:
    default_tolerance_m: float = 3.0
    arrive_speed_mps: float = 0.5  # used for the final goal when it has no max_speed


@dataclass
class MissionStatus:
    state: MissionState
    goal_index: int
    current_goal: Optional[Goal]
    distance_to_goal: float
    reason: str = ""


class Mission:
    """Ordered goals with progress tracking. Pure logic; feed it ego pose/speed."""

    def __init__(self, goals: list[Goal], config: MissionConfig = MissionConfig()) -> None:
        self.config = config
        self.goals = list(goals)
        self.state = MissionState.IDLE
        self.index = 0
        self.reason = ""
        self.reached: list[str] = []

    @property
    def current_goal(self) -> Optional[Goal]:
        if self.state in (MissionState.ACTIVE, MissionState.IDLE) and self.index < len(self.goals):
            return self.goals[self.index]
        return None

    def start(self) -> None:
        if self.state != MissionState.IDLE:
            raise RuntimeError(f"cannot start mission in state {self.state.value}")
        if not self.goals:
            self.abort("no goals")
            return
        self.state = MissionState.ACTIVE

    def abort(self, reason: str = "aborted") -> None:
        if self.state in (MissionState.IDLE, MissionState.ACTIVE):
            self.state = MissionState.ABORTED
            self.reason = reason

    def _reached(self, goal: Goal, pose: Pose2D, speed: float, is_last: bool) -> bool:
        if math.hypot(goal.pose.x - pose.x, goal.pose.y - pose.y) > goal.tolerance_m:
            return False
        if goal.yaw_tolerance is not None and abs(wrap_angle(goal.pose.yaw - pose.yaw)) > goal.yaw_tolerance:
            return False
        vmax = goal.max_speed_mps
        if vmax is None and is_last:
            vmax = self.config.arrive_speed_mps
        if vmax is not None and speed > vmax:
            return False
        return True

    def update(self, pose: Pose2D, speed: float = 0.0) -> MissionStatus:
        """Advance through any goals the ego has reached this tick."""
        if self.state == MissionState.ACTIVE:
            while self.index < len(self.goals):
                g = self.goals[self.index]
                if not self._reached(g, pose, speed, self.index == len(self.goals) - 1):
                    break
                self.reached.append(g.name)
                self.index += 1
            if self.index >= len(self.goals):
                self.state = MissionState.ARRIVED
                self.reason = "all goals reached"
        g = self.current_goal
        dist = math.hypot(g.pose.x - pose.x, g.pose.y - pose.y) if g else 0.0
        return MissionStatus(self.state, self.index, g, dist, self.reason)

    # TODO(planning): mission persistence, re-ordering (TSP) of unordered goals,
    # goal reachability check against the route graph, timeouts per goal.
