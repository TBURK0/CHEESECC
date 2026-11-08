"""Pose tracking: odom dead reckoning + map->odom correction, with time-indexed history.

Mirrors the ROS REP-105 split:
- `odom -> base_link` is continuous (integrated odometry deltas, never jumps).
- `map -> odom` absorbs absolute corrections (GPS / map matching), may jump.
- `map -> base_link = (map->odom) * (odom->base_link)`.

The tracker keeps a history of odom poses so a delayed absolute fix stamped
in the past can be applied correctly: the correction is computed against the
odom pose *at the fix time*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from cheesestack.core import Pose2D, wrap_angle

from .time_sync import InterpolationBuffer


@dataclass
class PoseTrackerConfig:
    history_s: float = 5.0
    correction_gain: float = 1.0  # 1 = snap; <1 = smooth blend of map->odom
    max_correction_m: float = 50.0  # larger jumps are rejected (relocalization needed)


class PoseTracker:
    def __init__(self, config: Optional[PoseTrackerConfig] = None) -> None:
        self.config = config or PoseTrackerConfig()
        self.odom_pose = Pose2D()
        self.map_to_odom = Pose2D()
        self.stamp: Optional[float] = None
        self._hist = InterpolationBuffer(self.config.history_s, max_extrapolation_s=0.05)
        self.distance_m = 0.0

    @property
    def map_pose(self) -> Pose2D:
        return self.map_to_odom.compose(self.odom_pose)

    def reset(self, stamp: float, map_pose: Pose2D) -> None:
        self.odom_pose = Pose2D()
        self.map_to_odom = map_pose
        self.stamp = stamp
        self._hist = InterpolationBuffer(self.config.history_s, max_extrapolation_s=0.05)
        self._hist.add(stamp, self.odom_pose)

    def add_delta(self, stamp: float, delta: Pose2D) -> Pose2D:
        """Apply a body-frame motion increment (e.g. from wheel odom or VO)."""
        self.odom_pose = self.odom_pose.compose(delta)
        self.distance_m += math.hypot(delta.x, delta.y)
        self.stamp = stamp
        self._hist.add(stamp, self.odom_pose)
        return self.map_pose

    def add_velocity(self, stamp: float, v: float, yaw_rate: float) -> Pose2D:
        """Integrate (v, yaw_rate) since the last stamp using the exact arc."""
        if self.stamp is None:
            self.stamp = stamp
            self._hist.add(stamp, self.odom_pose)
            return self.map_pose
        dt = stamp - self.stamp
        if dt <= 0:
            return self.map_pose
        dth = yaw_rate * dt
        if abs(dth) > 1e-9:
            r = v / yaw_rate
            delta = Pose2D(r * math.sin(dth), r * (1 - math.cos(dth)), dth)
        else:
            delta = Pose2D(v * dt, 0.0, 0.0)
        return self.add_delta(stamp, delta)

    def odom_at(self, t: float) -> Optional[Pose2D]:
        return self._hist.interpolate(t)

    def map_pose_at(self, t: float) -> Optional[Pose2D]:
        o = self.odom_at(t)
        return None if o is None else self.map_to_odom.compose(o)

    def correct(self, stamp: float, map_pose: Pose2D, use_yaw: bool = True) -> bool:
        """Absolute map pose at `stamp`. Updates map->odom. Returns False if rejected."""
        odom_then = self.odom_at(stamp)
        if odom_then is None:
            odom_then = self.odom_pose
        if not use_yaw:
            cur = self.map_to_odom.compose(odom_then)
            map_pose = Pose2D(map_pose.x, map_pose.y, cur.yaw)
        target = map_pose.compose(odom_then.inverse())
        cur = self.map_to_odom
        dx, dy = target.x - cur.x, target.y - cur.y
        if math.hypot(dx, dy) > self.config.max_correction_m:
            return False
        g = self.config.correction_gain
        self.map_to_odom = Pose2D(cur.x + g * dx, cur.y + g * dy, wrap_angle(cur.yaw + g * wrap_angle(target.yaw - cur.yaw)))
        return True
    # TODO(localization): covariance propagation for the odom chain, loop-closure / pose-graph
    # smoothing of map->odom, multi-session map anchoring.
