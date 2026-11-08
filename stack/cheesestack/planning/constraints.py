"""Constraint evaluation for candidate trajectories.

Checks curvature, lateral acceleration (v^2 * kappa), longitudinal accel bounds,
longitudinal jerk (finite difference of a), speed bounds and collision against
oriented obstacle boxes propagated with constant velocity. Ego footprint is an
oriented box whose centre is `rear_to_center` ahead of base_link (rear axle).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cheesestack.core import Trajectory

from .common import ObstacleBox, boxes_overlap


@dataclass
class ConstraintConfig:
    max_curvature: float = 0.2  # 1/m (~5 m radius)
    max_lateral_accel: float = 3.0
    max_accel: float = 2.5
    min_accel: float = -6.0
    max_jerk: float = 8.0
    max_speed: float = 35.0
    ego_length: float = 4.7
    ego_width: float = 1.9
    rear_to_center: float = 1.4
    collision_margin: float = 0.3
    check_collision_every: int = 1  # check every k-th point (speed/accuracy trade)


@dataclass
class ConstraintReport:
    feasible: bool
    violations: list = field(default_factory=list)  # (name, index, value)
    collision_index: Optional[int] = None
    collision_track: Optional[int] = None
    max_abs_curvature: float = 0.0
    max_lateral_accel: float = 0.0
    max_abs_jerk: float = 0.0

    @property
    def names(self) -> set:
        return {v[0] for v in self.violations}


def ego_box_corners(x: float, y: float, yaw: float, cfg: ConstraintConfig, inflate: float = 0.0) -> np.ndarray:
    cx = x + cfg.rear_to_center * np.cos(yaw)
    cy = y + cfg.rear_to_center * np.sin(yaw)
    return ObstacleBox(cx, cy, yaw, cfg.ego_length, cfg.ego_width).corners(inflate)


class ConstraintChecker:
    def __init__(self, config: ConstraintConfig = ConstraintConfig()) -> None:
        self.config = config

    def check_array(self, arr: np.ndarray, obstacles: Optional[list[ObstacleBox]] = None, stop_at_first: bool = False) -> ConstraintReport:
        """arr: (N, 7) as from Trajectory.as_array(): t, x, y, yaw, v, a, curvature."""
        c = self.config
        rep = ConstraintReport(True)
        if len(arr) == 0:
            return rep
        t, x, y, yaw, v, a, k = arr.T
        lat = v**2 * np.abs(k)
        rep.max_abs_curvature = float(np.abs(k).max())
        rep.max_lateral_accel = float(lat.max())

        def flag(name, mask, vals):
            idx = np.flatnonzero(mask)
            if len(idx):
                rep.violations.append((name, int(idx[0]), float(vals[idx[0]])))

        flag("curvature", np.abs(k) > c.max_curvature, k)
        flag("lateral_accel", lat > c.max_lateral_accel, lat)
        flag("accel_max", a > c.max_accel, a)
        flag("accel_min", a < c.min_accel, a)
        flag("speed", (v > c.max_speed) | (v < -1e-3), v)
        if len(arr) >= 2:
            dt = np.diff(t)
            j = np.diff(a) / np.where(dt > 1e-9, dt, np.inf)
            rep.max_abs_jerk = float(np.abs(j).max())
            flag("jerk", np.abs(j) > c.max_jerk, j)
        if rep.violations and stop_at_first:
            rep.feasible = False
            return rep
        if obstacles:
            step = max(1, c.check_collision_every)
            for i in range(0, len(arr), step):
                ego = ego_box_corners(x[i], y[i], yaw[i], c, c.collision_margin)
                hit = None
                for ob in obstacles:
                    ob_t = ob.at(t[i])
                    # cheap circle pre-check
                    if np.hypot(ob_t.x - x[i], ob_t.y - y[i]) > (ob.length + c.ego_length + c.rear_to_center) + 2 * c.collision_margin:
                        continue
                    if boxes_overlap(ego, ob_t.corners()):
                        hit = ob
                        break
                if hit is not None:
                    rep.collision_index, rep.collision_track = i, hit.track_id
                    rep.violations.append(("collision", i, float(t[i])))
                    break
        rep.feasible = not rep.violations
        return rep

    def check(self, traj: Trajectory, obstacles: Optional[list[ObstacleBox]] = None) -> ConstraintReport:
        return self.check_array(traj.as_array(), obstacles)

    # TODO(planning): obstacle prediction uncertainty (inflate by covariance growth),
    # multi-modal predicted trajectories, road boundary / drivable-area check,
    # steering-rate (curvature-rate) limit from the vehicle layer.
