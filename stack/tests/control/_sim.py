"""Tiny kinematic plant + path builders shared by control tests."""
import math

import numpy as np

from cheesestack.core import Header, Pose2D, Trajectory, TrajectoryPoint, VehicleState


def straight_path(length=200.0, n=401, y=0.0, v=0.0):
    xs = np.linspace(0, length, n)
    return Trajectory(Header(0.0, "map"), [TrajectoryPoint(0.0, float(x), y, 0.0, v) for x in xs])


def circle_xy(radius=30.0, n=600, arc=1.5 * math.pi):
    th = np.linspace(0, arc, n)
    xy = np.stack([radius * np.sin(th), radius * (1 - np.cos(th))], axis=1)
    return xy, th  # CCW (left turn) circle starting at origin heading +x


def step_kinematic(pose, v, kappa, dt, wheelbase=2.83):
    th = pose.yaw
    x = pose.x + v * math.cos(th) * dt
    y = pose.y + v * math.sin(th) * dt
    return Pose2D(x, y, th + v * kappa * dt)


def state(pose, v, t=0.0, yaw_rate=0.0, curvature=0.0):
    return VehicleState(Header(t, "map"), pose, v=v, yaw_rate=yaw_rate, curvature=curvature)
