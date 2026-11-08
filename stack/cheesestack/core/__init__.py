"""Shared primitives every layer depends on. Keep this small and stable."""

from .time import Clock, SimClock, WallClock, Stamped, Header
from .geometry import Pose2D, Vec2, wrap_angle, rot2, SE2
from .types import (
    Confidence,
    ObjectClass,
    BoundingBox2D,
    BoundingBox3D,
    Detection,
    TrackedObject,
    VehicleState,
    TrajectoryPoint,
    Trajectory,
    ControlCommand,
    Severity,
    Fault,
)
from .bus import MessageBus, Subscription
from .node import Node, NodeStatus

__all__ = [
    "Clock", "SimClock", "WallClock", "Stamped", "Header",
    "Pose2D", "Vec2", "wrap_angle", "rot2", "SE2",
    "Confidence", "ObjectClass", "BoundingBox2D", "BoundingBox3D", "Detection",
    "TrackedObject", "VehicleState", "TrajectoryPoint", "Trajectory",
    "ControlCommand", "Severity", "Fault",
    "MessageBus", "Subscription", "Node", "NodeStatus",
]
