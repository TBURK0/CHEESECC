"""Cross-layer message types. Layers may define richer local types, but anything
crossing a layer boundary should use (or extend) these."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .geometry import Pose2D
from .time import Header


Confidence = float  # always in [0, 1]


class ObjectClass(enum.Enum):
    UNKNOWN = "unknown"
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    PEDESTRIAN = "pedestrian"
    ANIMAL = "animal"
    TRAFFIC_SIGN = "traffic_sign"
    TRAFFIC_LIGHT = "traffic_light"
    STATIC_OBSTACLE = "static_obstacle"


@dataclass
class BoundingBox2D:
    """Image-space box, pixels. (u, v) is top-left."""

    u: float
    v: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def iou(self, other: "BoundingBox2D") -> float:
        x1, y1 = max(self.u, other.u), max(self.v, other.v)
        x2 = min(self.u + self.w, other.u + other.w)
        y2 = min(self.v + self.h, other.v + other.h)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class BoundingBox3D:
    """Ground-plane oriented box in a metric frame (center, size, yaw)."""

    pose: Pose2D
    length: float
    width: float
    height: float = 1.5


@dataclass
class Detection:
    header: Header
    cls: ObjectClass
    score: Confidence
    box2d: Optional[BoundingBox2D] = None
    box3d: Optional[BoundingBox3D] = None
    attributes: dict = field(default_factory=dict)


@dataclass
class TrackedObject:
    track_id: int
    header: Header
    cls: ObjectClass
    pose: Pose2D
    velocity: np.ndarray  # (vx, vy) in header.frame_id
    length: float = 4.5
    width: float = 1.8
    confidence: Confidence = 1.0
    covariance: Optional[np.ndarray] = None  # 4x4 over (x, y, vx, vy)
    age: int = 0


@dataclass
class VehicleState:
    """Ego state in the `map`/`odom` frame."""

    header: Header
    pose: Pose2D
    v: float = 0.0  # longitudinal speed m/s
    a: float = 0.0  # longitudinal accel m/s^2
    yaw_rate: float = 0.0  # rad/s
    steer: float = 0.0  # road-wheel angle rad
    curvature: float = 0.0  # 1/m
    covariance: Optional[np.ndarray] = None


@dataclass
class TrajectoryPoint:
    t: float  # seconds relative to trajectory header stamp
    x: float
    y: float
    yaw: float
    v: float
    a: float = 0.0
    curvature: float = 0.0


@dataclass
class Trajectory:
    header: Header
    points: list[TrajectoryPoint] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.points)

    def as_array(self) -> np.ndarray:
        """(N, 7) array of t, x, y, yaw, v, a, curvature."""
        return np.array([[p.t, p.x, p.y, p.yaw, p.v, p.a, p.curvature] for p in self.points]).reshape(-1, 7)


@dataclass
class ControlCommand:
    """Actuator request. Mirrors the app's DriveCommand fields at a high level."""

    header: Header
    steer_wheel_deg: float = 0.0
    throttle: float = 0.0  # [0, 1]
    brake: float = 0.0  # [0, 1]
    curvature: float = 0.0
    target_speed_mps: float = 0.0
    source: str = "control"


class Severity(enum.IntEnum):
    INFO = 0
    WARNING = 1
    DEGRADED = 2
    CRITICAL = 3
    FATAL = 4


@dataclass
class Fault:
    code: str
    severity: Severity
    source: str
    stamp: float
    message: str = ""
    data: dict = field(default_factory=dict)
