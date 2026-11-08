"""Planar geometry. Convention: x forward, y left, yaw CCW from +x, radians."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Vec2 = np.ndarray  # shape (2,)


def wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rot2(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


@dataclass(frozen=True)
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw])

    def compose(self, other: "Pose2D") -> "Pose2D":
        """self * other: express `other` (given in self's frame) in self's parent frame."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return Pose2D(
            self.x + c * other.x - s * other.y,
            self.y + s * other.x + c * other.y,
            wrap_angle(self.yaw + other.yaw),
        )

    def inverse(self) -> "Pose2D":
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return Pose2D(-c * self.x - s * self.y, s * self.x - c * self.y, wrap_angle(-self.yaw))

    def transform_point(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return rot2(self.yaw) @ p + np.array([self.x, self.y])

    def transform_points(self, pts) -> np.ndarray:
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        return pts @ rot2(self.yaw).T + np.array([self.x, self.y])


SE2 = Pose2D
