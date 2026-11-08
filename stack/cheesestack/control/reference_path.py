"""Reference path utilities shared by the path-tracking controllers.

A `ReferencePath` is a polyline with per-point yaw, curvature and speed, built
from a `core.Trajectory` or raw arrays. It projects a pose onto the path and
returns the signed cross-track error (positive = vehicle LEFT of the path),
heading error (vehicle yaw - path yaw, wrapped), arc length and path curvature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from cheesestack.core import Pose2D, Trajectory, wrap_angle


@dataclass
class PathProjection:
    s: float  # arc length of the projected point, m
    x: float
    y: float
    yaw: float
    curvature: float
    speed: float
    accel: float
    cross_track: float  # signed, +left of path
    heading_error: float  # vehicle yaw - path yaw, rad
    index: int  # segment start index


def _menger_curvature(xy: np.ndarray, yaw: np.ndarray, s: np.ndarray) -> np.ndarray:
    if len(xy) < 3:
        return np.zeros(len(xy))
    dyaw = np.unwrap(yaw)
    k = np.gradient(dyaw, s, edge_order=1)
    return k


class ReferencePath:
    def __init__(
        self,
        xy: np.ndarray,
        yaw: Optional[np.ndarray] = None,
        curvature: Optional[np.ndarray] = None,
        speed: Optional[np.ndarray] = None,
        accel: Optional[np.ndarray] = None,
    ) -> None:
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        if len(xy) < 2:
            raise ValueError("ReferencePath needs at least 2 points")
        self.xy = xy
        seg = np.diff(xy, axis=0)
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.s = np.concatenate([[0.0], np.cumsum(seg_len)])
        if yaw is None:
            seg_yaw = np.arctan2(seg[:, 1], seg[:, 0])
            yaw = np.concatenate([seg_yaw, seg_yaw[-1:]])
        self.yaw = np.asarray(yaw, dtype=float)
        if curvature is None:
            curvature = _menger_curvature(xy, self.yaw, np.maximum.accumulate(self.s + 1e-9 * np.arange(len(xy))))
        self.curvature = np.asarray(curvature, dtype=float)
        n = len(xy)
        self.speed = np.zeros(n) if speed is None else np.asarray(speed, dtype=float)
        self.accel = np.zeros(n) if accel is None else np.asarray(accel, dtype=float)
        self._last_index = 0

    @classmethod
    def from_trajectory(cls, traj: Trajectory) -> "ReferencePath":
        a = traj.as_array()
        has_k = np.any(np.abs(a[:, 6]) > 0)
        return cls(a[:, 1:3], a[:, 3], a[:, 6] if has_k else None, a[:, 4], a[:, 5])

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def project(self, pose: Pose2D, window: Optional[int] = None) -> PathProjection:
        """Nearest-point projection. `window` limits the search around the last index
        (monotone progress, avoids jumping between path overlaps)."""
        p = np.array([pose.x, pose.y])
        a, b = self.xy[:-1], self.xy[1:]
        if window is not None:
            lo = max(0, self._last_index - window)
            hi = min(len(a), self._last_index + window + 1)
        else:
            lo, hi = 0, len(a)
        aa, bb = a[lo:hi], b[lo:hi]
        d = bb - aa
        L2 = np.maximum(np.einsum("ij,ij->i", d, d), 1e-12)
        t = np.clip(np.einsum("ij,ij->i", p - aa, d) / L2, 0.0, 1.0)
        proj = aa + t[:, None] * d
        dist2 = np.sum((proj - p) ** 2, axis=1)
        j = int(np.argmin(dist2))
        i = lo + j
        self._last_index = i
        tt = float(t[j])
        s = self.s[i] + tt * (self.s[i + 1] - self.s[i])
        seg_yaw = math.atan2(d[j, 1], d[j, 0])
        yaw = wrap_angle(self.yaw[i] + tt * wrap_angle(self.yaw[i + 1] - self.yaw[i]))
        px, py = proj[j]
        # Cross-track uses the segment normal (left positive). Beyond the ends, extrapolate along tangent.
        dx, dy = pose.x - px, pose.y - py
        e = -math.sin(seg_yaw) * dx + math.cos(seg_yaw) * dy
        lerp = lambda arr: float(arr[i] + tt * (arr[i + 1] - arr[i]))
        return PathProjection(
            s=float(s), x=float(px), y=float(py), yaw=yaw,
            curvature=lerp(self.curvature), speed=lerp(self.speed), accel=lerp(self.accel),
            cross_track=float(e), heading_error=wrap_angle(pose.yaw - yaw), index=i,
        )

    def point_at(self, s: float) -> tuple[float, float, float]:
        """(x, y, yaw) at arc length s, clamped to the path ends."""
        s = float(np.clip(s, 0.0, self.length))
        x = float(np.interp(s, self.s, self.xy[:, 0]))
        y = float(np.interp(s, self.s, self.xy[:, 1]))
        i = int(np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self.s) - 2))
        return x, y, float(self.yaw[i])

    def value_at(self, name: str, s: float) -> float:
        return float(np.interp(s, self.s, getattr(self, name)))
