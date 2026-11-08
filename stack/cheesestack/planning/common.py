"""Shared planning primitives: reference path with Frenet conversion, polynomials, boxes.

These are planning-local types (not in `core`) because only the planner needs them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from cheesestack.core import Pose2D, TrackedObject, wrap_angle


# ---------------------------------------------------------------------------
# Polynomials
# ---------------------------------------------------------------------------


class QuinticPolynomial:
    """x(t) = sum a_i t^i, i=0..5, matching position/velocity/accel at t=0 and t=T.

    The minimum-jerk solution between two full kinematic states.
    """

    def __init__(self, x0: float, v0: float, a0: float, x1: float, v1: float, a1: float, T: float) -> None:
        if T <= 0:
            raise ValueError("T must be positive")
        self.T = float(T)
        self.a = np.zeros(6)
        self.a[0], self.a[1], self.a[2] = x0, v0, a0 / 2.0
        A = np.array(
            [
                [T**3, T**4, T**5],
                [3 * T**2, 4 * T**3, 5 * T**4],
                [6 * T, 12 * T**2, 20 * T**3],
            ]
        )
        b = np.array(
            [
                x1 - (x0 + v0 * T + 0.5 * a0 * T**2),
                v1 - (v0 + a0 * T),
                a1 - a0,
            ]
        )
        self.a[3:] = np.linalg.solve(A, b)

    def _eval(self, t, d: int):
        t = np.asarray(t, dtype=float)
        coeffs = self.a
        for _ in range(d):
            coeffs = np.polynomial.polynomial.polyder(coeffs)
        return np.polynomial.polynomial.polyval(t, coeffs)

    def x(self, t):
        return self._eval(t, 0)

    def v(self, t):
        return self._eval(t, 1)

    def acc(self, t):
        return self._eval(t, 2)

    def jerk(self, t):
        return self._eval(t, 3)


class QuarticPolynomial(QuinticPolynomial):
    """x(t) quartic matching x0, v0, a0 at t=0 and v1, a1 at t=T (free end position).

    Used for velocity keeping in the longitudinal direction.
    """

    def __init__(self, x0: float, v0: float, a0: float, v1: float, a1: float, T: float) -> None:  # noqa: D401
        if T <= 0:
            raise ValueError("T must be positive")
        self.T = float(T)
        self.a = np.zeros(5)
        self.a[0], self.a[1], self.a[2] = x0, v0, a0 / 2.0
        A = np.array([[3 * T**2, 4 * T**3], [6 * T, 12 * T**2]])
        b = np.array([v1 - (v0 + a0 * T), a1 - a0])
        self.a[3:] = np.linalg.solve(A, b)


# ---------------------------------------------------------------------------
# Reference path / Frenet frame
# ---------------------------------------------------------------------------


class ReferencePath:
    """Piecewise-linear reference polyline with arc length, heading and curvature.

    Frenet convention: s along the path, d positive to the left (CCW normal).
    Projection is exact nearest-segment search (O(N) per query) — fine for
    short local paths. # TODO(planning): spline reference + KD/windowed search.
    """

    def __init__(self, points) -> None:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(pts) < 2:
            raise ValueError("ReferencePath needs at least 2 points")
        # Drop duplicate consecutive points.
        keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-9]
        pts = pts[keep]
        if len(pts) < 2:
            raise ValueError("ReferencePath points are degenerate")
        self.points = pts
        seg = np.diff(pts, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.s = np.r_[0.0, np.cumsum(self.seg_len)]
        self.seg_yaw = np.arctan2(seg[:, 1], seg[:, 0])
        # Vertex heading: unwrapped mean of adjacent segments.
        yaw_u = np.unwrap(self.seg_yaw)
        vy = np.empty(len(pts))
        vy[0], vy[-1] = yaw_u[0], yaw_u[-1]
        if len(pts) > 2:
            vy[1:-1] = 0.5 * (yaw_u[:-1] + yaw_u[1:])
        self.yaw_unwrapped = vy
        self.yaw = np.array([wrap_angle(a) for a in vy])
        if len(pts) > 2:
            self.kappa = np.gradient(vy, self.s)
        else:
            self.kappa = np.zeros(len(pts))

    @property
    def length(self) -> float:
        return float(self.s[-1])

    @classmethod
    def straight(cls, length: float, step: float = 1.0, start: Pose2D = Pose2D()) -> "ReferencePath":
        s = np.arange(0.0, length + 1e-9, step)
        local = np.stack([s, np.zeros_like(s)], axis=1)
        return cls(start.transform_points(local))

    def interpolate(self, s):
        """Return (x, y, yaw, kappa) arrays at arc lengths s (clamped extrapolation linear)."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        idx = np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self.seg_len) - 1)
        ds = s - self.s[idx]
        yaw_seg = self.seg_yaw[idx]
        x = self.points[idx, 0] + ds * np.cos(yaw_seg)
        y = self.points[idx, 1] + ds * np.sin(yaw_seg)
        sc = np.clip(s, 0.0, self.length)
        yaw = np.interp(sc, self.s, self.yaw_unwrapped)
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
        kappa = np.interp(sc, self.s, self.kappa)
        return x, y, yaw, kappa

    def to_cartesian(self, s, d):
        s = np.atleast_1d(np.asarray(s, dtype=float))
        d = np.broadcast_to(np.asarray(d, dtype=float), s.shape)
        x, y, yaw, _ = self.interpolate(s)
        return x - d * np.sin(yaw), y + d * np.cos(yaw)

    def to_frenet(self, x: float, y: float) -> tuple[float, float]:
        p = np.array([x, y], dtype=float)
        a = self.points[:-1]
        seg = np.diff(self.points, axis=0)
        L2 = self.seg_len**2
        t = np.clip(np.einsum("ij,ij->i", p - a, seg) / L2, 0.0, 1.0)
        # allow extrapolation on first/last segments
        t_raw = np.einsum("ij,ij->i", p - a, seg) / L2
        t[0] = min(t_raw[0], 1.0)
        t[-1] = max(t_raw[-1], 0.0) if len(t) > 1 else t_raw[-1]
        proj = a + t[:, None] * seg
        dist = np.linalg.norm(p - proj, axis=1)
        i = int(np.argmin(dist))
        s = self.s[i] + t[i] * self.seg_len[i]
        rel = p - proj[i]
        cross = seg[i, 0] * rel[1] - seg[i, 1] * rel[0]
        d = math.copysign(dist[i], cross) if dist[i] > 0 else 0.0
        return float(s), float(d)


# ---------------------------------------------------------------------------
# Obstacles
# ---------------------------------------------------------------------------


@dataclass
class ObstacleBox:
    """Oriented box obstacle with constant-velocity motion (map/odom frame)."""

    x: float
    y: float
    yaw: float
    length: float = 4.5
    width: float = 1.8
    vx: float = 0.0
    vy: float = 0.0
    track_id: Optional[int] = None

    @classmethod
    def from_tracked(cls, obj: TrackedObject) -> "ObstacleBox":
        v = np.asarray(obj.velocity, dtype=float).reshape(-1)
        return cls(obj.pose.x, obj.pose.y, obj.pose.yaw, obj.length, obj.width, float(v[0]), float(v[1]), obj.track_id)

    def at(self, t: float) -> "ObstacleBox":
        return ObstacleBox(self.x + self.vx * t, self.y + self.vy * t, self.yaw, self.length, self.width, self.vx, self.vy, self.track_id)

    def corners(self, inflate: float = 0.0) -> np.ndarray:
        hl, hw = self.length / 2 + inflate, self.width / 2 + inflate
        local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
        return Pose2D(self.x, self.y, self.yaw).transform_points(local)


def boxes_overlap(c1: np.ndarray, c2: np.ndarray) -> bool:
    """Separating-axis test for two convex quads given as (4,2) corner arrays."""
    for c in (c1, c2):
        for i in range(4):
            edge = c[(i + 1) % 4] - c[i]
            axis = np.array([-edge[1], edge[0]])
            p1, p2 = c1 @ axis, c2 @ axis
            if p1.max() < p2.min() or p2.max() < p1.min():
                return False
    return True


def object_frenet(obj: TrackedObject, ref: Optional[ReferencePath] = None) -> tuple[float, float, float]:
    """(s, d, v_s) of a tracked object. Without a reference path, assumes a straight
    road along +x of the object's frame (base_link-style)."""
    v = np.asarray(obj.velocity, dtype=float).reshape(-1)
    if ref is None:
        return obj.pose.x, obj.pose.y, float(v[0])
    s, d = ref.to_frenet(obj.pose.x, obj.pose.y)
    _, _, yaw, _ = ref.interpolate(s)
    vs = float(v[0] * math.cos(yaw[0]) + v[1] * math.sin(yaw[0]))
    return s, d, vs
