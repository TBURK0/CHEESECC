"""Calibrated rigid transforms, odometry-frame memory and sensor fusion."""
import math
from dataclasses import dataclass

from .core import LocalMap, Point, finite_point


@dataclass(frozen=True)
class Pose2:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def world(self, p):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return Point(self.x + c * p.x - s * p.y, self.y + s * p.x + c * p.y)

    def local(self, p):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        x, y = p.x - self.x, p.y - self.y
        return Point(c * x + s * y, -s * x + c * y)


class RigidTransform:
    def __init__(self, translation, quaternion):
        if len(translation) != 3 or len(quaternion) != 4:
            raise ValueError('invalid transform dimensions')
        if not all(math.isfinite(v) for v in tuple(translation) + tuple(quaternion)):
            raise ValueError('non-finite transform')
        norm = math.sqrt(sum(v * v for v in quaternion))
        if norm < 1e-9:
            raise ValueError('zero quaternion')
        self.t = translation
        self.q = tuple(v / norm for v in quaternion)

    def apply(self, p):
        x, y, z, w = self.q
        px, py, pz = p
        # q * p * conjugate(q), expanded without a matrix dependency.
        tx, ty, tz = 2 * (y * pz - z * py), 2 * (z * px - x * pz), 2 * (x * py - y * px)
        return (px + w * tx + y * tz - z * ty + self.t[0],
                py + w * ty + z * tx - x * tz + self.t[1],
                pz + w * tz + x * ty - y * tx + self.t[2])


def calibrated_depth(rows, fx, fy, cx, cy, scale, transform, stride=4):
    if not all(math.isfinite(v) for v in (fx, fy, cx, cy, scale)) or min(fx, fy, scale) <= 0:
        raise ValueError('invalid intrinsics')
    points = []
    for v in range(0, len(rows), stride):
        for u in range(0, len(rows[v]), stride):
            z = rows[v][u] * scale
            if not math.isfinite(z) or not 0.1 <= z <= 4.0:
                continue
            x, y, height = transform.apply(((u - cx) * z / fx, (v - cy) * z / fy, z))
            if 0.04 <= height <= 0.6:
                points.append(Point(x, y))
    return points


def ultrasonic_endpoint(distance, minimum, maximum, transform):
    """ROS Range uses sensor +x. +/-inf and NaN provide no obstacle endpoint."""
    if not math.isfinite(distance) or not minimum <= distance <= maximum:
        return None
    x, y, _ = transform.apply((distance, 0, 0))
    return Point(x, y)


class ObstacleMemory:
    """Short-lived obstacle tracks in odom, bounded in size and lifetime.

    Requires external odometry; this is not a SLAM/localization estimator. Memory
    points only add occupied cells, never certify old free space as currently clear.
    """
    def __init__(self, ttl=0.5, merge_distance=0.1, max_tracks=2000):
        self.ttl, self.merge_distance, self.max_tracks = ttl, merge_distance, max_tracks
        self.tracks = []
        self.last_time = None

    def update(self, points, pose, now):
        if not math.isfinite(now) or not all(math.isfinite(v) for v in (pose.x, pose.y, pose.yaw)):
            raise ValueError('invalid odometry or time')
        if self.last_time is not None and now < self.last_time:
            self.tracks = []
        self.last_time = now
        self.tracks = [(p, t) for p, t in self.tracks if 0 <= now - t <= self.ttl]
        bins = {}
        def cell(p):
            return math.floor(p.x / self.merge_distance), math.floor(p.y / self.merge_distance)
        for i, (p, _) in enumerate(self.tracks):
            bins.setdefault(cell(p), []).append(i)
        for p in points:
            if not finite_point(p):
                continue
            p = pose.world(p)
            x, y = cell(p)
            candidates = [i for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                          for i in bins.get((x+dx, y+dy), [])]
            match = next((i for i in candidates if math.hypot(p.x-self.tracks[i][0].x,
                                                             p.y-self.tracks[i][0].y) < self.merge_distance), None)
            if match is None:
                bins.setdefault((x, y), []).append(len(self.tracks))
                self.tracks.append((p, now))
            else:
                previous = cell(self.tracks[match][0])
                if previous != (x, y):
                    bins[previous].remove(match)
                    bins.setdefault((x, y), []).append(match)
                self.tracks[match] = (p, now)
        self.tracks = self.tracks[-self.max_tracks:]
        return [pose.local(p) for p, _ in self.tracks]

    def snapshot(self, pose, now):
        return [pose.local(p) for p, t in self.tracks if 0 <= now - t <= self.ttl]


def fuse_points(camera, ranges, voxel=0.05):
    """Conservative union: a near-field return is never averaged behind a far return."""
    cells = {}
    for p in list(camera) + list(ranges):
        if p is not None and finite_point(p):
            key = (math.floor(p.x / voxel), math.floor(p.y / voxel))
            if key not in cells or math.hypot(p.x, p.y) < math.hypot(cells[key].x, cells[key].y):
                cells[key] = p
    return list(cells.values())


def mark_memory_occupied(grid, points):
    for p in points:
        c = grid.cell(p)
        if c:
            grid.data[c[1] * grid.width + c[0]] = 100
