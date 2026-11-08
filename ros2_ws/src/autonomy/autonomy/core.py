"""ROS-independent local autonomy primitives. Metres, seconds, REP-103 base_link.

The map is a single-frame local snapshot, not SLAM. Unknown cells stay unknown.
"""
import heapq
import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    x: float
    y: float


def finite_point(p):
    return math.isfinite(p.x) and math.isfinite(p.y)


def depth_points(rows, fx, fy, cx, cy, scale=0.001, stride=4,
                 min_depth=0.1, max_depth=4.0, camera_height=0.2):
    """Aligned optical depth -> base_link, assuming a level forward camera.

    Optical x right, y down, z forward. Excludes floor and above-car returns.
    Mount translation/rotation beyond height must be calibrated before deployment.
    """
    if not all(math.isfinite(v) for v in (fx, fy, cx, cy, scale, camera_height)):
        raise ValueError('non-finite camera calibration')
    if fx <= 0 or fy <= 0 or scale <= 0 or stride < 1:
        raise ValueError('invalid camera calibration or stride')
    result = []
    for v in range(0, len(rows), stride):
        for u in range(0, len(rows[v]), stride):
            z = rows[v][u] * scale
            if not math.isfinite(z) or not min_depth <= z <= max_depth:
                continue
            height = camera_height - (v - cy) * z / fy
            if 0.04 <= height <= 0.6:
                result.append(Point(z, -(u - cx) * z / fx))
    return result


class LocalMap:
    def __init__(self, resolution=0.1, width=60, height=60):
        if resolution <= 0 or width < 3 or height < 3:
            raise ValueError('invalid map dimensions')
        self.resolution, self.width, self.height = resolution, width, height
        self.origin = Point(-width * resolution / 2, -height * resolution / 2)
        self.data = [-1] * (width * height)

    def cell(self, p):
        x = math.floor((p.x - self.origin.x) / self.resolution)
        y = math.floor((p.y - self.origin.y) / self.resolution)
        return (x, y) if 0 <= x < self.width and 0 <= y < self.height else None

    def center(self, c):
        return Point(self.origin.x + (c[0] + 0.5) * self.resolution,
                     self.origin.y + (c[1] + 0.5) * self.resolution)

    def update(self, points, origin=Point(0, 0)):
        """Ray clear only to measured endpoints. Rebuild to avoid ego-motion ghosts."""
        self.data = [-1] * len(self.data)
        endpoints = []
        rays = {}
        if not finite_point(origin):
            return
        for p in points:
            if not finite_point(p):
                continue
            end = self.cell(p)
            if end is None:
                continue
            if end not in rays or math.hypot(p.x-origin.x, p.y-origin.y) < math.hypot(rays[end].x-origin.x, rays[end].y-origin.y):
                rays[end] = p
        for end, p in rays.items():
            endpoints.append(end)
            if self.cell(origin) is None:
                continue
            steps = max(1, math.ceil(math.hypot(p.x - origin.x, p.y - origin.y) / (self.resolution / 2)))
            for i in range(steps):
                c = self.cell(Point(origin.x + (p.x - origin.x) * i / steps,
                                    origin.y + (p.y - origin.y) * i / steps))
                if c:
                    self.data[c[1] * self.width + c[0]] = 0
        # Occupied wins where rays intersect.
        for x, y in endpoints:
            self.data[y * self.width + x] = 100

    def plan(self, goal, clearance=0.25):
        """4-connected A*, restricted to observed free space with obstacle inflation."""
        if not finite_point(goal) or clearance < 0:
            return []
        start, target = self.cell(Point(0, 0)), self.cell(goal)
        blocked = set()
        radius = math.ceil(clearance / self.resolution)
        for i, value in enumerate(self.data):
            if value != 100:
                continue
            x, y = i % self.width, i // self.width
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if math.hypot(dx, dy) * self.resolution <= clearance + self.resolution / 2:
                        blocked.add((x + dx, y + dy))
        def free(c):
            return (c is not None and 0 <= c[0] < self.width and
                    0 <= c[1] < self.height and c not in blocked and
                    self.data[c[1] * self.width + c[0]] == 0)
        if not free(start) or not free(target):
            return []
        queue, costs, parents = [(0, start)], {start: 0}, {}
        while queue:
            _, c = heapq.heappop(queue)
            if c == target:
                path = [c]
                while c in parents:
                    c = parents[c]
                    path.append(c)
                return [self.center(c) for c in reversed(path)]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (c[0] + dx, c[1] + dy)
                cost = costs[c] + 1
                if free(n) and cost < costs.get(n, math.inf):
                    costs[n], parents[n] = cost, c
                    heapq.heappush(queue, (cost + abs(n[0] - target[0]) + abs(n[1] - target[1]), n))
        return []


def scene_summary(points, corridor_half_width=0.3):
    """Geometric scene understanding; does not infer semantic object classes."""
    valid = [p for p in points if finite_point(p)]
    front = [p.x for p in valid if p.x >= 0 and abs(p.y) <= corridor_half_width]
    return {'obstacle_count': len(valid), 'front_clearance_m': min(front) if front else None,
            'left_count': sum(p.y > corridor_half_width for p in valid),
            'right_count': sum(p.y < -corridor_half_width for p in valid)}


def safe_command(throttle, steering, speed, points, now, observation_time,
                 link_ok, armed, estop=False, timeout=0.3):
    """Fail closed on stale/invalid input; stopping distance includes reaction margin."""
    reason = None
    if not all(math.isfinite(v) for v in (throttle, steering, speed, now, observation_time)):
        reason = 'invalid_input'
    elif estop or not link_ok or not armed:
        reason = 'vehicle_unavailable'
    elif not 0 <= now - observation_time <= timeout:
        reason = 'stale_scene'
    elif any(not finite_point(p) for p in points):
        reason = 'invalid_scene'
    else:
        distance = scene_summary(points)['front_clearance_m']
        stopping = 0.25 + abs(speed) * 0.3 + speed * speed / (2 * 0.8)
        if distance is not None and distance <= stopping:
            reason = 'obstacle'
    if reason:
        return {'throttle': 0.0, 'brake': 1.0, 'steering': 0.0, 'reason': reason}
    return {'throttle': min(0.25, max(0.0, throttle)), 'brake': 0.0,
            'steering': min(1.0, max(-1.0, steering)), 'reason': 'ok'}


class CommandInbox:
    """Bounded JSON command validation for trusted local transports, not authentication.

    Sender and receiver must share a clock. Sequence numbers never reset in-session.
    No ARM command is accepted here; arming remains an explicit bridge operation.
    """
    def __init__(self):
        self.last_sequence = -1

    def receive(self, line, now):
        if not isinstance(line, str) or len(line) > 1024 or not math.isfinite(now):
            raise ValueError('invalid envelope')
        try:
            msg = json.loads(line)
            if set(msg) not in ({'seq', 'stamp', 'throttle', 'steering'},
                                {'seq', 'stamp', 'throttle', 'steering', 'brake'}):
                raise ValueError('unexpected command fields')
            seq = msg['seq']
            if type(seq) is not int or seq <= self.last_sequence or seq < 0:
                raise ValueError('replayed command')
            for key in ('stamp', 'throttle', 'steering'):
                if type(msg[key]) not in (int, float) or not math.isfinite(msg[key]):
                    raise ValueError('invalid numeric field')
            if not 0 <= now - msg['stamp'] <= 0.2:
                raise ValueError('expired command')
            if not 0 <= msg['throttle'] <= 0.25 or not -1 <= msg['steering'] <= 1:
                raise ValueError('command outside limits')
            if 'brake' in msg and (type(msg['brake']) not in (int, float) or
                                   not math.isfinite(msg['brake']) or not 0 <= msg['brake'] <= 1):
                raise ValueError('invalid brake')
        except (TypeError, KeyError, OverflowError, RecursionError) as exc:
            raise ValueError('invalid command') from exc
        self.last_sequence = seq
        return msg
