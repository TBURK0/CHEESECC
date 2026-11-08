"""Forward bicycle lattice planner, pure pursuit and swept footprint checks."""
import heapq
import math
from dataclasses import dataclass

from .core import Point
from .spatial import Pose2


@dataclass(frozen=True)
class VehicleGeometry:
    wheelbase: float = 0.26
    front: float = 0.32  # rear axle to nose
    rear: float = 0.10
    half_width: float = 0.12
    max_steer: float = 0.45  # wheel angle radians
    margin: float = 0.05

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.wheelbase, self.front, self.rear, self.half_width, self.max_steer)):
            raise ValueError('invalid vehicle geometry')
        if not 0 <= self.margin < 1 or self.max_steer >= math.pi / 2:
            raise ValueError('invalid vehicle margin/steering')


def advance(pose, distance, steering, geometry):
    if not all(math.isfinite(v) for v in (pose.x, pose.y, pose.yaw, distance, steering)):
        raise ValueError('non-finite bicycle input')
    if abs(steering) > geometry.max_steer + 1e-9:
        raise ValueError('steering outside mechanical limit')
    curvature = math.tan(steering) / geometry.wheelbase
    if abs(curvature) < 1e-9:
        return Pose2(pose.x + distance * math.cos(pose.yaw),
                     pose.y + distance * math.sin(pose.yaw), pose.yaw)
    yaw = pose.yaw + distance * curvature
    return Pose2(pose.x + (math.sin(yaw) - math.sin(pose.yaw)) / curvature,
                 pose.y - (math.cos(yaw) - math.cos(pose.yaw)) / curvature, yaw)


def footprint_clear(grid, pose, geometry):
    """Check every grid cell intersecting the oriented padded rectangle.

    Half-cell diagonal expansion avoids missing occupied cells at edges. Unknown
    cells and out-of-map portions fail closed, including under the current car.
    """
    g = geometry
    radius = math.hypot(max(g.front, g.rear) + g.margin, g.half_width + g.margin)
    r = radius + grid.resolution
    x0 = math.floor((pose.x - r - grid.origin.x) / grid.resolution)
    x1 = math.ceil((pose.x + r - grid.origin.x) / grid.resolution)
    y0 = math.floor((pose.y - r - grid.origin.y) / grid.resolution)
    y1 = math.ceil((pose.y + r - grid.origin.y) / grid.resolution)
    pad = grid.resolution / math.sqrt(2)
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            p = pose.local(grid.center((x, y)))
            if -g.rear - g.margin - pad <= p.x <= g.front + g.margin + pad and abs(p.y) <= g.half_width + g.margin + pad:
                if not (0 <= x < grid.width and 0 <= y < grid.height):
                    return False
                if grid.data[y * grid.width + x] != 0:
                    return False
    return True


def swept_clear(grid, pose, distance, steering, geometry):
    yaw_change = abs(distance * math.tan(steering) / geometry.wheelbase)
    radius = math.hypot(max(geometry.front, geometry.rear) + geometry.margin,
                        geometry.half_width + geometry.margin)
    steps = max(1, math.ceil((abs(distance) + radius * yaw_change) / (grid.resolution / 3)))
    return all(footprint_clear(grid, advance(pose, distance * i / steps, steering, geometry), geometry)
               for i in range(steps + 1))


def lattice_plan(grid, goal, geometry, max_expansions=1500):
    """Forward-only car-feasible primitives; returns poses or an empty stop route."""
    if not all(math.isfinite(v) for v in (goal.x, goal.y)) or goal.x < 0:
        return []
    start = Pose2()
    if not footprint_clear(grid, start, geometry):
        return []
    def key(p):
        return (round(p.x / grid.resolution), round(p.y / grid.resolution),
                round((p.yaw % (2 * math.pi)) / (math.pi / 18)) % 36)
    def heuristic(p):
        return math.hypot(goal.x - p.x, goal.y - p.y)
    if heuristic(start) <= 0.15:
        return []
    step = max(0.15, grid.resolution * 2)
    queue, counter = [(heuristic(start), 0, start, 0.0, [start])], 0
    costs = {key(start): 0}
    for _ in range(max_expansions):
        if not queue:
            break
        _, _, pose, cost, route = heapq.heappop(queue)
        if cost > costs.get(key(pose), math.inf) + 1e-9:
            continue
        if heuristic(pose) <= 0.15:
            return route
        for steering in (-geometry.max_steer, -geometry.max_steer / 2, 0,
                         geometry.max_steer / 2, geometry.max_steer):
            nxt = advance(pose, step, steering, geometry)
            new_cost = cost + step * (1 + 0.1 * abs(steering))
            k = key(nxt)
            if new_cost >= costs.get(k, math.inf):
                continue
            if not swept_clear(grid, pose, step, steering, geometry):
                continue
            costs[k] = new_cost
            counter += 1
            heapq.heappush(queue, (new_cost + heuristic(nxt), counter, nxt, new_cost, route + [nxt]))
    return []


def pursuit(route, geometry, lookahead=0.5, throttle=0.15):
    if len(route) < 2:
        return 0.0, 1.0, 0.0
    target = next((p for p in route[1:] if math.hypot(p.x, p.y) >= lookahead), route[-1])
    distance2 = target.x * target.x + target.y * target.y
    if target.x <= 0 or distance2 < 0.02:
        return 0.0, 1.0, 0.0
    angle = math.atan2(2 * geometry.wheelbase * target.y, distance2)
    return min(0.25, max(0, throttle)), 0.0, min(1.0, max(-1.0, angle / geometry.max_steer))


def command_sweep_clear(grid, speed, steering_normalized, geometry):
    if not math.isfinite(speed) or not math.isfinite(steering_normalized) or abs(steering_normalized) > 1:
        return False
    distance = 0.25 + abs(speed) * 0.3 + speed * speed / (2 * 0.8)
    return swept_clear(grid, Pose2(), distance, steering_normalized * geometry.max_steer, geometry)
