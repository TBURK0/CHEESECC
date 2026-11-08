"""Deterministic supervised-course behavior and geometric path tracking."""
import math
import heapq


class Behavior:
    def __init__(self, dwell_s=2.0):
        self.state = 'WAITING'
        self.stop_until = None
        self.stop_latched = False
        self.dwell_s = dwell_s

    def update(self, now, localized, perception_ok, signal, stop_sign, speed, arrived=False):
        if not math.isfinite(now) or not math.isfinite(speed):
            self.stop_until = None
            self.state = 'SAFE_STOP'
        elif arrived:
            self.state = 'ARRIVED'
        elif not localized or not perception_ok:
            self.stop_until = None
            self.state = 'SAFE_STOP'
        elif signal not in ('GREEN', 'NONE'):
            self.stop_until = None
            self.state = 'SIGNAL_STOP'
        elif stop_sign and not self.stop_latched:
            if abs(speed) > 0.02:
                self.stop_until = None
                self.state = 'STOPPING'
            else:
                if self.stop_until is None:
                    self.stop_until = now + self.dwell_s
                self.state = 'STOP_DWELL'
                if now >= self.stop_until:
                    self.stop_latched = True
                    self.state = 'DRIVING'
        else:
            self.state = 'DRIVING'
        if not stop_sign:
            self.stop_latched, self.stop_until = False, None
        return self.state == 'DRIVING'


def astar(start, goal, occupied=(), size=(20, 20)):
    occupied = set(occupied)
    def valid(p):
        return len(p) == 2 and all(isinstance(v, int) for v in p) and 0 <= p[0] < size[0] and 0 <= p[1] < size[1] and p not in occupied
    if not valid(start) or not valid(goal):
        raise ValueError('goal/start outside approved free course')
    queue, cost, parent = [(0, start)], {start: 0}, {}
    while queue:
        _, p = heapq.heappop(queue)
        if p == goal:
            route = [p]
            while p != start:
                p = parent[p]
                route.append(p)
            return list(reversed(route))
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            q = p[0]+dx, p[1]+dy
            if valid(q) and cost[p]+1 < cost.get(q, float('inf')):
                cost[q], parent[q] = cost[p]+1, p
                heapq.heappush(queue, (cost[q]+abs(q[0]-goal[0])+abs(q[1]-goal[1]), q))
    raise ValueError('unreachable goal')


def pursuit(x, y, yaw, path, wheelbase=0.25, lookahead=0.5):
    if not path or not all(math.isfinite(v) for v in (x, y, yaw)):
        return 0.0, 0.0
    if math.hypot(path[-1][0]-x, path[-1][1]-y) < 0.15:
        return 0.0, 0.0
    nearest = min(range(len(path)), key=lambda i: math.hypot(path[i][0]-x, path[i][1]-y))
    target = next((p for p in path[nearest:] if math.hypot(p[0]-x, p[1]-y) >= lookahead), path[-1])
    distance = max(0.01, math.hypot(target[0]-x, target[1]-y))
    curvature = 2*math.sin(math.atan2(target[1]-y, target[0]-x)-yaw)/distance
    steer = max(-0.5, min(0.5, math.atan(wheelbase*curvature)))
    speed = min(0.3, math.hypot(path[-1][0]-x, path[-1][1]-y))
    return speed, speed*math.tan(steer)/wheelbase
