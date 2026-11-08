"""Bounded odom-frame occupancy history for visualization, never safety free-space."""
import math
from .core import LocalMap, Point


class EvidenceMap:
    def __init__(self, resolution=0.1, ttl=5.0, max_cells=40000):
        self.resolution, self.ttl, self.max_cells = resolution, ttl, max_cells
        self.cells, self.last_stamp = {}, None

    def integrate(self, local, pose, stamp):
        if not math.isfinite(stamp):
            raise ValueError('invalid mapping timestamp')
        if self.last_stamp is not None and stamp < self.last_stamp:
            self.cells.clear()
        self.last_stamp = stamp
        updates = {}
        for i, value in enumerate(local.data):
            if value == -1:
                continue
            p = pose.world(local.center((i % local.width, i // local.width)))
            key = (math.floor(p.x / self.resolution), math.floor(p.y / self.resolution))
            updates[key] = max(updates.get(key, 0), value)
        for key, value in updates.items():
            self.cells[key] = (value, stamp)
        self.cells = {k: v for k, v in self.cells.items() if 0 <= stamp - v[1] <= self.ttl}
        if len(self.cells) > self.max_cells:
            self.cells = dict(sorted(self.cells.items(), key=lambda item: item[1][1])[-self.max_cells:])

    def snapshot(self, pose, now, width=100, height=100):
        grid = LocalMap(self.resolution, width, height)
        grid.origin = Point(math.floor(pose.x / self.resolution) * self.resolution - width*self.resolution/2,
                            math.floor(pose.y / self.resolution) * self.resolution - height*self.resolution/2)
        for (x, y), (value, stamp) in self.cells.items():
            if not 0 <= now - stamp <= self.ttl:
                continue
            c = grid.cell(Point((x + 0.5) * self.resolution, (y + 0.5) * self.resolution))
            if c:
                grid.data[c[1] * grid.width + c[0]] = value
        return grid
