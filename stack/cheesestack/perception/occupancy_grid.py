"""Ego-centred log-odds occupancy grid.

The grid is a square of `size_m` x `size_m` cells at `resolution` m,
centred on the ego (base_link), x forward = increasing row index, y left =
increasing column index. Cell (i, j) centre is
``x = (i - n/2 + 0.5) * res``, ``y = (j - n/2 + 0.5) * res``.

Operations:
- :meth:`insert_scan` - ray casting from a sensor origin: cells along each
  ray get `l_free`, the end cell gets `l_occ` if the return is a hit
  (range < max_range).
- :meth:`insert_points` / :meth:`insert_detections` - mark obstacle cells
  (e.g. stereo points, detection footprints from core.Detection.box3d).
- :meth:`decay` - log-odds relax toward the prior with time constant
  `decay_tau_s` (forgets moving objects).
- :meth:`shift` - ego-motion compensation: resample the grid into the
  new base_link frame given the ego motion as a Pose2D.

Limits: 2D, single-height, nearest-neighbour resampling (blurs over time
with rotation), no dynamic-cell classification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from cheesestack.core import Detection, Header, Pose2D


def prob_to_logodds(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def logodds_to_prob(l):
    return 1.0 / (1.0 + np.exp(-l))


@dataclass
class OccupancyGridConfig:
    size_m: float = 40.0
    resolution: float = 0.2
    p_hit: float = 0.7
    p_miss: float = 0.4
    l_min: float = -4.0
    l_max: float = 4.0
    decay_tau_s: float = 2.0
    max_range: float = 30.0
    occupied_threshold: float = 0.65
    free_threshold: float = 0.35


class OccupancyGrid:
    def __init__(self, config: Optional[OccupancyGridConfig] = None) -> None:
        self.config = config or OccupancyGridConfig()
        c = self.config
        self.n = int(round(c.size_m / c.resolution))
        self.logodds = np.zeros((self.n, self.n), dtype=np.float32)
        self.l_occ = float(prob_to_logodds(c.p_hit))
        self.l_free = float(prob_to_logodds(c.p_miss))
        self.stamp: Optional[float] = None

    # -- coordinates ----------------------------------------------------
    def world_to_cell(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(N,2) base_link points -> (i, j, inside mask)."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        res = self.config.resolution
        i = np.floor(pts[:, 0] / res + self.n / 2).astype(int)
        j = np.floor(pts[:, 1] / res + self.n / 2).astype(int)
        inside = (i >= 0) & (i < self.n) & (j >= 0) & (j < self.n)
        return i, j, inside

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        res = self.config.resolution
        c = (np.arange(self.n) - self.n / 2 + 0.5) * res
        return np.meshgrid(c, c, indexing="ij")  # X[i,j], Y[i,j]

    def _add(self, i, j, value: float) -> None:
        # one update per cell per call (dedupe)
        if i.size == 0:
            return
        flat = np.unique(i * self.n + j)
        lo = self.logodds.ravel()
        lo[flat] = np.clip(lo[flat] + value, self.config.l_min, self.config.l_max)

    # -- updates --------------------------------------------------------
    def insert_scan(self, origin: Sequence[float], angles: np.ndarray, ranges: np.ndarray) -> None:
        """Ray-cast a planar range scan. origin = (x, y, yaw) of the sensor in base_link."""
        c = self.config
        ox, oy, oyaw = float(origin[0]), float(origin[1]), float(origin[2]) if len(origin) > 2 else 0.0
        angles = np.asarray(angles, float)
        ranges = np.asarray(ranges, float)
        ok = np.isfinite(ranges) & (ranges > 0)
        angles, ranges = angles[ok], ranges[ok]
        hit = ranges < c.max_range
        r = np.minimum(ranges, c.max_range)
        step = c.resolution * 0.5
        free_pts = []
        for a, rr in zip(angles + oyaw, r):
            s = np.arange(0.0, max(rr - c.resolution, 0.0), step)
            free_pts.append(np.stack([ox + s * math.cos(a), oy + s * math.sin(a)], axis=1))
        end = np.stack([ox + r * np.cos(angles + oyaw), oy + r * np.sin(angles + oyaw)], axis=1)
        ei, ej, ein = self.world_to_cell(end[hit])
        hit_cells = set((ei[ein] * self.n + ej[ein]).tolist())
        if free_pts:
            fp = np.concatenate(free_pts)
            fi, fj, fin = self.world_to_cell(fp)
            fi, fj = fi[fin], fj[fin]
            keep = ~np.isin(fi * self.n + fj, list(hit_cells)) if hit_cells else np.ones(fi.size, bool)
            self._add(fi[keep], fj[keep], self.l_free)
        self._add(ei[ein], ej[ein], self.l_occ)
        # TODO(perception): proper Bresenham / beam model with sensor noise, 3D-to-2D height filtering.

    def insert_points(self, pts: np.ndarray, weight: float = 1.0) -> None:
        i, j, inside = self.world_to_cell(pts)
        self._add(i[inside], j[inside], self.l_occ * weight)

    def insert_detections(self, dets: Sequence[Detection], weight: float = 1.0) -> None:
        """Rasterise each Detection.box3d footprint (in base_link) as occupied, weighted by score."""
        X, Y = self.cell_centers()
        for d in dets:
            b = d.box3d
            if b is None:
                continue
            cy, sy = math.cos(b.pose.yaw), math.sin(b.pose.yaw)
            dx, dy = X - b.pose.x, Y - b.pose.y
            lx = cy * dx + sy * dy
            ly = -sy * dx + cy * dy
            m = (np.abs(lx) <= b.length / 2 + self.config.resolution / 2) & \
                (np.abs(ly) <= b.width / 2 + self.config.resolution / 2)
            i, j = np.nonzero(m)
            self._add(i, j, self.l_occ * weight * float(d.score))

    def decay(self, dt: float) -> None:
        if dt > 0:
            self.logodds *= np.float32(math.exp(-dt / self.config.decay_tau_s))

    def shift(self, motion: Pose2D) -> None:
        """Ego moved by `motion` (new base_link pose expressed in old base_link)."""
        X, Y = self.cell_centers()
        new_pts = np.stack([X.ravel(), Y.ravel()], axis=1)
        old_pts = motion.transform_points(new_pts)
        i, j, inside = self.world_to_cell(old_pts)
        out = np.zeros(self.n * self.n, dtype=np.float32)
        out[inside] = self.logodds[i[inside], j[inside]]
        self.logodds = out.reshape(self.n, self.n)

    def step(self, stamp: float, motion: Optional[Pose2D] = None) -> None:
        """Convenience per-cycle: decay by elapsed time, then shift by ego motion."""
        if self.stamp is not None:
            self.decay(stamp - self.stamp)
        if motion is not None:
            self.shift(motion)
        self.stamp = stamp

    # -- queries --------------------------------------------------------
    def probability(self) -> np.ndarray:
        return logodds_to_prob(self.logodds)

    def occupied_mask(self) -> np.ndarray:
        return self.probability() >= self.config.occupied_threshold

    def free_mask(self) -> np.ndarray:
        return self.probability() <= self.config.free_threshold

    def query(self, x: float, y: float) -> float:
        """Occupancy probability at a base_link point (0.5 outside the grid)."""
        i, j, inside = self.world_to_cell(np.array([[x, y]]))
        return float(logodds_to_prob(self.logodds[i[0], j[0]])) if inside[0] else 0.5

    def header(self) -> Header:
        return Header(stamp=self.stamp or 0.0, frame_id="base_link", source="perception.occupancy_grid")
        # TODO(perception): publish as a core grid message once one exists; multi-resolution / GPU.
