"""Road-edge detection (curb / road boundary).

:class:`RoadEdgeDetector` takes a drivable-space mask (from
drivable_space.py or any model), or computes a crude one from color
similarity to the bottom-center patch, extracts per-row left/right
boundary points of the region connected to the ego, and fits each side
with a RANSAC quadratic ``u = a v^2 + b v + c``. With a BEV image and a
pixel scale the fit is also reported in metric base_link coordinates as
``y_left = c0 + c1 x + c2 x^2`` (same convention as lanes.py).

Limits: assumes one connected road region; no curb height cue; no
handling of junctions where the edge disappears.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from cheesestack.core import Header

from .detection import reconstruct, to_float_image


def ransac_polyfit(x: np.ndarray, y: np.ndarray, deg: int = 2, iters: int = 60, tol: float = 2.0,
                   min_inliers: int = 10, rng: Optional[np.random.Generator] = None
                   ) -> tuple[Optional[np.ndarray], np.ndarray]:
    """Robust polynomial fit y = poly(x). Returns (coeffs highest-first or None, inlier mask)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    if n < max(deg + 1, min_inliers):
        return None, np.zeros(n, bool)
    rng = rng or np.random.default_rng(0)
    best = np.zeros(n, bool)
    for _ in range(iters):
        idx = rng.choice(n, deg + 1, replace=False)
        if np.unique(x[idx]).size < deg + 1:
            continue
        c = np.polyfit(x[idx], y[idx], deg)
        inl = np.abs(np.polyval(c, x) - y) < tol
        if inl.sum() > best.sum():
            best = inl
    if best.sum() < min_inliers:
        return None, best
    c = np.polyfit(x[best], y[best], deg)
    inl = np.abs(np.polyval(c, x) - y) < tol
    return np.polyfit(x[inl], y[inl], deg), inl


@dataclass
class RoadEdge:
    side: str
    points_uv: np.ndarray  # (N, 2) inlier boundary pixels
    coeffs_px: np.ndarray  # u = poly(v)
    coeffs_m: Optional[np.ndarray] = None  # (c0, c1, c2) y_left(x) if metric scale known
    inlier_ratio: float = 0.0


@dataclass
class RoadEdges:
    header: Header
    left: Optional[RoadEdge] = None
    right: Optional[RoadEdge] = None
    road_width_m: float = float("nan")


@dataclass
class RoadEdgeConfig:
    color_threshold: float = 0.12  # used only when no mask is given
    min_row_run_px: int = 3
    ransac_iters: int = 60
    ransac_tol_px: float = 2.0
    min_inliers: int = 10
    deg: int = 2
    edge_margin_px: int = 1  # ignore boundaries touching the image border (not a real edge)
    # optional BEV metric scale (see lanes.py)
    m_per_px_x: Optional[float] = None
    m_per_px_y: Optional[float] = None
    x_offset_m: float = 0.0
    ego_u: Optional[float] = None


class RoadEdgeDetector:
    def __init__(self, config: Optional[RoadEdgeConfig] = None, seed: int = 0) -> None:
        self.config = config or RoadEdgeConfig()
        self._rng = np.random.default_rng(seed)

    def crude_mask(self, image: np.ndarray) -> np.ndarray:
        img = to_float_image(image)
        if img.ndim == 2:
            img = img[..., None]
        h, w = img.shape[:2]
        patch = img[int(h * 0.85):, int(w * 0.4):int(w * 0.6)].reshape(-1, img.shape[-1])
        ref = np.median(patch, axis=0)
        cand = np.sqrt(((img - ref) ** 2).mean(-1)) < self.config.color_threshold
        seed = np.zeros_like(cand)
        seed[int(h * 0.85):, int(w * 0.4):int(w * 0.6)] = True
        return reconstruct(seed & cand, cand)

    def boundary_points(self, mask: np.ndarray, ego_u: float) -> tuple[np.ndarray, np.ndarray]:
        """Per row, left/right edge of the drivable run containing (or nearest) ego_u."""
        cfg = self.config
        h, w = mask.shape
        left, right = [], []
        cu = int(round(ego_u))
        for v in range(h):
            row = mask[v]
            if row.sum() < cfg.min_row_run_px:
                continue
            if not row[cu]:
                cols = np.nonzero(row)[0]
                c = cols[np.argmin(np.abs(cols - cu))]
            else:
                c = cu
            blocked_l = np.nonzero(~row[:c])[0]
            blocked_r = np.nonzero(~row[c:])[0]
            ul = blocked_l[-1] + 1 if blocked_l.size else 0
            ur = c + blocked_r[0] - 1 if blocked_r.size else w - 1
            if ur - ul + 1 < cfg.min_row_run_px:
                continue
            if ul >= cfg.edge_margin_px:
                left.append((ul, v))
            if ur <= w - 1 - cfg.edge_margin_px:
                right.append((ur, v))
        return np.array(left, float).reshape(-1, 2), np.array(right, float).reshape(-1, 2)

    def _metric(self, pts_uv: np.ndarray, h: int, w: int) -> Optional[np.ndarray]:
        cfg = self.config
        if cfg.m_per_px_x is None or cfg.m_per_px_y is None or len(pts_uv) < cfg.deg + 1:
            return None
        ego_u = cfg.ego_u if cfg.ego_u is not None else (w - 1) / 2.0
        x = (h - 1 - pts_uv[:, 1]) * cfg.m_per_px_x + cfg.x_offset_m
        y = (ego_u - pts_uv[:, 0]) * cfg.m_per_px_y
        return np.polyfit(x, y, cfg.deg)[::-1]

    def detect(self, header: Header, mask: Optional[np.ndarray] = None,
               image: Optional[np.ndarray] = None) -> RoadEdges:
        cfg = self.config
        if mask is None:
            if image is None:
                raise ValueError("need a drivable mask or an image")
            mask = self.crude_mask(image)
        mask = np.asarray(mask, bool)
        h, w = mask.shape
        ego_u = cfg.ego_u if cfg.ego_u is not None else (w - 1) / 2.0
        lp, rp = self.boundary_points(mask, ego_u)
        out = RoadEdges(header=header)
        for side, pts in (("left", lp), ("right", rp)):
            if len(pts) == 0:
                continue
            c, inl = ransac_polyfit(pts[:, 1], pts[:, 0], cfg.deg, cfg.ransac_iters, cfg.ransac_tol_px,
                                    cfg.min_inliers, self._rng)
            if c is None:
                continue
            edge = RoadEdge(side, pts[inl], c, self._metric(pts[inl], h, w), float(inl.mean()))
            setattr(out, side, edge)
        if out.left is not None and out.right is not None and out.left.coeffs_m is not None \
                and out.right.coeffs_m is not None:
            out.road_width_m = float(out.left.coeffs_m[0] - out.right.coeffs_m[0])
        return out
        # TODO(perception): curb detection from depth/stereo or learned edge head,
        # TODO(perception): temporal smoothing and edge type (curb, grass, barrier).
