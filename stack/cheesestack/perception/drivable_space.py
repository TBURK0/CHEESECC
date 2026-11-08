"""Drivable-space segmentation: model interface + color/texture region-growing baseline.

:class:`RegionGrowingSegmenter` assumes the patch just in front of the ego
(bottom-center of the image) is road. It builds a Gaussian model of that
patch over features (R, G, B, local texture), marks pixels whose
Mahalanobis distance is below a threshold, and keeps only the region
connected to the seed (morphological reconstruction = flood fill).

The free-space boundary is reported per image column as the first row,
scanning up from the bottom, that is not drivable (the "stixel" base).
`boundary_to_ground` projects those pixels to metric ground points with an
image->ground homography supplied by the caller (calibration lives
elsewhere).

Limits: fails if the seed patch is not road (car directly ahead, lane
paint), on strong shadows, and with similarly-colored sidewalks.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cheesestack.core import Header

from .detection import box_filter, reconstruct, sobel_xy, to_float_image, to_gray


@dataclass
class DrivableSpace:
    header: Header
    mask: np.ndarray  # HxW bool
    probability: np.ndarray  # HxW float [0,1]
    boundary_v: np.ndarray  # (W,) int, see free_space_boundary()
    confidence: float = 0.0
    debug: dict = field(default_factory=dict)

    def free_fraction(self) -> float:
        return float(self.mask.mean())


class DrivableSpaceModel(abc.ABC):
    @abc.abstractmethod
    def segment(self, image: np.ndarray, header: Header) -> DrivableSpace:
        ...


def free_space_boundary(mask: np.ndarray) -> np.ndarray:
    """Per column, the row index just above the bottom-connected free run.

    Returns (W,) int array b where rows b+1..H-1 are drivable and row b is not.
    b = H-1 means the bottom pixel itself is blocked; b = -1 means the whole
    column is free.
    """
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    blocked = ~mask[::-1, :]  # row 0 = bottom
    any_blocked = blocked.any(axis=0)
    first = np.argmax(blocked, axis=0)  # rows from bottom
    b = h - 1 - first
    return np.where(any_blocked, b, -1).astype(int)


def boundary_to_ground(boundary_v: np.ndarray, H_img_to_ground: np.ndarray, image_height: int,
                       columns: Optional[np.ndarray] = None) -> np.ndarray:
    """Project boundary pixels (u, b+1) through a 3x3 homography to (N, 2) ground points.

    Columns whose boundary is -1 (fully free) project the top row instead.
    """
    b = np.asarray(boundary_v)
    u = np.arange(b.size, dtype=float) if columns is None else np.asarray(columns, dtype=float)
    v = np.clip(b + 1, 0, image_height - 1).astype(float)
    pts = np.stack([u, v, np.ones_like(u)])
    g = np.asarray(H_img_to_ground, dtype=float) @ pts
    return (g[:2] / g[2]).T


@dataclass
class RegionGrowingConfig:
    seed_rows_frac: float = 0.12  # seed patch height as a fraction of H (bottom of image)
    seed_cols_frac: float = 0.3  # seed patch width fraction, centered
    seed_center_u: Optional[float] = None
    texture_window: int = 5
    texture_weight: float = 1.0
    mahal_threshold: float = 3.5
    reg: float = 1e-3  # covariance regularization
    horizon_v: int = 0  # rows above this are never drivable
    connectivity: int = 4
    open_iters: int = 1  # morphological opening to cut thin bridges


class RegionGrowingSegmenter(DrivableSpaceModel):
    """Seeded Gaussian color/texture classifier + connected region growing."""

    def __init__(self, config: Optional[RegionGrowingConfig] = None) -> None:
        self.config = config or RegionGrowingConfig()

    def features(self, image: np.ndarray) -> np.ndarray:
        img = to_float_image(image)
        if img.ndim == 2:
            img = img[..., None]
        gray = to_gray(img if img.shape[-1] == 3 else img[..., 0])
        gx, gy = sobel_xy(gray)
        tex = box_filter(np.hypot(gx, gy), self.config.texture_window) * self.config.texture_weight
        return np.concatenate([img, tex[..., None]], axis=-1)

    def seed_mask(self, shape: tuple[int, int]) -> np.ndarray:
        cfg = self.config
        h, w = shape
        sh = max(1, int(round(h * cfg.seed_rows_frac)))
        sw = max(1, int(round(w * cfg.seed_cols_frac)))
        cu = cfg.seed_center_u if cfg.seed_center_u is not None else w / 2.0
        u0 = int(np.clip(round(cu - sw / 2), 0, w - 1))
        m = np.zeros((h, w), bool)
        m[h - sh:, u0:min(w, u0 + sw)] = True
        return m

    def segment(self, image: np.ndarray, header: Header) -> DrivableSpace:
        cfg = self.config
        f = self.features(image)
        h, w, c = f.shape
        seed = self.seed_mask((h, w))
        X = f[seed]
        mu = X.mean(axis=0)
        cov = np.cov(X.T) + cfg.reg * np.eye(c)
        icov = np.linalg.inv(cov)
        d = (f.reshape(-1, c) - mu)
        m2 = np.einsum("ij,jk,ik->i", d, icov, d).reshape(h, w)
        maha = np.sqrt(np.maximum(m2, 0))
        cand = maha < cfg.mahal_threshold
        cand[:cfg.horizon_v, :] = False
        for _ in range(cfg.open_iters):  # erosion then dilation (4-neighborhood)
            p = np.pad(cand, 1)
            er = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
            p = np.pad(er, 1)
            cand = (p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]) & cand
        mask = reconstruct(seed & cand, cand, cfg.connectivity)
        prob = np.exp(-0.5 * (maha / cfg.mahal_threshold) ** 2 * 4.0) * mask
        seed_ok = float((seed & cand).sum()) / max(1, seed.sum())
        return DrivableSpace(header=header, mask=mask, probability=prob.astype(np.float32),
                             boundary_v=free_space_boundary(mask), confidence=seed_ok,
                             debug={"mahalanobis": maha})
        # TODO(perception): learned segmentation network behind DrivableSpaceModel,
        # TODO(perception): adaptive seed (reject seed if occupied, use lane prior), temporal fusion in BEV.
