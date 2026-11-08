"""Lens distortion (Brown-Conrady) and image remapping.

Model (OpenCV ordering k1, k2, p1, p2, k3) on normalized coords (x, y) = K^-1 [u v 1]:
    r2 = x^2 + y^2
    radial = 1 + k1 r2 + k2 r2^2 + k3 r2^3
    xd = x*radial + 2 p1 x y + p2 (r2 + 2 x^2)
    yd = y*radial + p1 (r2 + 2 y^2) + 2 p2 x y

Undistortion of points inverts this with Newton iterations (analytic 2x2 Jacobian),
falling back to fixed-point when the Jacobian is near singular. Image undistortion
builds a (map_u, map_v) lookup once (`UndistortMap`) and samples with
`remap_bilinear`.

Limits: valid only inside the radius where the polynomial is monotonic; strong
fisheye lenses need the Kannala-Brandt model.
# TODO(perception): fisheye (Kannala-Brandt) model, optimal new camera matrix
#   (alpha / valid-ROI), fixed-point int16 maps for speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BrownConrady:
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    @classmethod
    def from_array(cls, d) -> "BrownConrady":
        d = list(np.asarray(d, dtype=float).ravel()) + [0.0] * 5
        return cls(*d[:5])

    def as_array(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])

    @property
    def is_identity(self) -> bool:
        return not np.any(self.as_array())

    def distort_normalized(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        x, y = xy[..., 0], xy[..., 1]
        r2 = x * x + y * y
        rad = 1 + self.k1 * r2 + self.k2 * r2 ** 2 + self.k3 * r2 ** 3
        xd = x * rad + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
        yd = y * rad + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
        return np.stack([xd, yd], axis=-1)

    def _jacobian(self, x: np.ndarray, y: np.ndarray):
        k1, k2, k3, p1, p2 = self.k1, self.k2, self.k3, self.p1, self.p2
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        drad = k1 + 2 * k2 * r2 + 3 * k3 * r2 ** 2  # d rad / d r2
        dxdx = rad + x * drad * 2 * x + 2 * p1 * y + p2 * 6 * x
        dxdy = x * drad * 2 * y + 2 * p1 * x + p2 * 2 * y
        dydx = y * drad * 2 * x + p1 * 2 * x + 2 * p2 * y
        dydy = rad + y * drad * 2 * y + p1 * 6 * y + 2 * p2 * x
        return dxdx, dxdy, dydx, dydy

    def undistort_normalized(self, xyd: np.ndarray, iters: int = 20, tol: float = 1e-12) -> np.ndarray:
        xyd = np.asarray(xyd, dtype=float)
        if self.is_identity:
            return xyd.copy()
        x, y = xyd[..., 0].copy(), xyd[..., 1].copy()
        for _ in range(iters):
            f = self.distort_normalized(np.stack([x, y], axis=-1))
            ex, ey = f[..., 0] - xyd[..., 0], f[..., 1] - xyd[..., 1]
            a, b, c, d = self._jacobian(x, y)
            det = a * d - b * c
            ok = np.abs(det) > 1e-9
            sdet = np.where(ok, det, 1.0)
            dx = np.where(ok, (d * ex - b * ey) / sdet, ex)
            dy = np.where(ok, (-c * ex + a * ey) / sdet, ey)
            x, y = x - dx, y - dy
            if np.max(np.abs(dx), initial=0) < tol and np.max(np.abs(dy), initial=0) < tol:
                break
        return np.stack([x, y], axis=-1)


def _to_norm(uv: np.ndarray, K: np.ndarray) -> np.ndarray:
    uv = np.asarray(uv, dtype=float)
    fx, fy, cx, cy, s = K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[0, 1]
    y = (uv[..., 1] - cy) / fy
    x = (uv[..., 0] - cx - s * y) / fx
    return np.stack([x, y], axis=-1)


def _to_pix(xy: np.ndarray, K: np.ndarray) -> np.ndarray:
    u = K[0, 0] * xy[..., 0] + K[0, 1] * xy[..., 1] + K[0, 2]
    v = K[1, 1] * xy[..., 1] + K[1, 2]
    return np.stack([u, v], axis=-1)


def distort_points(uv: np.ndarray, K: np.ndarray, dist: BrownConrady) -> np.ndarray:
    """Ideal (undistorted) pixels -> distorted pixels."""
    return _to_pix(dist.distort_normalized(_to_norm(uv, K)), K)


def undistort_points(uv: np.ndarray, K: np.ndarray, dist: BrownConrady, K_new: Optional[np.ndarray] = None) -> np.ndarray:
    """Distorted pixels -> undistorted pixels (in K_new, default K)."""
    xy = dist.undistort_normalized(_to_norm(uv, K))
    return _to_pix(xy, K if K_new is None else K_new)


def remap_bilinear(img: np.ndarray, map_u: np.ndarray, map_v: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """out[i, j] = img sampled at (map_u[i,j], map_v[i,j]) with bilinear interpolation.
    Samples outside [0, W-1] x [0, H-1] get `fill`. Works for HxW and HxWxC."""
    src = np.asarray(img)
    f = src.astype(np.float32) / 255.0 if src.dtype == np.uint8 else src.astype(np.float32, copy=False)
    H, W = f.shape[:2]
    u, v = np.asarray(map_u, dtype=np.float64), np.asarray(map_v, dtype=np.float64)
    valid = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1) & np.isfinite(u) & np.isfinite(v)
    uc, vc = np.clip(np.nan_to_num(u), 0, W - 1), np.clip(np.nan_to_num(v), 0, H - 1)
    u0, v0 = np.floor(uc).astype(int), np.floor(vc).astype(int)
    u1, v1 = np.minimum(u0 + 1, W - 1), np.minimum(v0 + 1, H - 1)
    wu, wv = (uc - u0).astype(np.float32), (vc - v0).astype(np.float32)
    if f.ndim == 3:
        wu, wv, valid_b = wu[..., None], wv[..., None], valid[..., None]
    else:
        valid_b = valid
    out = (f[v0, u0] * (1 - wu) * (1 - wv) + f[v0, u1] * wu * (1 - wv)
           + f[v1, u0] * (1 - wu) * wv + f[v1, u1] * wu * wv)
    return np.where(valid_b, out, np.float32(fill)).astype(np.float32)


@dataclass
class UndistortConfig:
    fill: float = 0.0
    # TODO(perception): `alpha` to choose K_new trading black borders vs. lost pixels.


class UndistortMap:
    """Precomputed lookup that turns a distorted image into an undistorted one.

    For each output (ideal) pixel in K_new we compute where it lands in the
    distorted source image (forward distortion — no iteration needed)."""

    def __init__(self, K: np.ndarray, dist: BrownConrady, size_hw: tuple[int, int],
                 K_new: Optional[np.ndarray] = None, config: Optional[UndistortConfig] = None) -> None:
        self.K, self.dist = np.asarray(K, float), dist
        self.K_new = self.K if K_new is None else np.asarray(K_new, float)
        self.config = config or UndistortConfig()
        H, W = size_hw
        vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
        xy = _to_norm(np.stack([uu, vv], axis=-1), self.K_new)
        src = _to_pix(dist.distort_normalized(xy), self.K)
        self.map_u, self.map_v = src[..., 0], src[..., 1]

    def apply(self, img: np.ndarray) -> np.ndarray:
        return remap_bilinear(img, self.map_u, self.map_v, self.config.fill)


def distort_image(img: np.ndarray, K: np.ndarray, dist: BrownConrady) -> np.ndarray:
    """Synthesize a distorted image from an ideal one (inverse map; for tests/sim)."""
    H, W = img.shape[:2]
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
    src = undistort_points(np.stack([uu, vv], axis=-1), K, dist)
    return remap_bilinear(img, src[..., 0], src[..., 1])
