"""Image preprocessing in pure numpy.

Functions operate on HxW or HxWxC arrays (uint8 or float). `to_float` maps to
float32 [0,1]; everything else returns float32 unless noted.

- `to_float`, `to_uint8`, `grayscale` (BT.601 luma)
- `resize` (nearest or bilinear, align-corners=False / pixel-center convention)
- `normalize` (per-channel mean/std), `crop_roi`
- `gaussian_kernel1d`, `gaussian_blur` (separable, reflect padding)
- `equalize_hist` (global), `clahe_lite` (tiled equalization, no bilinear blend)
- `Preprocessor`: configurable chain used ahead of detectors.

# TODO(perception): proper CLAHE with tile interpolation, area-averaging downscale
#   (anti-aliasing), GPU/Accelerate backends, letterboxing for detector input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


def to_float(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32, copy=False)


def to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def grayscale(img: np.ndarray) -> np.ndarray:
    """RGB -> luma (BT.601). Gray input is returned as float."""
    f = to_float(img)
    if f.ndim == 2:
        return f
    if f.shape[2] == 1:
        return f[..., 0]
    return (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]).astype(np.float32)


def resize(img: np.ndarray, out_hw: tuple[int, int], method: str = "bilinear") -> np.ndarray:
    """Resize to (H, W). Pixel-center convention: src = (dst + 0.5) * scale - 0.5."""
    f = to_float(img)
    H, W = f.shape[:2]
    oh, ow = int(out_hw[0]), int(out_hw[1])
    sy, sx = H / oh, W / ow
    ys = (np.arange(oh) + 0.5) * sy - 0.5
    xs = (np.arange(ow) + 0.5) * sx - 0.5
    if method == "nearest":
        yi = np.clip(np.floor(ys + 0.5), 0, H - 1).astype(int)
        xi = np.clip(np.floor(xs + 0.5), 0, W - 1).astype(int)
        return f[yi][:, xi]
    if method != "bilinear":
        raise ValueError(f"unknown method {method}")
    ys, xs = np.clip(ys, 0, H - 1), np.clip(xs, 0, W - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, H - 1), np.minimum(x0 + 1, W - 1)
    wy, wx = (ys - y0).astype(np.float32), (xs - x0).astype(np.float32)
    if f.ndim == 3:
        wy, wx = wy[:, None, None], wx[None, :, None]
    else:
        wy, wx = wy[:, None], wx[None, :]
    top = f[y0][:, x0] * (1 - wx) + f[y0][:, x1] * wx
    bot = f[y1][:, x0] * (1 - wx) + f[y1][:, x1] * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


def normalize(img: np.ndarray, mean: Sequence[float] | float = 0.5, std: Sequence[float] | float = 0.5) -> np.ndarray:
    """(img - mean) / std, broadcast per channel. Input mapped to [0,1] first."""
    f = to_float(img)
    m = np.asarray(mean, dtype=np.float32)
    s = np.asarray(std, dtype=np.float32)
    return ((f - m) / s).astype(np.float32)


def crop_roi(img: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Crop (u, v, w, h) clamped to image bounds. Returns a view."""
    u, v, w, h = (int(round(x)) for x in roi)
    H, W = img.shape[:2]
    u0, v0 = max(0, u), max(0, v)
    u1, v1 = min(W, u + w), min(H, v + h)
    if u1 <= u0 or v1 <= v0:
        return img[0:0, 0:0]
    return img[v0:v1, u0:u1]


def gaussian_kernel1d(sigma: float, radius: Optional[int] = None) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    r = int(np.ceil(3.0 * sigma)) if radius is None else int(radius)
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(np.float32)


def convolve1d(img: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """1D correlation along `axis` with reflect padding (works for 2D/3D)."""
    r = len(kernel) // 2
    pad = [(0, 0)] * img.ndim
    pad[axis] = (r, r)
    p = np.pad(img, pad, mode="reflect") if img.shape[axis] > r else np.pad(img, pad, mode="edge")
    out = np.zeros_like(img, dtype=np.float32)
    n = img.shape[axis]
    for i, kv in enumerate(kernel):
        out += kv * np.take(p, np.arange(i, i + n), axis=axis)
    return out


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    f = to_float(img)
    if sigma <= 0:
        return f.copy()
    k = gaussian_kernel1d(sigma)
    return convolve1d(convolve1d(f, k, 0), k, 1)


def equalize_hist(img: np.ndarray, bins: int = 256) -> np.ndarray:
    """Global histogram equalization of a grayscale image -> float32 [0,1].
    Color input is equalized on luma and channels rescaled by the luma gain."""
    f = to_float(img)
    if f.ndim == 3:
        y = grayscale(f)
        ye = equalize_hist(y, bins)
        gain = ye / np.maximum(y, 1e-6)
        return np.clip(f * gain[..., None], 0, 1).astype(np.float32)
    q = np.clip((f * (bins - 1) + 0.5).astype(int), 0, bins - 1)
    hist = np.bincount(q.ravel(), minlength=bins)
    cdf = np.cumsum(hist).astype(np.float64)
    nz = cdf[cdf > 0]
    cmin = nz[0] if nz.size else 0.0
    denom = cdf[-1] - cmin
    if denom <= 0:
        return f.copy()
    lut = np.clip((cdf - cmin) / denom, 0, 1).astype(np.float32)
    return lut[q]


def clahe_lite(img: np.ndarray, tiles: tuple[int, int] = (4, 4), clip_limit: float = 2.0, bins: int = 64) -> np.ndarray:
    """Per-tile clipped histogram equalization (no inter-tile interpolation -> visible seams).
    # TODO(perception): bilinear blending of tile LUTs (real CLAHE)."""
    g = grayscale(img)
    H, W = g.shape
    out = np.empty_like(g)
    ty, tx = tiles
    ye = np.linspace(0, H, ty + 1).astype(int)
    xe = np.linspace(0, W, tx + 1).astype(int)
    for i in range(ty):
        for j in range(tx):
            t = g[ye[i]:ye[i + 1], xe[j]:xe[j + 1]]
            if t.size == 0:
                continue
            q = np.clip((t * (bins - 1) + 0.5).astype(int), 0, bins - 1)
            hist = np.bincount(q.ravel(), minlength=bins).astype(np.float64)
            limit = clip_limit * t.size / bins
            excess = np.maximum(hist - limit, 0).sum()
            hist = np.minimum(hist, limit) + excess / bins
            cdf = np.cumsum(hist)
            lut = (cdf / cdf[-1]).astype(np.float32)
            out[ye[i]:ye[i + 1], xe[j]:xe[j + 1]] = lut[q]
    return out


@dataclass
class PreprocessConfig:
    out_hw: Optional[tuple[int, int]] = None  # resize target (H, W)
    roi: Optional[tuple[int, int, int, int]] = None  # (u, v, w, h) applied before resize
    gray: bool = False
    blur_sigma: float = 0.0
    equalize: bool = False
    normalize: bool = False
    mean: tuple = (0.5,)
    std: tuple = (0.5,)
    resize_method: str = "bilinear"


class Preprocessor:
    """ROI -> resize -> gray -> equalize -> blur -> normalize. Records the geometric
    transform so downstream can map pixels back to the raw image (`to_raw`)."""

    def __init__(self, config: Optional[PreprocessConfig] = None) -> None:
        self.config = config or PreprocessConfig()
        self.scale = (1.0, 1.0)  # (sx, sy) raw px per output px
        self.offset = (0.0, 0.0)

    def process(self, img: np.ndarray) -> np.ndarray:
        c = self.config
        f = to_float(img)
        self.offset, self.scale = (0.0, 0.0), (1.0, 1.0)
        if c.roi is not None:
            u, v = max(0, int(round(c.roi[0]))), max(0, int(round(c.roi[1])))
            f = crop_roi(f, c.roi)
            self.offset = (float(u), float(v))
        if c.out_hw is not None and f.size:
            H, W = f.shape[:2]
            f = resize(f, c.out_hw, c.resize_method)
            self.scale = (W / c.out_hw[1], H / c.out_hw[0])
        if c.gray:
            f = grayscale(f)
        if c.equalize:
            f = equalize_hist(f)
        if c.blur_sigma > 0:
            f = gaussian_blur(f, c.blur_sigma)
        if c.normalize:
            f = normalize(f, c.mean, c.std)
        return f

    def to_raw(self, uv: np.ndarray) -> np.ndarray:
        """Map output-image pixel coords (N,2) back to raw-image coords (pixel-center convention)."""
        uv = np.asarray(uv, dtype=float).reshape(-1, 2)
        sx, sy = self.scale
        return np.stack([(uv[:, 0] + 0.5) * sx - 0.5 + self.offset[0],
                         (uv[:, 1] + 0.5) * sy - 0.5 + self.offset[1]], axis=1)
