"""Label-aware, seeded augmentation pipelines for numpy images.

Every transform maps ``(image, annotation, rng) -> (image, annotation)`` and keeps
boxes / polylines / masks consistent with the pixels. Images are HxW or HxWxC,
float in [0,1] or uint8; dtype is preserved. ``Compose`` owns a seeded
``np.random.Generator`` so a pipeline with the same seed is fully reproducible.

Geometry convention: continuous pixel coords, image spans [0, W] x [0, H], so a
horizontal flip maps u -> W - u.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

import numpy as np

from .annotation import Annotation, BoxLabel, MaskLabel, PolylineLabel


class Transform(Protocol):
    def __call__(self, image: np.ndarray, ann: Optional[Annotation],
                 rng: np.random.Generator) -> tuple[np.ndarray, Optional[Annotation]]: ...


def _to_float(img: np.ndarray) -> tuple[np.ndarray, np.dtype]:
    dt = img.dtype
    if dt == np.uint8:
        return img.astype(np.float32) / 255.0, dt
    return img.astype(np.float32, copy=True), dt


def _restore(img: np.ndarray, dt: np.dtype) -> np.ndarray:
    img = np.clip(img, 0.0, 1.0)
    if dt == np.uint8:
        return np.round(img * 255.0).astype(np.uint8)
    return img.astype(dt)


@dataclass
class HorizontalFlip:
    p: float = 0.5
    class_swap: dict = field(default_factory=lambda: {})  # e.g. {"turn_left": "turn_right"}

    def __call__(self, image, ann, rng):
        if rng.random() >= self.p:
            return image, ann
        W = image.shape[1]
        image = image[:, ::-1].copy()
        if ann is None:
            return image, ann
        ann = ann.copy()
        swap = {**self.class_swap, **{v: k for k, v in self.class_swap.items()}}
        for b in ann.boxes:
            b.u = W - (b.u + b.w)
            b.cls = swap.get(b.cls, b.cls)
        for pl in ann.polylines:
            pl.points[:, 0] = W - pl.points[:, 0]
            pl.cls = swap.get(pl.cls, pl.cls)
        for m in ann.masks:
            m.mask = m.mask[:, ::-1].copy()
        return image, ann


@dataclass
class RandomCrop:
    """Crop a random (h, w) window. Boxes are clipped; boxes keeping less than
    ``min_visibility`` of their area are dropped. Polyline points outside are dropped."""

    height: int
    width: int
    min_visibility: float = 0.3
    p: float = 1.0

    def __call__(self, image, ann, rng):
        H, W = image.shape[:2]
        if rng.random() >= self.p or (self.height >= H and self.width >= W):
            return image, ann
        ch, cw = min(self.height, H), min(self.width, W)
        y0 = int(rng.integers(0, H - ch + 1))
        x0 = int(rng.integers(0, W - cw + 1))
        image = image[y0:y0 + ch, x0:x0 + cw].copy()
        if ann is None:
            return image, ann
        out = Annotation(ann.image_id, cw, ch, tags=list(ann.tags), needs_review=ann.needs_review)
        for b in ann.boxes:
            u1, v1 = max(b.u - x0, 0.0), max(b.v - y0, 0.0)
            u2, v2 = min(b.u + b.w - x0, cw), min(b.v + b.h - y0, ch)
            nw, nh = u2 - u1, v2 - v1
            if nw <= 0 or nh <= 0 or b.area <= 0 or nw * nh / b.area < self.min_visibility:
                continue
            out.boxes.append(BoxLabel(b.cls, u1, v1, nw, nh, b.score, dict(b.attributes)))
        for pl in ann.polylines:
            pts = pl.points - np.array([x0, y0], dtype=float)
            keep = (pts[:, 0] >= 0) & (pts[:, 0] <= cw) & (pts[:, 1] >= 0) & (pts[:, 1] <= ch)
            # TODO(ml): clip segments at the crop border instead of dropping vertices.
            if keep.sum() >= 2:
                out.polylines.append(PolylineLabel(pl.cls, pts[keep], dict(pl.attributes)))
        for m in ann.masks:
            out.masks.append(MaskLabel(m.cls, m.mask[y0:y0 + ch, x0:x0 + cw].copy()))
        return image, out


@dataclass
class BrightnessContrast:
    """img' = (img - mean) * c + mean + b with b ~ U(+-brightness), c ~ U(1 +- contrast)."""

    brightness: float = 0.2
    contrast: float = 0.2
    p: float = 1.0

    def __call__(self, image, ann, rng):
        if rng.random() >= self.p:
            return image, ann
        f, dt = _to_float(image)
        b = rng.uniform(-self.brightness, self.brightness)
        c = rng.uniform(1 - self.contrast, 1 + self.contrast)
        mean = f.mean()
        return _restore((f - mean) * c + mean + b, dt), ann


@dataclass
class GaussianNoise:
    sigma: float = 0.03  # in [0,1] intensity units
    p: float = 1.0

    def __call__(self, image, ann, rng):
        if rng.random() >= self.p:
            return image, ann
        f, dt = _to_float(image)
        return _restore(f + rng.normal(0.0, self.sigma, size=f.shape), dt), ann


def gaussian_kernel1d(sigma: float, radius: Optional[int] = None) -> np.ndarray:
    radius = int(np.ceil(3 * sigma)) if radius is None else radius
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    return k / k.sum()


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with reflect padding (float in, float out)."""
    k = gaussian_kernel1d(sigma)
    r = len(k) // 2
    img = image.astype(np.float32)
    squeeze = img.ndim == 2
    if squeeze:
        img = img[..., None]
    pad = np.pad(img, ((r, r), (0, 0), (0, 0)), mode="reflect")
    img = sum(k[i] * pad[i:i + img.shape[0]] for i in range(len(k)))
    pad = np.pad(img, ((0, 0), (r, r), (0, 0)), mode="reflect")
    img = sum(k[i] * pad[:, i:i + img.shape[1]] for i in range(len(k)))
    return img[..., 0] if squeeze else img


@dataclass
class GaussianBlur:
    sigma_range: tuple[float, float] = (0.5, 1.5)
    p: float = 0.5

    def __call__(self, image, ann, rng):
        if rng.random() >= self.p:
            return image, ann
        f, dt = _to_float(image)
        return _restore(gaussian_blur(f, rng.uniform(*self.sigma_range)), dt), ann


@dataclass
class Cutout:
    """Fill ``n_holes`` random squares with ``fill``. Boxes covered more than
    ``max_occlusion`` are dropped (the object is no longer visible)."""

    n_holes: int = 1
    size: int = 16
    fill: float = 0.0
    max_occlusion: float = 0.7
    p: float = 0.5

    def __call__(self, image, ann, rng):
        if rng.random() >= self.p:
            return image, ann
        f, dt = _to_float(image)
        H, W = f.shape[:2]
        holes = []
        for _ in range(self.n_holes):
            cy, cx = int(rng.integers(0, H)), int(rng.integers(0, W))
            y1, y2 = max(cy - self.size // 2, 0), min(cy + self.size // 2, H)
            x1, x2 = max(cx - self.size // 2, 0), min(cx + self.size // 2, W)
            f[y1:y2, x1:x2] = self.fill
            holes.append((x1, y1, x2, y2))
        if ann is not None and ann.boxes:
            ann = ann.copy()
            occl = np.zeros((H, W), dtype=bool)
            for x1, y1, x2, y2 in holes:
                occl[y1:y2, x1:x2] = True
            kept = []
            for b in ann.boxes:
                u1, v1 = int(np.floor(b.u)), int(np.floor(b.v))
                u2, v2 = int(np.ceil(b.u + b.w)), int(np.ceil(b.v + b.h))
                region = occl[max(v1, 0):min(v2, H), max(u1, 0):min(u2, W)]
                if region.size == 0 or region.mean() <= self.max_occlusion:
                    kept.append(b)
            ann.boxes = kept
        return _restore(f, dt), ann


class Compose:
    """Apply transforms in order with one seeded RNG.

    # TODO(ml): photometric ops per channel (hue/saturation), affine/perspective
    # warps, mosaic/mixup, weather sims (rain, glare) for the iPhone camera domain.
    """

    def __init__(self, transforms: Sequence[Transform], seed: Optional[int] = 0) -> None:
        self.transforms = list(transforms)
        self.rng = np.random.default_rng(seed)

    def __call__(self, image: np.ndarray, ann: Optional[Annotation] = None):
        for t in self.transforms:
            image, ann = t(image, ann, self.rng)
        return image, ann
