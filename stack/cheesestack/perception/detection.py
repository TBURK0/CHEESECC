"""Object detection: detector interface, NMS, and a connected-components blob baseline.

Also hosts small numpy image helpers (grayscale, Sobel, HSV, connected
components, hole filling) shared by the scene-understanding modules
(lanes, drivable space, signs, lights) so they need no OpenCV.

The baseline :class:`BlobDetector` finds foreground blobs either from a
caller-supplied mask, a background frame (frame differencing) or, failing
both, saliency against the image's median color. It is a stand-in for a
learned detector (YOLO-style) and is only useful in controlled scenes.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from cheesestack.core import BoundingBox2D, Detection, Header, ObjectClass


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def to_float_image(image: np.ndarray) -> np.ndarray:
    """Return image as float32 in [0, 1] (accepts uint8 or float)."""
    img = np.asarray(image)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return np.clip(img.astype(np.float32), 0.0, 1.0)


def to_gray(image: np.ndarray) -> np.ndarray:
    """HxW float32 luminance (Rec.601) from HxW or HxWx3 input."""
    img = to_float_image(image)
    if img.ndim == 2:
        return img
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)


def convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Same-size 2D correlation with edge padding (small kernels only)."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="edge")
    out = np.zeros_like(img, dtype=np.float32)
    h, w = img.shape
    for i in range(kh):
        for j in range(kw):
            if kernel[i, j] != 0:
                out += kernel[i, j] * padded[i:i + h, j:j + w]
    return out


_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0


def sobel_xy(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sobel gradients (gx, gy), normalized so a unit step gives ~0.5."""
    return convolve2d(gray, _SOBEL_X), convolve2d(gray, _SOBEL_X.T)


def box_filter(img: np.ndarray, k: int) -> np.ndarray:
    """Mean filter of odd size k via integral image."""
    r = k // 2
    p = np.pad(img.astype(np.float64), ((r + 1, r), (r + 1, r)), mode="edge")
    s = p.cumsum(0).cumsum(1)
    h, w = img.shape
    tot = s[k:k + h, k:k + w] - s[0:h, k:k + w] - s[k:k + h, 0:w] + s[0:h, 0:w]
    return (tot / (k * k)).astype(np.float32)


def rgb_to_hsv(image: np.ndarray) -> np.ndarray:
    """HxWx3 RGB -> HSV, all channels in [0, 1] (hue 0 = red, wraps)."""
    img = to_float_image(image)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx = img.max(axis=-1)
    mn = img.min(axis=-1)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-6
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = ((g[rm] - b[rm]) / d[rm]) % 6.0
    h[gm] = (b[gm] - r[gm]) / d[gm] + 2.0
    h[bm] = (r[bm] - g[bm]) / d[bm] + 4.0
    h = h / 6.0
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    return np.stack([h, s, mx], axis=-1).astype(np.float32)


def hue_mask(hsv: np.ndarray, hue_lo: float, hue_hi: float, s_min: float, v_min: float) -> np.ndarray:
    """Boolean mask of pixels with hue in [lo, hi] (wraps if lo > hi) and S, V above minimums."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if hue_lo <= hue_hi:
        hm = (h >= hue_lo) & (h <= hue_hi)
    else:
        hm = (h >= hue_lo) | (h <= hue_hi)
    return hm & (s >= s_min) & (v >= v_min)


def connected_components(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    """Label connected True regions. Returns (labels HxW int32, 0 = background, n).

    Vectorized min-label propagation with pointer jumping; converges in
    O(log diameter)-ish iterations for typical blobs.
    """
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    big = np.int64(h * w + 1)
    idx = np.arange(h * w, dtype=np.int64).reshape(h, w)
    lab = np.where(mask, idx, big)
    while True:
        p = np.pad(lab, 1, mode="constant", constant_values=big)
        nb = [p[1:-1, 1:-1], p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]]
        if connectivity == 8:
            nb += [p[:-2, :-2], p[:-2, 2:], p[2:, :-2], p[2:, 2:]]
        new = np.minimum.reduce(nb)
        new = np.where(mask, new, big)
        # pointer jumping: follow label -> label of that pixel
        flat = new.ravel()
        m = flat < big
        flat_j = flat.copy()
        flat_j[m] = flat[flat[m]]
        new = np.minimum(new, flat_j.reshape(h, w))
        if np.array_equal(new, lab):
            break
        lab = new
    out = np.zeros((h, w), dtype=np.int32)
    roots = np.unique(lab[mask])
    if roots.size:
        out[mask] = (np.searchsorted(roots, lab[mask]) + 1).astype(np.int32)
    return out, int(roots.size)


def reconstruct(seed: np.ndarray, mask: np.ndarray, connectivity: int = 4, max_iter: int = 10000) -> np.ndarray:
    """Morphological reconstruction: grow `seed` inside `mask` until stable (flood fill)."""
    cur = seed & mask
    for _ in range(max_iter):
        p = np.pad(cur, 1)
        grown = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
        if connectivity == 8:
            grown |= p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
        grown &= mask
        if np.array_equal(grown, cur):
            return cur
        cur = grown
    return cur


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed background holes of a boolean mask."""
    bg = ~mask
    border = np.zeros_like(mask)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    outside = reconstruct(border & bg, bg)
    return ~outside


def component_boxes(labels: np.ndarray, n: int) -> list[tuple[int, BoundingBox2D, int]]:
    """(label, box, area) per component."""
    out = []
    if n == 0:
        return out
    vs, us = np.nonzero(labels)
    ls = labels[vs, us]
    for k in range(1, n + 1):
        sel = ls == k
        if not sel.any():
            continue
        u0, u1 = us[sel].min(), us[sel].max()
        v0, v1 = vs[sel].min(), vs[sel].max()
        out.append((k, BoundingBox2D(float(u0), float(v0), float(u1 - u0 + 1), float(v1 - v0 + 1)), int(sel.sum())))
    return out


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def boxes_to_array(boxes: Sequence[BoundingBox2D]) -> np.ndarray:
    """(N, 4) array of x1, y1, x2, y2."""
    return np.array([[b.u, b.v, b.u + b.w, b.v + b.h] for b in boxes], dtype=float).reshape(-1, 4)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy arrays."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aa[:, None] + ab[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """Greedy non-maximum suppression. Returns kept indices sorted by score desc."""
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ious = iou_matrix(boxes_xyxy[i:i + 1], boxes_xyxy[order[1:]])[0]
        order = order[1:][ious <= iou_threshold]
    return np.array(keep, dtype=int)


def nms_detections(dets: Sequence[Detection], iou_threshold: float = 0.5, class_aware: bool = True) -> list[Detection]:
    """NMS over Detections with box2d. Class-aware suppresses only within a class."""
    with_box = [d for d in dets if d.box2d is not None]
    if not with_box:
        return list(dets)
    groups: dict = {}
    for d in with_box:
        groups.setdefault(d.cls if class_aware else None, []).append(d)
    kept: list[Detection] = []
    for g in groups.values():
        idx = nms(boxes_to_array([d.box2d for d in g]), np.array([d.score for d in g]), iou_threshold)
        kept.extend(g[i] for i in idx)
    kept.sort(key=lambda d: -d.score)
    return kept + [d for d in dets if d.box2d is None]


# ---------------------------------------------------------------------------
# Detector interface + baseline
# ---------------------------------------------------------------------------

class Detector(abc.ABC):
    """Image -> list of core.Detection (box2d in pixels, header copied)."""

    @abc.abstractmethod
    def detect(self, image: np.ndarray, header: Header) -> list[Detection]:
        ...


class ModelDetector(Detector):
    """Adapter for a learned detector.

    `infer(image) -> (boxes_xyxy (N,4), scores (N,), labels list[str])`. Labels
    are mapped to ObjectClass by `label_map` (see classification.map_label).
    """

    def __init__(self, infer: Callable, label_map: Callable[[str], ObjectClass],
                 score_threshold: float = 0.3, iou_threshold: float = 0.5) -> None:
        self.infer = infer
        self.label_map = label_map
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

    def detect(self, image: np.ndarray, header: Header) -> list[Detection]:
        boxes, scores, labels = self.infer(image)
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        scores = np.asarray(scores, dtype=float)
        dets = []
        for (x1, y1, x2, y2), s, lab in zip(boxes, scores, labels):
            if s < self.score_threshold:
                continue
            dets.append(Detection(header, self.label_map(lab), float(s),
                                  box2d=BoundingBox2D(x1, y1, x2 - x1, y2 - y1), attributes={"label": lab}))
        return nms_detections(dets, self.iou_threshold)
        # TODO(perception): batched inference, letterbox un-mapping, per-class thresholds.


@dataclass
class BlobDetectorConfig:
    diff_threshold: float = 0.15  # color distance for foreground
    min_area_px: int = 20
    max_area_frac: float = 0.5  # reject blobs larger than this fraction of image
    min_aspect: float = 0.1  # h/w bounds
    max_aspect: float = 10.0
    connectivity: int = 8
    nms_iou: float = 0.5
    min_score: float = 0.1


class BlobDetector(Detector):
    """Foreground blob detector via connected components.

    Foreground = |image - reference| > threshold, where reference is the
    background frame if given, else the per-channel image median. Score grows
    with blob contrast and size. Class is UNKNOWN unless a `classifier`
    callable `(image, box) -> (ObjectClass, score)` is given.
    """

    def __init__(self, config: Optional[BlobDetectorConfig] = None,
                 classifier: Optional[Callable] = None) -> None:
        self.config = config or BlobDetectorConfig()
        self.classifier = classifier
        self.background: Optional[np.ndarray] = None

    def set_background(self, image: np.ndarray) -> None:
        self.background = to_float_image(image)

    def foreground_mask(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        img = to_float_image(image)
        if img.ndim == 2:
            img = img[..., None]
        if self.background is not None:
            ref = self.background if self.background.ndim == 3 else self.background[..., None]
        else:
            ref = np.median(img.reshape(-1, img.shape[-1]), axis=0)[None, None, :]
        dist = np.sqrt(((img - ref) ** 2).mean(axis=-1))
        return dist > self.config.diff_threshold, dist

    def detect(self, image: np.ndarray, header: Header, mask: Optional[np.ndarray] = None) -> list[Detection]:
        cfg = self.config
        if mask is None:
            mask, dist = self.foreground_mask(image)
        else:
            mask = np.asarray(mask, dtype=bool)
            dist = mask.astype(np.float32)
        labels, n = connected_components(mask, cfg.connectivity)
        hw = mask.size
        dets = []
        for k, box, area in component_boxes(labels, n):
            if area < cfg.min_area_px or area > cfg.max_area_frac * hw:
                continue
            aspect = box.h / max(box.w, 1e-6)
            if not (cfg.min_aspect <= aspect <= cfg.max_aspect):
                continue
            contrast = float(dist[labels == k].mean())
            fill = area / max(box.area, 1.0)
            score = float(np.clip(contrast * 2.0, 0, 1) * (0.5 + 0.5 * fill) * (1 - np.exp(-area / 200.0)))
            if score < cfg.min_score:
                continue
            cls, cscore = ObjectClass.UNKNOWN, 1.0
            if self.classifier is not None:
                cls, cscore = self.classifier(image, box)
            dets.append(Detection(header, cls, float(score * cscore), box2d=box,
                                  attributes={"area_px": area, "fill": fill, "contrast": contrast}))
        return nms_detections(dets, cfg.nms_iou)
        # TODO(perception): replace with learned detector (ModelDetector); add 3D lifting via ground plane.
