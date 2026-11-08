"""Lane detection: model interface + classical bird's-eye baseline.

Pipeline of :class:`ClassicalLaneDetector`:

1. (optional) warp the camera image to a bird's-eye view (BEV) with a
   homography (`warp_perspective`, nearest-neighbor, numpy only).
2. Binary lane-marking mask: Sobel-x gradient threshold OR bright/yellow
   color threshold.
3. Histogram of the bottom half picks the left/right base columns, then a
   sliding-window search collects marking pixels up the image (or a
   search around the previous fit when tracking).
4. Each lane line is fit in metric base_link coordinates as
   ``y_left = c0 + c1 * x + c2 * x^2`` (x forward), giving lane width,
   ego lateral offset from lane center and signed curvature.

BEV convention: bottom row is nearest the ego, row index grows toward the
ego; column `ego_u` (default image center) is the ego centerline. Metric:
``x = (H - 1 - v) * m_per_px_x + x_offset_m``, ``y = (ego_u - u) * m_per_px_y``.

Limits: assumes flat ground, a known homography, two visible markings and
modest curvature. No dashed/solid typing, no multi-lane output.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cheesestack.core import Header

from .detection import sobel_xy, to_float_image, to_gray, rgb_to_hsv


def warp_perspective(image: np.ndarray, H: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Warp `image` with homography H (src pixel -> dst pixel), nearest neighbor.

    out_shape = (height, width). Pixels mapping outside the source are 0.
    """
    img = np.asarray(image)
    oh, ow = out_shape
    Hinv = np.linalg.inv(np.asarray(H, dtype=float))
    vv, uu = np.mgrid[0:oh, 0:ow]
    pts = np.stack([uu.ravel(), vv.ravel(), np.ones(oh * ow)], axis=0).astype(float)
    src = Hinv @ pts
    with np.errstate(divide="ignore", invalid="ignore"):
        su = src[0] / src[2]
        sv = src[1] / src[2]
    su_i = np.round(np.clip(np.nan_to_num(su, nan=-1.0, posinf=-1.0, neginf=-1.0), -1, 1e9)).astype(np.int64)
    sv_i = np.round(np.clip(np.nan_to_num(sv, nan=-1.0, posinf=-1.0, neginf=-1.0), -1, 1e9)).astype(np.int64)
    valid = (su_i >= 0) & (su_i < img.shape[1]) & (sv_i >= 0) & (sv_i < img.shape[0]) & (src[2] > 0)
    out = np.zeros((oh, ow) + img.shape[2:], dtype=img.dtype)
    flat = out.reshape(oh * ow, *img.shape[2:])
    flat[valid] = img[sv_i[valid], su_i[valid]]
    return out
    # TODO(perception): bilinear sampling + per-camera cached remap tables.


@dataclass
class LaneLine:
    """One lane boundary. `coeffs_m` = (c0, c1, c2): y_left = c0 + c1 x + c2 x^2."""

    coeffs_m: np.ndarray
    coeffs_px: np.ndarray  # u = a v^2 + b v + c (BEV pixels)
    n_pixels: int
    side: str  # "left" | "right"

    def y_at(self, x: float) -> float:
        c0, c1, c2 = self.coeffs_m
        return float(c0 + c1 * x + c2 * x * x)


@dataclass
class LaneEstimate:
    header: Header
    left: Optional[LaneLine] = None
    right: Optional[LaneLine] = None
    lane_width_m: float = float("nan")
    center_offset_m: float = float("nan")  # ego y minus lane-center y (>0: ego left of center)
    heading_error_rad: float = float("nan")  # lane heading relative to ego x axis at x=0
    curvature: float = 0.0  # 1/m, signed (CCW / left turn positive)
    confidence: float = 0.0
    debug: dict = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.left is not None and self.right is not None and self.confidence > 0


class LaneModel(abc.ABC):
    """Interface for lane detectors (classical or learned)."""

    @abc.abstractmethod
    def detect(self, image: np.ndarray, header: Header) -> LaneEstimate:
        ...

    def reset(self) -> None:
        """Drop temporal state."""


@dataclass
class LaneDetectorConfig:
    # BEV geometry
    m_per_px_x: float = 0.1  # forward metres per BEV row
    m_per_px_y: float = 0.05  # lateral metres per BEV column
    x_offset_m: float = 0.0  # forward distance of the bottom BEV row from base_link
    ego_u: Optional[float] = None  # ego centerline column; None = image center
    homography: Optional[np.ndarray] = None  # camera -> BEV pixel homography
    bev_shape: Optional[tuple[int, int]] = None
    # thresholds
    grad_threshold: float = 0.08
    bright_threshold: float = 0.75
    yellow_hue: tuple[float, float] = (0.10, 0.19)
    yellow_s_min: float = 0.35
    # search
    n_windows: int = 9
    margin_px: int = 12
    min_pix_recenter: int = 8
    min_pix_line: int = 40
    prior_margin_px: int = 15
    use_prior: bool = True
    # sanity
    expected_width_m: float = 3.5
    width_tolerance_m: float = 1.2
    smoothing: float = 0.5  # EMA on coefficients (0 = no smoothing)


class ClassicalLaneDetector(LaneModel):
    """Gradient/color threshold + histogram/sliding-window + quadratic fit."""

    def __init__(self, config: Optional[LaneDetectorConfig] = None) -> None:
        self.config = config or LaneDetectorConfig()
        self._prev: Optional[LaneEstimate] = None

    def reset(self) -> None:
        self._prev = None

    # -- stages ---------------------------------------------------------
    def to_bev(self, image: np.ndarray) -> np.ndarray:
        cfg = self.config
        if cfg.homography is None:
            return image
        shape = cfg.bev_shape or image.shape[:2]
        return warp_perspective(image, cfg.homography, shape)

    def binary(self, bev: np.ndarray) -> np.ndarray:
        cfg = self.config
        gray = to_gray(bev)
        gx, _ = sobel_xy(gray)
        grad = np.abs(gx) > cfg.grad_threshold
        bright = gray > cfg.bright_threshold
        mask = bright
        if bev.ndim == 3:
            hsv = rgb_to_hsv(to_float_image(bev))
            h, s = hsv[..., 0], hsv[..., 1]
            yellow = (h >= cfg.yellow_hue[0]) & (h <= cfg.yellow_hue[1]) & (s >= cfg.yellow_s_min) & (hsv[..., 2] > 0.4)
            mask = mask | yellow
        # Gradient pixels only count next to a bright/yellow region (rejects shadows edges).
        near = np.zeros_like(mask)
        near[:, 1:] |= mask[:, :-1]
        near[:, :-1] |= mask[:, 1:]
        return mask | (grad & near)

    def _px_to_metric(self, u: np.ndarray, v: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        ego_u = cfg.ego_u if cfg.ego_u is not None else (w - 1) / 2.0
        x = (h - 1 - v) * cfg.m_per_px_x + cfg.x_offset_m
        y = (ego_u - u) * cfg.m_per_px_y
        return x, y

    def _sliding_window(self, mask: np.ndarray, base: int) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        h, w = mask.shape
        vs, us = np.nonzero(mask)
        win_h = max(1, h // cfg.n_windows)
        cur = base
        sel_all = []
        for k in range(cfg.n_windows):
            v_hi = h - k * win_h
            v_lo = max(0, v_hi - win_h)
            sel = (vs >= v_lo) & (vs < v_hi) & (us >= cur - cfg.margin_px) & (us < cur + cfg.margin_px)
            idx = np.nonzero(sel)[0]
            sel_all.append(idx)
            if idx.size >= cfg.min_pix_recenter:
                cur = int(round(us[idx].mean()))
        idx = np.concatenate(sel_all) if sel_all else np.zeros(0, int)
        return us[idx], vs[idx]

    def _prior_search(self, mask: np.ndarray, coeffs_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vs, us = np.nonzero(mask)
        pred = np.polyval(coeffs_px, vs)
        sel = np.abs(us - pred) < self.config.prior_margin_px
        return us[sel], vs[sel]

    def _fit(self, us, vs, h, w, side) -> Optional[LaneLine]:
        if us.size < self.config.min_pix_line or np.ptp(vs) < h * 0.2:
            return None
        px = np.polyfit(vs.astype(float), us.astype(float), 2)
        x, y = self._px_to_metric(us.astype(float), vs.astype(float), h, w)
        cm = np.polyfit(x, y, 2)[::-1]  # (c0, c1, c2)
        return LaneLine(coeffs_m=cm, coeffs_px=px, n_pixels=int(us.size), side=side)

    # -- main -----------------------------------------------------------
    def detect(self, image: np.ndarray, header: Header) -> LaneEstimate:
        cfg = self.config
        bev = self.to_bev(image)
        mask = self.binary(bev)
        h, w = mask.shape
        ego_u = cfg.ego_u if cfg.ego_u is not None else (w - 1) / 2.0
        prev = self._prev if (cfg.use_prior and self._prev is not None and self._prev.valid) else None

        if prev is not None:
            lu, lv = self._prior_search(mask, prev.left.coeffs_px)
            ru, rv = self._prior_search(mask, prev.right.coeffs_px)
        else:
            hist = mask[h // 2:, :].sum(axis=0).astype(float)
            mid = int(round(ego_u))
            lbase = int(np.argmax(hist[:mid])) if mid > 0 else 0
            rbase = mid + int(np.argmax(hist[mid:])) if mid < w else w - 1
            lu, lv = self._sliding_window(mask, lbase)
            ru, rv = self._sliding_window(mask, rbase)

        left = self._fit(lu, lv, h, w, "left")
        right = self._fit(ru, rv, h, w, "right")
        est = LaneEstimate(header=header, left=left, right=right, debug={"mask": mask})
        if left is None or right is None:
            if prev is not None:
                self._prev = None  # fall back to a full search next frame
            est.confidence = 0.0
            return est

        if prev is not None and cfg.smoothing > 0:
            a = cfg.smoothing
            left.coeffs_m = a * prev.left.coeffs_m + (1 - a) * left.coeffs_m
            right.coeffs_m = a * prev.right.coeffs_m + (1 - a) * right.coeffs_m

        x0 = cfg.x_offset_m
        yl, yr = left.y_at(x0), right.y_at(x0)
        cc = 0.5 * (left.coeffs_m + right.coeffs_m)
        est.lane_width_m = yl - yr
        est.center_offset_m = 0.0 - (cc[0] + cc[1] * x0 + cc[2] * x0 * x0)
        slope = cc[1] + 2 * cc[2] * x0
        est.heading_error_rad = float(np.arctan(slope))
        est.curvature = float(2 * cc[2] / (1 + slope * slope) ** 1.5)

        width_err = abs(est.lane_width_m - cfg.expected_width_m)
        width_conf = float(np.clip(1.0 - width_err / cfg.width_tolerance_m, 0.0, 1.0))
        support = min(left.n_pixels, right.n_pixels) / (cfg.min_pix_line * 4.0)
        est.confidence = float(width_conf * np.clip(support, 0.0, 1.0))
        self._prev = est if est.confidence > 0 else None
        return est
        # TODO(perception): learned lane model (e.g. row-anchor network) behind LaneModel,
        # TODO(perception): Kalman filter on lane coefficients with ego-motion compensation,
        # TODO(perception): multi-lane output, marking type (dashed/solid, color), single-line fallback.
