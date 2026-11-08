"""Lane-centering target generation from lane boundary polynomials.

Boundaries are cubic polynomials in base_link: y(x) = c0 + c1 x + c2 x^2 + c3 x^3
(coefficients low-order first, as `np.polynomial.polynomial` uses). The centreline
is the mean of left/right; with one boundary it is offset by half a nominal lane
width. The lateral offset target is evaluated at a speed-dependent lookahead
and passed through a deadband (hysteresis) + rate limit so small perception
jitter does not translate into steering wiggle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

P = np.polynomial.polynomial


@dataclass
class LaneBoundaries:
    left: Optional[np.ndarray] = None  # coefficients, low order first
    right: Optional[np.ndarray] = None
    left_confidence: float = 1.0
    right_confidence: float = 1.0


@dataclass
class LaneCenteringConfig:
    nominal_lane_width: float = 3.6
    min_lane_width: float = 2.5
    max_lane_width: float = 5.0
    min_confidence: float = 0.3
    lookahead_time_s: float = 1.0
    min_lookahead_m: float = 5.0
    max_lookahead_m: float = 40.0
    hysteresis_m: float = 0.10  # ignore target changes smaller than this
    max_rate_mps: float = 0.5  # lateral target slew limit
    bias_m: float = 0.0  # desired offset from centre (+left), e.g. to give trucks room


@dataclass
class LaneCenteringOutput:
    valid: bool
    target_offset: float  # lateral offset of target point in base_link, m
    lookahead_m: float
    heading_error: float  # rad, centreline heading at lookahead vs ego
    curvature: float  # centreline curvature at lookahead, 1/m
    lane_width: Optional[float]
    centerline: Optional[np.ndarray]
    reason: str = ""


def centerline_from_boundaries(b: LaneBoundaries, cfg: LaneCenteringConfig) -> tuple[Optional[np.ndarray], Optional[float], str]:
    """Returns (centreline coeffs, lane width at x=0, reason)."""
    use_l = b.left is not None and b.left_confidence >= cfg.min_confidence
    use_r = b.right is not None and b.right_confidence >= cfg.min_confidence
    if use_l and use_r:
        l, r = np.asarray(b.left, float), np.asarray(b.right, float)
        n = max(len(l), len(r))
        l, r = np.pad(l, (0, n - len(l))), np.pad(r, (0, n - len(r)))
        width = float(P.polyval(0.0, l) - P.polyval(0.0, r))
        if not (cfg.min_lane_width <= width <= cfg.max_lane_width):
            # Implausible width: trust the more confident side.
            if b.left_confidence >= b.right_confidence:
                use_r = False
            else:
                use_l = False
        else:
            return 0.5 * (l + r), width, "both boundaries"
    if use_l:
        c = np.asarray(b.left, float).copy()
        c[0] -= cfg.nominal_lane_width / 2
        return c, None, "left only"
    if use_r:
        c = np.asarray(b.right, float).copy()
        c[0] += cfg.nominal_lane_width / 2
        return c, None, "right only"
    return None, None, "no usable boundary"


class LaneCentering:
    def __init__(self, config: LaneCenteringConfig = LaneCenteringConfig()) -> None:
        self.config = config
        self._target: Optional[float] = None

    def reset(self) -> None:
        self._target = None

    def update(self, boundaries: LaneBoundaries, speed: float, dt: float) -> LaneCenteringOutput:
        cfg = self.config
        center, width, reason = centerline_from_boundaries(boundaries, cfg)
        la = float(np.clip(speed * cfg.lookahead_time_s, cfg.min_lookahead_m, cfg.max_lookahead_m))
        if center is None:
            return LaneCenteringOutput(False, self._target or 0.0, la, 0.0, 0.0, None, None, reason)
        raw = float(P.polyval(la, center)) + cfg.bias_m
        d1 = P.polyder(center)
        d2 = P.polyder(center, 2) if len(center) > 2 else np.zeros(1)
        yp, ypp = float(P.polyval(la, d1)), float(P.polyval(la, d2))
        heading = math.atan(yp)
        kappa = ypp / (1 + yp**2) ** 1.5

        if self._target is None:
            self._target = raw
        elif abs(raw - self._target) > cfg.hysteresis_m:
            step = float(np.clip(raw - self._target, -cfg.max_rate_mps * dt, cfg.max_rate_mps * dt))
            self._target += step
        return LaneCenteringOutput(True, self._target, la, heading, kappa, width, center, reason)

    # TODO(planning): fuse with map lane geometry, lane-width-aware bias (wide lanes,
    # trucks in adjacent lane), confidence-weighted blend instead of hard switching.
