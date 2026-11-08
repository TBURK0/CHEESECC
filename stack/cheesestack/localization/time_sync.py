"""Timestamp alignment: sensor clock offset/drift estimation and an interpolation buffer.

Model: host_receive = (1 + drift) * sensor_stamp + offset + delay, delay >= 0.
`ClockOffsetEstimator` fits drift/offset by least squares on a sliding window,
then shifts the intercept down to the lower envelope (minimum residual), since
transport delay is one-sided. `to_host()` maps sensor stamps onto the host
timeline that the rest of the stack uses.

`InterpolationBuffer` stores time-ordered samples (arrays or Pose2D) and returns
linearly interpolated values with angle-aware interpolation for chosen indices.
"""

from __future__ import annotations

import bisect
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from cheesestack.core import Pose2D, wrap_angle


@dataclass
class ClockSyncConfig:
    window: int = 200
    min_samples: int = 5
    estimate_drift: bool = True
    envelope_quantile: float = 0.05  # robust "minimum" residual


class ClockOffsetEstimator:
    def __init__(self, config: Optional[ClockSyncConfig] = None) -> None:
        self.config = config or ClockSyncConfig()
        self._pairs: deque = deque(maxlen=self.config.window)
        self.offset = 0.0
        self.drift = 0.0
        self.mean_delay = 0.0
        self._t_ref = 0.0

    @property
    def ready(self) -> bool:
        return len(self._pairs) >= self.config.min_samples

    def add(self, sensor_stamp: float, host_stamp: float) -> None:
        self._pairs.append((float(sensor_stamp), float(host_stamp)))
        if self.ready:
            self._fit()

    def _fit(self) -> None:
        a = np.array(self._pairs)
        s, h = a[:, 0], a[:, 1]
        self._t_ref = float(s[0])
        ds = s - self._t_ref
        if self.config.estimate_drift and np.ptp(ds) > 1e-9:
            A = np.column_stack([ds, np.ones_like(ds)])
            coef, *_ = np.linalg.lstsq(A, h - s, rcond=None)
            drift, icpt = float(coef[0]), float(coef[1])
        else:
            drift, icpt = 0.0, float(np.mean(h - s))
        res = (h - s) - (drift * ds + icpt)
        shift = float(np.quantile(res, self.config.envelope_quantile))
        self.drift = drift
        self.offset = icpt + shift
        self.mean_delay = float(np.mean(res - shift))

    def to_host(self, sensor_stamp: float) -> float:
        """Map a sensor stamp to host time (the event time, i.e. excluding transport delay)."""
        ds = sensor_stamp - self._t_ref
        return sensor_stamp + self.drift * ds + self.offset
    # TODO(localization): PTP/PPS hardware sync, convex-hull lower-envelope fit, jump detection on
    # sensor reboot, and per-sensor latency calibration against a common motion event.


class InterpolationBuffer:
    """Bounded, time-sorted buffer with linear (and angle-aware) interpolation.

    Values may be np arrays / floats or Pose2D (yaw interpolated on the circle).
    """

    def __init__(self, max_age_s: float = 5.0, max_len: int = 10000,
                 angle_indices: Sequence[int] = (), max_extrapolation_s: float = 0.0) -> None:
        self.max_age_s = max_age_s
        self.max_len = max_len
        self.angle_indices = tuple(angle_indices)
        self.max_extrapolation_s = max_extrapolation_s
        self._t: list[float] = []
        self._v: list[Any] = []

    def __len__(self) -> int:
        return len(self._t)

    @property
    def span(self) -> tuple[float, float]:
        return (self._t[0], self._t[-1]) if self._t else (math.nan, math.nan)

    def add(self, t: float, value: Any) -> None:
        i = bisect.bisect_right(self._t, t)
        self._t.insert(i, float(t))
        self._v.insert(i, value)
        newest = self._t[-1]
        while self._t and (newest - self._t[0] > self.max_age_s or len(self._t) > self.max_len):
            self._t.pop(0)
            self._v.pop(0)

    def nearest(self, t: float) -> Optional[tuple[float, Any]]:
        if not self._t:
            return None
        i = bisect.bisect_left(self._t, t)
        cands = [j for j in (i - 1, i) if 0 <= j < len(self._t)]
        j = min(cands, key=lambda k: abs(self._t[k] - t))
        return self._t[j], self._v[j]

    def _lerp(self, a: Any, b: Any, w: float) -> Any:
        if isinstance(a, Pose2D):
            return Pose2D(a.x + w * (b.x - a.x), a.y + w * (b.y - a.y), wrap_angle(a.yaw + w * wrap_angle(b.yaw - a.yaw)))
        av, bv = np.asarray(a, float), np.asarray(b, float)
        out = av + w * (bv - av)
        for i in self.angle_indices:
            out[i] = wrap_angle(av[i] + w * wrap_angle(bv[i] - av[i]))
        return out if out.ndim else float(out)

    def interpolate(self, t: float) -> Optional[Any]:
        """Value at t, or None if outside [first - ext, last + ext] or buffer has < 1 sample."""
        n = len(self._t)
        if n == 0:
            return None
        if n == 1:
            return self._v[0] if abs(t - self._t[0]) <= self.max_extrapolation_s else None
        if t < self._t[0] - self.max_extrapolation_s or t > self._t[-1] + self.max_extrapolation_s:
            return None
        i = bisect.bisect_left(self._t, t)
        i = min(max(i, 1), n - 1)
        t0, t1 = self._t[i - 1], self._t[i]
        w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        return self._lerp(self._v[i - 1], self._v[i], w)

    def between(self, t0: float, t1: float) -> list[tuple[float, Any]]:
        i = bisect.bisect_left(self._t, t0)
        j = bisect.bisect_right(self._t, t1)
        return list(zip(self._t[i:j], self._v[i:j]))
