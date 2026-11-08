"""Sensor plausibility checks per channel: freshness, NaN/Inf, range,
rate-of-change, frozen value, and timestamp sanity.

Each channel is a scalar or fixed-length vector (e.g. IMU accel xyz, GPS
lat/lon, wheel speed). Checks, in order (a non-finite sample skips the value
checks that would be meaningless):

  * SNS_TIME      stamp non-finite, in the future (> t + max_future_s) or older than the previous sample
  * SNS_NONFINITE any element NaN/Inf
  * SNS_RANGE     any element outside [lo, hi]
  * SNS_RATE      |dx/dt| > max_rate between consecutive valid samples
  * SNS_FROZEN    the last `frozen_samples` values all within `frozen_eps` (peak-to-peak)
                  - real sensors always carry noise, so an exactly constant signal
                  usually means a stuck driver/ADC.
  * SNS_STALE     from `check_freshness(t)`: no sample for > max_age_s (or never)

A sample that fails NaN/range is NOT used as the reference for the next rate check.
# TODO(safety): model-based checks (cross-sensor consistency e.g. wheel speed vs
# GPS speed vs IMU integration), noise-floor estimation, per-axis configs.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from cheesestack.core import Fault, Severity

from .faults import DEFAULT_REGISTRY, FaultRegistry

ArrayLike = Union[float, int, np.ndarray, list, tuple]


@dataclass
class SensorChannelConfig:
    name: str
    lo: ArrayLike = -math.inf
    hi: ArrayLike = math.inf
    max_age_s: float = 0.5
    max_rate: Optional[ArrayLike] = None  # units per second
    frozen_samples: int = 0  # 0 disables frozen detection
    frozen_eps: float = 0.0
    max_future_s: float = 0.05
    critical: bool = False  # escalate all faults of this channel to CRITICAL


@dataclass
class SensorReport:
    channel: str
    valid: bool
    faults: list[Fault] = field(default_factory=list)

    @property
    def codes(self) -> list[str]:
        return [f.code for f in self.faults]


class _ChannelState:
    def __init__(self, cfg: SensorChannelConfig) -> None:
        self.cfg = cfg
        self.last_value: Optional[np.ndarray] = None
        self.last_stamp: float = float("-inf")
        self.last_valid_stamp: float = float("-inf")
        self.recent: deque[np.ndarray] = deque(maxlen=max(1, cfg.frozen_samples))
        self.samples = 0
        self.rejected = 0


class SensorValidator:
    def __init__(self, channels: Optional[list[SensorChannelConfig]] = None,
                 registry: Optional[FaultRegistry] = None, source: str = "safety.sensors") -> None:
        self.registry = registry or DEFAULT_REGISTRY
        self.source = source
        self._ch: dict[str, _ChannelState] = {}
        for c in channels or []:
            self.add_channel(c)

    def add_channel(self, cfg: SensorChannelConfig) -> None:
        self._ch[cfg.name] = _ChannelState(cfg)

    def channels(self) -> list[str]:
        return list(self._ch)

    def _fault(self, st: _ChannelState, code: str, t: float, msg: str, **data) -> Fault:
        sev = Severity.CRITICAL if st.cfg.critical else None
        return self.registry.make(code, f"{self.source}/{st.cfg.name}", t, msg, sev, **data)

    def validate(self, name: str, value: ArrayLike, stamp: float, t: Optional[float] = None) -> SensorReport:
        """Validate one sample. `t` is the current time (defaults to `stamp`)."""
        st = self._ch[name]
        cfg = st.cfg
        t = float(stamp if t is None else t)
        x = np.atleast_1d(np.asarray(value, dtype=float))
        faults: list[Fault] = []

        if not math.isfinite(stamp) or stamp > t + cfg.max_future_s or stamp < st.last_stamp:
            faults.append(self._fault(st, "SNS_TIME", t, f"{name}: bad stamp {stamp} (now {t}, last {st.last_stamp})",
                                      stamp=stamp))
            st.rejected += 1
            return SensorReport(name, False, faults)

        st.samples += 1
        finite = bool(np.all(np.isfinite(x)))
        if not finite:
            faults.append(self._fault(st, "SNS_NONFINITE", t, f"{name}: non-finite value"))
        else:
            lo = np.broadcast_to(np.asarray(cfg.lo, dtype=float), x.shape)
            hi = np.broadcast_to(np.asarray(cfg.hi, dtype=float), x.shape)
            if np.any(x < lo) or np.any(x > hi):
                faults.append(self._fault(st, "SNS_RANGE", t, f"{name}: {x.tolist()} outside [{cfg.lo}, {cfg.hi}]",
                                          value=x.tolist()))
            elif cfg.max_rate is not None and st.last_value is not None and stamp > st.last_stamp:
                dt = stamp - st.last_stamp
                rate = np.abs(x - st.last_value) / dt
                mr = np.broadcast_to(np.asarray(cfg.max_rate, dtype=float), x.shape)
                if np.any(rate > mr):
                    faults.append(self._fault(st, "SNS_RATE", t, f"{name}: rate {rate.max():.3g}/s > {cfg.max_rate}",
                                              rate=rate.tolist()))

        value_ok = finite and not any(f.code == "SNS_RANGE" for f in faults)
        if value_ok:
            if cfg.frozen_samples > 1:
                st.recent.append(x.copy())
                if len(st.recent) == cfg.frozen_samples:
                    arr = np.stack(st.recent)
                    if np.all(np.ptp(arr, axis=0) <= cfg.frozen_eps):
                        faults.append(self._fault(st, "SNS_FROZEN", t,
                                                  f"{name}: constant over {cfg.frozen_samples} samples"))
            st.last_value = x
            st.last_valid_stamp = stamp
        st.last_stamp = stamp
        if faults:
            st.rejected += 1
        # SNS_RATE alone still marks the sample invalid; consumers should hold the last good value.
        return SensorReport(name, not faults, faults)

    def check_freshness(self, t: float) -> list[Fault]:
        """Stale channels: no *valid* sample within max_age_s."""
        out = []
        for st in self._ch.values():
            age = t - st.last_valid_stamp
            if age > st.cfg.max_age_s:
                out.append(self._fault(st, "SNS_STALE", t, f"{st.cfg.name}: age {age:.3f}s > {st.cfg.max_age_s}s",
                                       age_s=age))
        return out

    def last_valid(self, name: str) -> Optional[np.ndarray]:
        return self._ch[name].last_value

    def health(self, t: float) -> float:
        """Fraction of channels that are fresh (valid sample within max_age_s)."""
        if not self._ch:
            return 1.0
        fresh = sum(1 for st in self._ch.values() if t - st.last_valid_stamp <= st.cfg.max_age_s)
        return fresh / len(self._ch)
