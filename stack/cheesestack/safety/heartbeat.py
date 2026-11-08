"""Heartbeat monitoring: statistics on periodic signals.

Where the watchdog asks "is it late?", the heartbeat monitor characterizes the
stream: measured rate, jitter (std-dev of inter-arrival intervals), and an
estimate of missed beats. Liveness:

    age <= stale_factor * period   -> ALIVE
    age <= dead_factor  * period   -> STALE   (HB_STALE, DEGRADED)
    otherwise / never seen         -> DEAD    (HB_DEAD, CRITICAL)

Missed beats per interval = max(0, round(interval / period) - 1), accumulated.
Out-of-order / duplicate beats (t <= last) are counted and ignored.
# TODO(safety): per-source configurable fault severities; correlate with
# transport-level E2E counters to distinguish producer stalls from bus loss.
"""

from __future__ import annotations

import enum
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cheesestack.core import Fault

from .faults import DEFAULT_REGISTRY, FaultRegistry


class Liveness(enum.IntEnum):
    ALIVE = 0
    STALE = 1
    DEAD = 2


@dataclass
class HeartbeatConfig:
    expected_rate_hz: float = 10.0
    stale_factor: float = 2.5
    dead_factor: float = 10.0
    max_jitter_frac: float = 0.5  # jitter std / period above this -> HB_JITTER
    min_rate_frac: float = 0.7  # measured rate below frac*expected -> HB_RATE
    window: int = 50  # intervals kept for statistics
    min_samples: int = 5  # intervals needed before jitter/rate faults


@dataclass
class HeartbeatStats:
    name: str
    liveness: Liveness
    age_s: float
    rate_hz: float
    jitter_s: float
    missed: int
    beats: int
    out_of_order: int


class _Source:
    def __init__(self, name: str, cfg: HeartbeatConfig, registered_at: float) -> None:
        self.name = name
        self.cfg = cfg
        self.registered_at = registered_at
        self.last = float("-inf")
        self.intervals: deque[float] = deque(maxlen=cfg.window)
        self.missed = 0
        self.beats = 0
        self.out_of_order = 0

    @property
    def period(self) -> float:
        return 1.0 / self.cfg.expected_rate_hz

    def beat(self, t: float) -> None:
        if t <= self.last:
            self.out_of_order += 1
            return
        if self.last != float("-inf"):
            dt = t - self.last
            self.intervals.append(dt)
            self.missed += max(0, int(round(dt / self.period)) - 1)
        self.last = t
        self.beats += 1

    def stats(self, t: float) -> HeartbeatStats:
        age = t - self.last if self.last != float("-inf") else float("inf")
        p = self.period
        if age <= self.cfg.stale_factor * p:
            live = Liveness.ALIVE
        elif age <= self.cfg.dead_factor * p:
            live = Liveness.STALE
        else:
            live = Liveness.DEAD
        if self.intervals:
            iv = np.asarray(self.intervals)
            rate = 1.0 / float(iv.mean()) if iv.mean() > 0 else float("inf")
            jitter = float(iv.std())
        else:
            rate, jitter = 0.0, 0.0
        return HeartbeatStats(self.name, live, age, rate, jitter, self.missed, self.beats, self.out_of_order)


class HeartbeatMonitor:
    def __init__(self, config: Optional[HeartbeatConfig] = None, registry: Optional[FaultRegistry] = None,
                 source: str = "safety.heartbeat") -> None:
        self.config = config or HeartbeatConfig()
        self.registry = registry or DEFAULT_REGISTRY
        self.source = source
        self._sources: dict[str, _Source] = {}

    def register(self, name: str, t: float = 0.0, config: Optional[HeartbeatConfig] = None) -> None:
        cfg = config or self.config
        if cfg.expected_rate_hz <= 0 or not math.isfinite(cfg.expected_rate_hz):
            raise ValueError("expected_rate_hz must be positive")
        self._sources[name] = _Source(name, cfg, t)

    def beat(self, name: str, t: float) -> None:
        if name not in self._sources:
            self.register(name, t)
        self._sources[name].beat(float(t))

    def stats(self, name: str, t: float) -> HeartbeatStats:
        return self._sources[name].stats(t)

    def liveness(self, name: str, t: float) -> Liveness:
        return self.stats(name, t).liveness

    def check(self, t: float) -> list[Fault]:
        faults: list[Fault] = []
        for s in self._sources.values():
            st = s.stats(t)
            src = f"{self.source}/{s.name}"
            if st.liveness == Liveness.DEAD:
                faults.append(self.registry.make("HB_DEAD", src, t, f"{s.name} dead (age {st.age_s:.3f}s)",
                                                 age_s=st.age_s))
                continue
            if st.liveness == Liveness.STALE:
                faults.append(self.registry.make("HB_STALE", src, t, f"{s.name} stale (age {st.age_s:.3f}s)",
                                                 age_s=st.age_s))
            if len(s.intervals) >= s.cfg.min_samples:
                if st.jitter_s > s.cfg.max_jitter_frac * s.period:
                    faults.append(self.registry.make("HB_JITTER", src, t, f"{s.name} jitter {st.jitter_s:.4f}s",
                                                     jitter_s=st.jitter_s))
                if st.rate_hz < s.cfg.min_rate_frac * s.cfg.expected_rate_hz:
                    faults.append(self.registry.make("HB_RATE", src, t, f"{s.name} rate {st.rate_hz:.2f}Hz",
                                                     rate_hz=st.rate_hz))
        return faults

    def health(self, t: float) -> float:
        if not self._sources:
            return 1.0
        w = {Liveness.ALIVE: 1.0, Liveness.STALE: 0.5, Liveness.DEAD: 0.0}
        return sum(w[s.stats(t).liveness] for s in self._sources.values()) / len(self._sources)
