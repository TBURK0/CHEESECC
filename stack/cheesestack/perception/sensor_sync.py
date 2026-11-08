"""Sensor synchronization: time-indexed stream buffers and multi-stream matching.

- `StreamBuffer`: bounded, time-sorted history for one stream with nearest-neighbour
  lookup and linear interpolation (numeric payloads: float, np.ndarray, or dataclass
  fields via a user `interp_fn`).
- `match_nearest` / `match_interpolated`: align a set of streams to a reference stamp
  within a tolerance.
- `ApproximateTimeSynchronizer`: ROS-style approximate-time policy. Callers `add()`
  messages per stream; whenever one message from every stream lies within `slop`
  seconds of each other the set is emitted (callback and/or returned) and older
  messages are discarded.

Limits: the approximate policy is greedy (pivot = newest queue head), not the
optimal-window search of ROS message_filters. Interpolation is linear and does not
wrap angles unless `angle_indices` is set.

# TODO(perception): per-sensor clock offset/skew estimation (e.g. from IMU-camera
#   cross-correlation), rolling-shutter line-time compensation, SLERP for rotations.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from cheesestack.core import wrap_angle


def lerp_payload(a: Any, b: Any, alpha: float, angle_indices: Sequence[int] = ()) -> Any:
    """Linear interpolation of numeric payloads (scalars or arrays). Indices in
    `angle_indices` (flat) are interpolated along the shortest arc."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) + alpha * (float(b) - float(a))
    aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    out = aa + alpha * (bb - aa)
    if angle_indices:
        fa, fb, fo = aa.ravel(), bb.ravel(), out.ravel()
        for i in angle_indices:
            fo[i] = wrap_angle(fa[i] + alpha * wrap_angle(fb[i] - fa[i]))
        out = fo.reshape(aa.shape)
    return out


@dataclass
class StreamBufferConfig:
    capacity: int = 200
    max_age: float = 2.0  # seconds kept behind newest stamp
    max_interp_gap: float = 0.25  # refuse to interpolate across larger gaps


class StreamBuffer:
    """Time-sorted (stamp, msg) buffer for one stream. Out-of-order inserts are sorted in."""

    def __init__(self, name: str, config: Optional[StreamBufferConfig] = None,
                 interp_fn: Optional[Callable[[Any, Any, float], Any]] = None) -> None:
        self.name = name
        self.config = config or StreamBufferConfig()
        self.interp_fn = interp_fn or (lambda a, b, al: lerp_payload(a, b, al))
        self._t: list[float] = []
        self._m: list[Any] = []

    def add(self, stamp: float, msg: Any) -> None:
        i = bisect.bisect_right(self._t, stamp)
        self._t.insert(i, float(stamp))
        self._m.insert(i, msg)
        self._prune()

    def _prune(self) -> None:
        c = self.config
        newest = self._t[-1]
        k = 0
        while k < len(self._t) and (newest - self._t[k] > c.max_age or len(self._t) - k > c.capacity):
            k += 1
        if k:
            del self._t[:k], self._m[:k]

    def __len__(self) -> int:
        return len(self._t)

    @property
    def stamps(self) -> list[float]:
        return list(self._t)

    def nearest(self, stamp: float, tolerance: float = math.inf) -> Optional[tuple[float, Any]]:
        if not self._t:
            return None
        i = bisect.bisect_left(self._t, stamp)
        cands = [j for j in (i - 1, i) if 0 <= j < len(self._t)]
        j = min(cands, key=lambda j: abs(self._t[j] - stamp))
        if abs(self._t[j] - stamp) > tolerance:
            return None
        return self._t[j], self._m[j]

    def interpolate(self, stamp: float) -> Optional[Any]:
        """Linearly interpolate at `stamp`. Exact hits return the message itself.
        Returns None outside the buffered span or across a gap > max_interp_gap."""
        if not self._t or stamp < self._t[0] - 1e-12 or stamp > self._t[-1] + 1e-12:
            return None
        i = bisect.bisect_left(self._t, stamp)
        if i < len(self._t) and abs(self._t[i] - stamp) < 1e-12:
            return self._m[i]
        if i == 0:
            return self._m[0]
        t0, t1 = self._t[i - 1], self._t[i]
        if t1 - t0 > self.config.max_interp_gap:
            return None
        return self.interp_fn(self._m[i - 1], self._m[i], (stamp - t0) / (t1 - t0))

    def pop_until(self, stamp: float) -> None:
        """Drop all entries with stamp <= `stamp`."""
        k = bisect.bisect_right(self._t, stamp)
        del self._t[:k], self._m[:k]

    def head(self) -> Optional[tuple[float, Any]]:
        return (self._t[0], self._m[0]) if self._t else None


def match_nearest(buffers: dict[str, StreamBuffer], stamp: float, tolerance: float) -> Optional[dict[str, tuple[float, Any]]]:
    """Nearest message from each buffer within `tolerance` of `stamp`, or None if any stream misses."""
    out = {}
    for name, b in buffers.items():
        r = b.nearest(stamp, tolerance)
        if r is None:
            return None
        out[name] = r
    return out


def match_interpolated(buffers: dict[str, StreamBuffer], stamp: float) -> Optional[dict[str, Any]]:
    """Interpolated payload of every buffer at `stamp`, or None if any can't be interpolated."""
    out = {}
    for name, b in buffers.items():
        r = b.interpolate(stamp)
        if r is None:
            return None
        out[name] = r
    return out


@dataclass
class SyncedSet:
    stamp: float  # pivot stamp
    messages: dict[str, tuple[float, Any]]
    spread: float  # max - min stamp in the set

    def __getitem__(self, name: str) -> Any:
        return self.messages[name][1]


@dataclass
class ApproximateTimeConfig:
    slop: float = 0.02  # max allowed spread within a set (s)
    queue_size: int = 30


class ApproximateTimeSynchronizer:
    """Greedy approximate-time synchronizer over N named streams.

    Algorithm (per `add`): while every queue is non-empty, take pivot = the newest
    queue head. For each stream choose the message closest to the pivot. If the
    set's spread <= slop, emit and discard every message up to the chosen ones;
    otherwise discard the oldest head (it can never match a later pivot better).
    """

    def __init__(self, streams: Sequence[str], config: Optional[ApproximateTimeConfig] = None,
                 callback: Optional[Callable[[SyncedSet], None]] = None) -> None:
        if len(streams) < 1:
            raise ValueError("need at least one stream")
        self.config = config or ApproximateTimeConfig()
        self.callback = callback
        cfg = StreamBufferConfig(capacity=self.config.queue_size, max_age=math.inf)
        self.buffers = {s: StreamBuffer(s, cfg) for s in streams}
        self.emitted = 0
        self.discarded = 0

    def add(self, stream: str, stamp: float, msg: Any) -> list[SyncedSet]:
        before = len(self.buffers[stream])
        self.buffers[stream].add(stamp, msg)
        if len(self.buffers[stream]) <= before:
            self.discarded += 1  # capacity eviction
        return self._process()

    def _process(self) -> list[SyncedSet]:
        out: list[SyncedSet] = []
        slop = self.config.slop
        while all(len(b) for b in self.buffers.values()):
            heads = {n: b.head()[0] for n, b in self.buffers.items()}
            pivot = max(heads.values())
            chosen = {n: b.nearest(pivot) for n, b in self.buffers.items()}
            ts = [c[0] for c in chosen.values()]
            spread = max(ts) - min(ts)
            if spread <= slop + 1e-12:
                s = SyncedSet(pivot, chosen, spread)
                for n, b in self.buffers.items():
                    b.pop_until(chosen[n][0])
                self.emitted += 1
                out.append(s)
                if self.callback:
                    self.callback(s)
            else:
                oldest = min(heads, key=heads.get)
                self.buffers[oldest].pop_until(heads[oldest])
                self.discarded += 1
        return out
