"""Metrics: counters, gauges, histograms (with percentiles), timers.

``MetricsRegistry`` owns named metrics, each optionally labelled
(``registry.counter("bus_msgs_total", topic="/x")``). Histograms keep fixed
Prometheus-style cumulative buckets *and* a bounded reservoir of raw samples
for exact-ish percentiles (linear interpolation, numpy). ``Timer`` is a context
manager that observes elapsed seconds from an injectable time function.
``render_prometheus()`` produces Prometheus text exposition format 0.0.4.

Thread-safe via one lock per registry. TODO(infra): push gateway / OTLP export,
exemplars, summary quantile sketches (t-digest) for unbounded streams.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

DEFAULT_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


@dataclass
class MetricsConfig:
    namespace: str = "cheese"
    reservoir_size: int = 4096
    default_buckets: tuple = DEFAULT_BUCKETS
    seed: int = 0


LabelKey = tuple[tuple[str, str], ...]


def _labels(kw: dict) -> LabelKey:
    return tuple(sorted((k, str(v)) for k, v in kw.items()))


class Counter:
    kind = "counter"

    def __init__(self, lock: threading.Lock) -> None:
        self._lock = lock
        self.value = 0.0

    def inc(self, amount: float = 1.0) -> None:
        if amount < 0:
            raise ValueError("counters only go up")
        with self._lock:
            self.value += amount


class Gauge:
    kind = "gauge"

    def __init__(self, lock: threading.Lock) -> None:
        self._lock = lock
        self.value = 0.0

    def set(self, v: float) -> None:
        with self._lock:
            self.value = float(v)

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self.value += amount

    def dec(self, amount: float = 1.0) -> None:
        self.inc(-amount)


class Histogram:
    kind = "histogram"

    def __init__(self, lock: threading.Lock, buckets: tuple, reservoir_size: int, seed: int) -> None:
        self._lock = lock
        self.buckets = tuple(sorted(buckets))
        self.bucket_counts = [0] * len(self.buckets)
        self.count = 0
        self.sum = 0.0
        self.min = math.inf
        self.max = -math.inf
        self._res: list[float] = []
        self._res_size = reservoir_size
        self._rng = np.random.default_rng(seed)

    def observe(self, v: float) -> None:
        v = float(v)
        with self._lock:
            self.count += 1
            self.sum += v
            self.min = min(self.min, v)
            self.max = max(self.max, v)
            for i, b in enumerate(self.buckets):
                if v <= b:
                    self.bucket_counts[i] += 1
                    break
            # Vitter's algorithm R reservoir sampling.
            if len(self._res) < self._res_size:
                self._res.append(v)
            else:
                j = int(self._rng.integers(0, self.count))
                if j < self._res_size:
                    self._res[j] = v

    def percentile(self, q: float) -> float:
        """q in [0, 100]. NaN if empty."""
        with self._lock:
            if not self._res:
                return float("nan")
            return float(np.percentile(np.asarray(self._res), q))

    def percentiles(self, qs=(50, 90, 99)) -> dict[float, float]:
        return {q: self.percentile(q) for q in qs}

    @property
    def mean(self) -> float:
        return self.sum / self.count if self.count else float("nan")

    def cumulative(self) -> list[tuple[float, int]]:
        out, acc = [], 0
        for b, c in zip(self.buckets, self.bucket_counts):
            acc += c
            out.append((b, acc))
        out.append((math.inf, self.count))
        return out


class Timer:
    """``with registry.timer("plan_seconds"):`` observes elapsed seconds."""

    def __init__(self, hist: Histogram, time_fn: Callable[[], float]) -> None:
        self.hist = hist
        self.time_fn = time_fn
        self.elapsed: Optional[float] = None
        self._t0 = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = self.time_fn()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = self.time_fn() - self._t0
        self.hist.observe(self.elapsed)


@dataclass
class _Family:
    name: str
    kind: str
    help: str
    children: dict = field(default_factory=dict)


class MetricsRegistry:
    def __init__(self, config: Optional[MetricsConfig] = None,
                 time_fn: Callable[[], float] = time.perf_counter) -> None:
        # perf_counter is fine here: it measures compute latency, not stack time.
        self.config = config or MetricsConfig()
        self.time_fn = time_fn
        self._lock = threading.Lock()
        self._families: dict[str, _Family] = {}

    def _full(self, name: str) -> str:
        ns = self.config.namespace
        return f"{ns}_{name}" if ns else name

    def _get(self, name: str, kind: str, help: str, labels: dict, factory):
        full = self._full(name)
        with self._lock:
            fam = self._families.get(full)
            if fam is None:
                fam = self._families[full] = _Family(full, kind, help)
            elif fam.kind != kind:
                raise TypeError(f"metric {full} already registered as {fam.kind}")
            key = _labels(labels)
            if key not in fam.children:
                fam.children[key] = factory()
            return fam.children[key]

    def counter(self, name: str, help: str = "", **labels) -> Counter:
        return self._get(name, "counter", help, labels, lambda: Counter(self._lock))

    def gauge(self, name: str, help: str = "", **labels) -> Gauge:
        return self._get(name, "gauge", help, labels, lambda: Gauge(self._lock))

    def histogram(self, name: str, help: str = "", buckets: Optional[tuple] = None, **labels) -> Histogram:
        b = buckets or self.config.default_buckets
        return self._get(name, "histogram", help, labels,
                         lambda: Histogram(threading.Lock(), b, self.config.reservoir_size, self.config.seed))

    def timer(self, name: str, help: str = "", **labels) -> Timer:
        return Timer(self.histogram(name, help, **labels), self.time_fn)

    def snapshot(self) -> dict:
        """JSON-friendly dict of all metric values (used by monitoring)."""
        out: dict = {}
        for fam in list(self._families.values()):
            for key, m in fam.children.items():
                lbl = ",".join(f"{k}={v}" for k, v in key)
                name = f"{fam.name}{{{lbl}}}" if lbl else fam.name
                if fam.kind == "histogram":
                    out[name] = {"count": m.count, "sum": m.sum, "mean": m.mean,
                                 "p50": m.percentile(50), "p90": m.percentile(90), "p99": m.percentile(99)}
                else:
                    out[name] = m.value
        return out

    def render_prometheus(self) -> str:
        lines: list[str] = []
        for fam in sorted(self._families.values(), key=lambda f: f.name):
            if fam.help:
                lines.append(f"# HELP {fam.name} {_esc_help(fam.help)}")
            lines.append(f"# TYPE {fam.name} {fam.kind}")
            for key, m in sorted(fam.children.items()):
                if fam.kind == "histogram":
                    for le, c in m.cumulative():
                        lines.append(f"{fam.name}_bucket{_fmt_labels(key + (('le', _fmt_num(le)),))} {c}")
                    lines.append(f"{fam.name}_sum{_fmt_labels(key)} {_fmt_num(m.sum)}")
                    lines.append(f"{fam.name}_count{_fmt_labels(key)} {m.count}")
                else:
                    lines.append(f"{fam.name}{_fmt_labels(key)} {_fmt_num(m.value)}")
        return "\n".join(lines) + "\n"


def _fmt_num(v: float) -> str:
    if v == math.inf:
        return "+Inf"
    if v == -math.inf:
        return "-Inf"
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return str(int(v)) if not isinstance(v, bool) else str(v)
    return repr(float(v))


def _esc(v: str) -> str:
    return v.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _esc_help(v: str) -> str:
    return v.replace("\\", "\\\\").replace("\n", "\\n")


def _fmt_labels(key: LabelKey) -> str:
    if not key:
        return ""
    return "{" + ",".join(f'{k}="{_esc(v)}"' for k, v in key) + "}"
