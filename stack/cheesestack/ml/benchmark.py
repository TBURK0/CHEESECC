"""Model benchmarking: latency / throughput harness and size / memory reports.

``benchmark(fn, inputs)`` calls ``fn`` on each input with warmup iterations
discarded, then reports mean / std / p50 / p90 / p99 / max latency and
throughput (samples/s using ``batch_size``). Python heap peak during timed calls
is measured with ``tracemalloc`` (numpy buffers are tracked; native allocations
of external runtimes are not).

The timer is injectable (default ``time.perf_counter``); benchmarking is the one
place that legitimately measures wall time, and tests pass a fake timer.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, asdict
from typing import Any, Callable, Optional, Sequence

import numpy as np


@dataclass
class BenchmarkConfig:
    warmup: int = 5
    iterations: int = 50
    batch_size: int = 1  # samples per call, for throughput
    track_memory: bool = True


@dataclass
class BenchmarkResult:
    name: str
    iterations: int
    mean_ms: float
    std_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    max_ms: float
    throughput_per_s: float
    peak_mem_bytes: int

    def as_dict(self) -> dict:
        return asdict(self)

    def meets(self, p99_budget_ms: float) -> bool:
        return self.p99_ms <= p99_budget_ms


def benchmark(fn: Callable[[Any], Any], inputs: Sequence[Any] | Any, config: Optional[BenchmarkConfig] = None,
              name: str = "model", timer: Callable[[], float] = time.perf_counter) -> BenchmarkResult:
    """Time ``fn(x)`` over inputs (cycled if fewer than warmup + iterations)."""
    cfg = config or BenchmarkConfig()
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    n = len(inputs)
    for i in range(cfg.warmup):
        fn(inputs[i % n])
    if cfg.track_memory:
        tracemalloc.start()
        tracemalloc.reset_peak()
    lat = np.empty(cfg.iterations)
    try:
        for i in range(cfg.iterations):
            x = inputs[(cfg.warmup + i) % n]
            t0 = timer()
            fn(x)
            lat[i] = timer() - t0
        peak = tracemalloc.get_traced_memory()[1] if cfg.track_memory else 0
    finally:
        if cfg.track_memory:
            tracemalloc.stop()
    ms = lat * 1e3
    total = lat.sum()
    return BenchmarkResult(
        name, cfg.iterations, float(ms.mean()), float(ms.std()),
        *(float(np.percentile(ms, q)) for q in (50, 90, 99)), float(ms.max()),
        float(cfg.iterations * cfg.batch_size / total) if total > 0 else float("inf"), int(peak),
    )


@dataclass
class ModelSizeReport:
    num_params: int
    size_bytes: int
    per_tensor: dict

    def size_mb(self) -> float:
        return self.size_bytes / 2 ** 20


def model_size(state: dict[str, np.ndarray], dtype: Optional[np.dtype] = None) -> ModelSizeReport:
    """Parameter count and storage size; pass ``dtype`` (e.g. float16/int8) to estimate
    a quantised export."""
    per, n, size = {}, 0, 0
    for k, v in state.items():
        a = np.asarray(v)
        itemsize = np.dtype(dtype).itemsize if dtype is not None else a.itemsize
        per[k] = {"shape": tuple(a.shape), "params": int(a.size), "bytes": int(a.size * itemsize)}
        n += a.size
        size += a.size * itemsize
    return ModelSizeReport(int(n), int(size), per)


def compare(results: Sequence[BenchmarkResult]) -> str:
    """Plain-text table for logs / CI."""
    hdr = f"{'name':<20}{'p50 ms':>10}{'p99 ms':>10}{'thr/s':>12}{'peak KB':>10}"
    rows = [hdr] + [f"{r.name:<20}{r.p50_ms:>10.3f}{r.p99_ms:>10.3f}{r.throughput_per_s:>12.1f}"
                    f"{r.peak_mem_bytes / 1024:>10.1f}" for r in results]
    # TODO(ml): on-device runs (CoreML on iPhone ANE/GPU) reported back through comms.
    return "\n".join(rows)
