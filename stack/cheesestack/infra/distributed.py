"""Distributed processing: a retrying task queue and a map-reduce helper.

``TaskQueue`` wraps ``concurrent.futures`` (threads by default, processes
optional) and resubmits failed tasks up to ``max_retries`` times with
exponential backoff (backoff sleeps happen in the worker, via an injectable
``sleep`` so tests run instantly). Each submission returns a ``TaskHandle``
with the final result/exception and the attempt count.

``map_reduce(shards, mapper, reducer, initial)`` maps a function over log
shards (e.g. segment files from ``storage.RecordStore``) in parallel and folds
results in *shard order*, so the output is deterministic regardless of worker
scheduling.

Limits: single host. TODO(infra): Ray/Dask/K8s-job backend behind the same
interface, task priorities, cancellation of running tasks, dead-letter queue.
"""

from __future__ import annotations

import concurrent.futures as cf
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence


@dataclass
class TaskQueueConfig:
    max_workers: int = 4
    max_retries: int = 2
    backoff_s: float = 0.05
    backoff_factor: float = 2.0
    use_processes: bool = False  # tasks must be picklable when True
    retry_on: tuple = (Exception,)


@dataclass
class TaskResult:
    ok: bool
    value: Any = None
    error: Optional[BaseException] = None
    attempts: int = 0
    errors: list = field(default_factory=list)


def _run_with_retries(fn, args, kwargs, max_retries, backoff_s, factor, retry_on, sleep) -> TaskResult:
    errors: list = []
    delay = backoff_s
    for attempt in range(1, max_retries + 2):
        try:
            return TaskResult(True, fn(*args, **kwargs), None, attempt, errors)
        except retry_on as e:  # type: ignore[misc]
            errors.append(repr(e))
            if attempt > max_retries:
                return TaskResult(False, None, e, attempt, errors)
            if delay > 0:
                sleep(delay)
            delay *= factor
        except BaseException as e:  # non-retryable
            errors.append(repr(e))
            return TaskResult(False, None, e, attempt, errors)
    raise AssertionError("unreachable")  # pragma: no cover


class TaskHandle:
    def __init__(self, future: cf.Future, name: str) -> None:
        self.future = future
        self.name = name

    def result(self, timeout: Optional[float] = None) -> TaskResult:
        return self.future.result(timeout)

    def done(self) -> bool:
        return self.future.done()


class TaskQueue:
    def __init__(self, config: Optional[TaskQueueConfig] = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.config = config or TaskQueueConfig()
        self._sleep = sleep
        ex = cf.ProcessPoolExecutor if self.config.use_processes else cf.ThreadPoolExecutor
        self._ex = ex(max_workers=self.config.max_workers)
        self.handles: list[TaskHandle] = []

    def submit(self, fn: Callable, *args, name: str = "", **kwargs) -> TaskHandle:
        c = self.config
        sleep = time.sleep if c.use_processes else self._sleep
        fut = self._ex.submit(_run_with_retries, fn, args, kwargs, c.max_retries,
                              c.backoff_s, c.backoff_factor, c.retry_on, sleep)
        h = TaskHandle(fut, name or getattr(fn, "__name__", "task"))
        self.handles.append(h)
        return h

    def map(self, fn: Callable, items: Iterable) -> list[TaskResult]:
        hs = [self.submit(fn, it) for it in items]
        return [h.result() for h in hs]

    def wait_all(self, timeout: Optional[float] = None) -> list[TaskResult]:
        return [h.result(timeout) for h in self.handles]

    def stats(self) -> dict:
        done = [h.result() for h in self.handles if h.done()]
        return {"submitted": len(self.handles), "done": len(done),
                "ok": sum(r.ok for r in done), "failed": sum(not r.ok for r in done),
                "retries": sum(max(0, r.attempts - 1) for r in done)}

    def shutdown(self, wait: bool = True) -> None:
        self._ex.shutdown(wait=wait)

    def __enter__(self) -> "TaskQueue":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()


class MapReduceError(RuntimeError):
    def __init__(self, failures: dict) -> None:
        super().__init__(f"{len(failures)} shard(s) failed: {sorted(failures)}")
        self.failures = failures


def map_reduce(shards: Sequence[Any], mapper: Callable[[Any], Any],
               reducer: Callable[[Any, Any], Any], initial: Any,
               config: Optional[TaskQueueConfig] = None, skip_failed: bool = False,
               sleep: Callable[[float], None] = time.sleep) -> Any:
    """Parallel map over shards, deterministic left fold in shard order.

    Raises ``MapReduceError`` if any shard fails after retries, unless
    ``skip_failed`` (then failed shards are omitted from the fold).
    """
    with TaskQueue(config, sleep=sleep) as q:
        results = q.map(mapper, shards)
    failures = {i: r.error for i, r in enumerate(results) if not r.ok}
    if failures and not skip_failed:
        raise MapReduceError(failures)
    acc = initial
    for r in results:
        if r.ok:
            acc = reducer(acc, r.value)
    return acc
