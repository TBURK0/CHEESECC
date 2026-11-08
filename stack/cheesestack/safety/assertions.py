"""Runtime assertions that raise in test mode and record faults in production.

    sa = SafetyAssertions(mode=AssertMode.TEST)       # raises SafetyAssertionError
    sa = SafetyAssertions(mode=AssertMode.PRODUCTION, sink=fault_table.report)

    sa.check(v >= 0, "speed negative", source="control", stamp=t)
    sa.finite(cmd.throttle, "throttle", source="control", stamp=t)
    x = sa.in_range(steer, -1, 1, "steer", ...)  # returns the clamped value in production

Every helper returns a bool (or the sanitized value) so production code can
take a safe branch on failure. `guard()` is a context manager that converts any
exception in its body into a fault (production) or re-raises (test).

Never use Python's `assert` for safety logic: it vanishes under `python -O`.
# TODO(safety): rate-limit identical assertion faults; compile-time disabling
# list reviewed per release.
"""

from __future__ import annotations

import contextlib
import enum
import math
from typing import Any, Callable, Iterator, Optional

import numpy as np

from cheesestack.core import Fault, Severity

from .faults import DEFAULT_REGISTRY, FaultRegistry


class AssertMode(enum.Enum):
    TEST = "test"
    PRODUCTION = "production"


class SafetyAssertionError(AssertionError):
    def __init__(self, fault: Fault) -> None:
        super().__init__(f"[{fault.code}] {fault.source}: {fault.message}")
        self.fault = fault


class SafetyAssertions:
    def __init__(self, mode: AssertMode = AssertMode.PRODUCTION, sink: Optional[Callable[[Fault], Any]] = None,
                 registry: Optional[FaultRegistry] = None, code: str = "AST_FAILED") -> None:
        self.mode = mode
        self.sink = sink
        self.registry = registry or DEFAULT_REGISTRY
        self.code = code
        self.recorded: list[Fault] = []
        self.max_recorded = 1000

    def _fail(self, message: str, source: str, stamp: float, severity: Optional[Severity], **data: Any) -> None:
        f = self.registry.make(self.code, source, stamp, message, severity, **data)
        if self.mode == AssertMode.TEST:
            raise SafetyAssertionError(f)
        if len(self.recorded) < self.max_recorded:
            self.recorded.append(f)
        if self.sink is not None:
            self.sink(f)

    def check(self, cond: Any, message: str, source: str = "unknown", stamp: float = 0.0,
              severity: Optional[Severity] = None, **data: Any) -> bool:
        ok = bool(cond)
        if not ok:
            self._fail(message, source, stamp, severity, **data)
        return ok

    def finite(self, value: Any, name: str, source: str = "unknown", stamp: float = 0.0,
               severity: Optional[Severity] = None) -> bool:
        try:
            ok = bool(np.all(np.isfinite(np.asarray(value, dtype=float))))
        except (TypeError, ValueError):
            ok = False
        return self.check(ok, f"{name} not finite: {value!r}", source, stamp, severity)

    def in_range(self, value: float, lo: float, hi: float, name: str, source: str = "unknown",
                 stamp: float = 0.0, severity: Optional[Severity] = None) -> float:
        """Returns value clamped to [lo, hi]. Non-finite values are returned as `lo` if lo is
        finite, else 0.0 - callers needing a different safe value should check `finite` first."""
        v = float(value)
        if not math.isfinite(v):
            self._fail(f"{name} not finite", source, stamp, severity)
            return lo if math.isfinite(lo) else 0.0
        if v < lo or v > hi:
            self._fail(f"{name}={v} outside [{lo}, {hi}]", source, stamp, severity, value=v)
            return min(max(v, lo), hi)
        return v

    @contextlib.contextmanager
    def guard(self, source: str = "unknown", stamp: float = 0.0, severity: Optional[Severity] = None) -> Iterator[None]:
        try:
            yield
        except SafetyAssertionError:
            raise
        except Exception as e:
            if self.mode == AssertMode.TEST:
                raise
            self._fail(f"exception in guarded block: {type(e).__name__}: {e}", source, stamp, severity,
                       exception=type(e).__name__)
