"""Sensor plausibility checking: range, rate-of-change, stuck-value, timeout, cross-sensor.

`PlausibilityChecker` holds a `SignalSpec` per named signal and evaluates each
new sample against it. Cross-sensor consistency is expressed as
`CrossCheck(a, b, predict, tol)`: the checker predicts signal `b` from the latest
value of `a` via `predict` and flags |b - predict(a)| > tol.
Each check result can be fed straight into a `DiagnosticManager` (code per
signal+check) for debouncing; this module itself is stateless per check except
for the history needed for rate and stuck detection.
"""

from __future__ import annotations

import enum
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional


class CheckType(enum.Enum):
    RANGE = "range"
    RATE = "rate"
    STUCK = "stuck"
    TIMEOUT = "timeout"
    NAN = "nan"
    CROSS = "cross"


@dataclass
class SignalSpec:
    min_value: float = -math.inf
    max_value: float = math.inf
    max_rate: float = math.inf  # units/s
    stuck_window_s: float = 0.0  # 0 disables stuck detection
    stuck_tolerance: float = 1e-6  # peak-to-peak below this over window => stuck
    stuck_min_abs: float = 0.0  # only flag stuck when |value| >= this (e.g. 0 m/s is legit when parked)
    timeout_s: float = math.inf


@dataclass
class CrossCheck:
    name: str
    signal_a: str
    signal_b: str
    predict: Callable[[float], float]  # b_hat = predict(a)
    tol: float
    max_age_s: float = 0.2  # both samples must be this fresh


@dataclass
class CheckResult:
    signal: str
    check: CheckType
    passed: bool
    value: float = float("nan")
    detail: str = ""

    @property
    def code(self) -> str:
        return f"PLAUS_{self.signal.upper()}_{self.check.value.upper()}"


@dataclass
class PlausibilityConfig:
    history_len: int = 512


class PlausibilityChecker:
    def __init__(self, specs: Optional[dict[str, SignalSpec]] = None, cross_checks: Optional[list[CrossCheck]] = None,
                 config: Optional[PlausibilityConfig] = None) -> None:
        self.config = config or PlausibilityConfig()
        self.specs: dict[str, SignalSpec] = dict(specs or {})
        self.cross_checks: list[CrossCheck] = list(cross_checks or [])
        self._hist: dict[str, deque] = {}

    def add_signal(self, name: str, spec: SignalSpec) -> None:
        self.specs[name] = spec

    def add_cross_check(self, cc: CrossCheck) -> None:
        self.cross_checks.append(cc)

    def latest(self, name: str) -> Optional[tuple[float, float]]:
        h = self._hist.get(name)
        return h[-1] if h else None

    def check(self, name: str, t: float, value: float) -> list[CheckResult]:
        """Evaluate one sample. Implausible samples (NaN / out of range) are not stored as history."""
        spec = self.specs.get(name, SignalSpec())
        h = self._hist.setdefault(name, deque(maxlen=self.config.history_len))
        res: list[CheckResult] = []
        if value is None or not math.isfinite(value):
            return [CheckResult(name, CheckType.NAN, False, float("nan"), "non-finite value")]
        in_range = spec.min_value <= value <= spec.max_value
        res.append(CheckResult(name, CheckType.RANGE, in_range, value,
                               "" if in_range else f"outside [{spec.min_value}, {spec.max_value}]"))
        if h and math.isfinite(spec.max_rate):
            t0, v0 = h[-1]
            if t > t0:
                rate = (value - v0) / (t - t0)
                ok = abs(rate) <= spec.max_rate
                res.append(CheckResult(name, CheckType.RATE, ok, rate, "" if ok else f"rate {rate:.3g} > {spec.max_rate}"))
        if not in_range:
            return res
        h.append((t, value))
        if spec.stuck_window_s > 0 and h[0][0] <= t - spec.stuck_window_s:
            win = [v for (ti, v) in h if ti >= t - spec.stuck_window_s]
            ptp = max(win) - min(win)
            stuck = ptp < spec.stuck_tolerance and abs(value) >= spec.stuck_min_abs and len(win) >= 3
            res.append(CheckResult(name, CheckType.STUCK, not stuck, ptp, "stuck" if stuck else ""))
        return res

    def check_timeouts(self, t: float) -> list[CheckResult]:
        out = []
        for name, spec in self.specs.items():
            if not math.isfinite(spec.timeout_s):
                continue
            last = self.latest(name)
            age = math.inf if last is None else t - last[0]
            out.append(CheckResult(name, CheckType.TIMEOUT, age <= spec.timeout_s, age,
                                   "" if age <= spec.timeout_s else f"no data for {age:.2f}s"))
        return out

    def check_cross(self, t: float) -> list[CheckResult]:
        out = []
        for cc in self.cross_checks:
            a, b = self.latest(cc.signal_a), self.latest(cc.signal_b)
            if a is None or b is None or t - a[0] > cc.max_age_s or t - b[0] > cc.max_age_s:
                continue  # not enough fresh data: no verdict
            err = b[1] - cc.predict(a[1])
            ok = abs(err) <= cc.tol
            out.append(CheckResult(cc.name, CheckType.CROSS, ok, err, "" if ok else f"{cc.signal_b} off by {err:.3g}"))
        return out

    def check_all(self, t: float, samples: dict[str, float]) -> list[CheckResult]:
        """Convenience: per-signal checks for `samples`, then timeouts and cross checks."""
        out = []
        for name, v in samples.items():
            out.extend(self.check(name, t, v))
        out.extend(self.check_timeouts(t))
        out.extend(self.check_cross(t))
        # TODO(vehicle): noise-floor (variance) checks, frozen-timestamp detection, signal-quality bits.
        return out


def default_vehicle_specs() -> dict[str, SignalSpec]:
    """Reasonable specs for the phone-rig / CAN signals used by the stack."""
    return {
        "speed": SignalSpec(-5.0, 70.0, max_rate=15.0, timeout_s=0.5),
        "yaw_rate": SignalSpec(-2.5, 2.5, max_rate=10.0, stuck_window_s=2.0, stuck_tolerance=1e-5, stuck_min_abs=1e-3, timeout_s=0.2),
        "lat_accel": SignalSpec(-15.0, 15.0, max_rate=200.0, timeout_s=0.2),
        "long_accel": SignalSpec(-15.0, 15.0, max_rate=200.0, timeout_s=0.2),
        "handwheel": SignalSpec(-math.radians(600), math.radians(600), max_rate=math.radians(1000), timeout_s=0.2),
    }
