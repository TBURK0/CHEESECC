"""Redundancy checks: 2-out-of-3 voting and dual-channel comparison.

`vote_2oo3(a, b, c, tol)`: channels agree pairwise when |x_i - x_j| <= tol
(element-wise for vectors; all elements must agree). Non-finite channels
never agree with anything.

  * all 3 agree                    -> median, ok
  * a channel agrees with no one   -> it is the outlier; result is the mean of
                                      the agreeing pair, ok but degraded (RED_DISAGREE)
  * chain (a~b, b~c, a!~c)         -> median (b is consistent with both), ok, degraded
  * fewer than 2 agreeing channels -> no majority; value None (RED_NO_MAJORITY)

`DualChannelComparator`: two independent channels (e.g. main and monitor
computation of the same quantity). Disagreement beyond abs+rel tolerance must
persist for `debounce_s` before a fault is raised (filters skew between
asynchronously sampled channels). On disagreement the comparator can return
the conservative channel (`select="min"` or `"max"`) - e.g. min of two
speed-limit estimates, max of two braking-distance estimates.

# TODO(safety): time-aligned comparison with interpolation, diagnostic coverage
# metrics, latent-fault detection (channels stuck together).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np

from cheesestack.core import Fault

from .faults import DEFAULT_REGISTRY, FaultRegistry

Value = Union[float, np.ndarray]


@dataclass
class VoteResult:
    value: Optional[Value]
    ok: bool  # a majority exists and `value` may be used
    degraded: bool  # not all channels agreed
    faulty: list[int] = field(default_factory=list)  # indices of outlier channels
    agree: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=bool))


def _agree(x: np.ndarray, y: np.ndarray, tol: np.ndarray) -> bool:
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return False
    return bool(np.all(np.abs(x - y) <= tol))


def vote_2oo3(a: Value, b: Value, c: Value, tol: Union[float, Sequence[float]]) -> VoteResult:
    xs = [np.atleast_1d(np.asarray(v, dtype=float)) for v in (a, b, c)]
    if not (xs[0].shape == xs[1].shape == xs[2].shape):
        raise ValueError("2oo3 channels must have the same shape")
    tol_arr = np.broadcast_to(np.asarray(tol, dtype=float), xs[0].shape)
    scalar = np.ndim(a) == 0
    m = np.zeros((3, 3), dtype=bool)
    for i in range(3):
        for j in range(i + 1, 3):
            m[i, j] = m[j, i] = _agree(xs[i], xs[j], tol_arr)
    n_agree = m.sum(axis=1)
    pairs = int(m[np.triu_indices(3, 1)].sum())

    def out(v: np.ndarray) -> Value:
        return float(v[0]) if scalar else v

    if pairs == 3:
        return VoteResult(out(np.median(np.stack(xs), axis=0)), True, False, [], m)
    if pairs == 2:
        # Chain: the middle channel agrees with both ends (so all three are finite).
        # The ends disagree, so no single outlier can be blamed; use the median.
        return VoteResult(out(np.median(np.stack(xs), axis=0)), True, True, [], m)
    if pairs == 1:
        i, j = [k for k in range(3) if n_agree[k] == 1]
        bad = [k for k in range(3) if n_agree[k] == 0]
        return VoteResult(out((xs[i] + xs[j]) / 2.0), True, True, bad, m)
    return VoteResult(None, False, True, [0, 1, 2], m)


@dataclass
class VoterConfig:
    tol: Union[float, tuple] = 0.1
    name: str = "voter"


class TripleVoter:
    """Stateful 2oo3 voter that emits faults and remembers the last good value."""

    def __init__(self, config: Optional[VoterConfig] = None, registry: Optional[FaultRegistry] = None,
                 source: str = "safety.redundancy") -> None:
        self.config = config or VoterConfig()
        self.registry = registry or DEFAULT_REGISTRY
        self.source = f"{source}/{self.config.name}"
        self.last_good: Optional[Value] = None

    def update(self, a: Value, b: Value, c: Value, t: float) -> tuple[VoteResult, list[Fault]]:
        r = vote_2oo3(a, b, c, self.config.tol)
        faults = []
        if not r.ok:
            faults.append(self.registry.make("RED_NO_MAJORITY", self.source, t, "no 2oo3 majority"))
        elif r.degraded:
            faults.append(self.registry.make("RED_DISAGREE", self.source, t,
                                             f"channel(s) {r.faulty} disagree", faulty=list(r.faulty)))
        if r.ok:
            self.last_good = r.value
        return r, faults


@dataclass
class DualChannelConfig:
    abs_tol: float = 0.1
    rel_tol: float = 0.0
    debounce_s: float = 0.1
    select: Optional[str] = None  # None -> channel A, "min", "max", or "mean" when agreeing
    name: str = "dual"


@dataclass
class DualResult:
    value: Optional[Value]
    agree: bool
    fault_active: bool
    diff: float


class DualChannelComparator:
    def __init__(self, config: Optional[DualChannelConfig] = None, registry: Optional[FaultRegistry] = None,
                 source: str = "safety.redundancy") -> None:
        self.config = config or DualChannelConfig()
        self.registry = registry or DEFAULT_REGISTRY
        self.source = f"{source}/{self.config.name}"
        self._disagree_since: Optional[float] = None

    def _select(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        s = self.config.select
        if s == "min":
            return np.minimum(a, b)
        if s == "max":
            return np.maximum(a, b)
        if s == "mean":
            return (a + b) / 2.0
        return a

    def update(self, a: Value, b: Value, t: float) -> tuple[DualResult, list[Fault]]:
        c = self.config
        scalar = np.ndim(a) == 0
        xa = np.atleast_1d(np.asarray(a, dtype=float))
        xb = np.atleast_1d(np.asarray(b, dtype=float))
        fa, fb = bool(np.all(np.isfinite(xa))), bool(np.all(np.isfinite(xb)))
        if fa and fb:
            diff = float(np.max(np.abs(xa - xb)))
            tol = c.abs_tol + c.rel_tol * float(np.max(np.maximum(np.abs(xa), np.abs(xb))))
            agree = diff <= tol
        else:
            diff, agree = float("inf"), False

        if agree:
            self._disagree_since = None
        elif self._disagree_since is None:
            self._disagree_since = t
        fault_active = (not agree) and (t - self._disagree_since >= c.debounce_s)

        if fa and fb:
            v = self._select(xa, xb)
        elif fa:
            v = xa
        elif fb:
            v = xb
        else:
            v = None
        if not agree and c.select is None and fa and fb:
            v = xa  # no conservative rule configured: trust primary but raise fault after debounce
        value = None if v is None else (float(v[0]) if scalar else v)
        faults = []
        if fault_active:
            faults.append(self.registry.make("RED_DISAGREE", self.source, t, f"dual-channel diff {diff:.4g}",
                                             diff=diff))
        if not fa and not fb:
            faults.append(self.registry.make("RED_NO_MAJORITY", self.source, t, "both channels non-finite"))
        return DualResult(value, agree, fault_active, diff), faults
