"""Hyperparameter tuning: grid search, random search, successive halving.

An *objective* is a callable ``objective(params) -> score`` (grid / random) or
``objective(params, budget) -> score`` (successive halving, where ``budget`` is
e.g. epochs). ``mode`` says whether higher (``"max"``) or lower is better.
Search spaces are dicts of name -> list (grid) or a ``Distribution`` (random).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np


class Distribution:
    def sample(self, rng: np.random.Generator) -> Any:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class Uniform(Distribution):
    low: float
    high: float

    def sample(self, rng):
        return float(rng.uniform(self.low, self.high))


@dataclass
class LogUniform(Distribution):
    low: float
    high: float

    def sample(self, rng):
        return float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))


@dataclass
class IntUniform(Distribution):
    low: int
    high: int  # inclusive

    def sample(self, rng):
        return int(rng.integers(self.low, self.high + 1))


@dataclass
class Choice(Distribution):
    options: Sequence[Any]

    def sample(self, rng):
        return self.options[int(rng.integers(len(self.options)))]


@dataclass
class Trial:
    params: dict
    score: float
    budget: Optional[float] = None


@dataclass
class TuningResult:
    trials: list[Trial] = field(default_factory=list)
    mode: str = "max"
    final_budget: Optional[float] = None  # if set, only trials at this budget compete

    @property
    def best(self) -> Trial:
        key = (lambda t: t.score) if self.mode == "max" else (lambda t: -t.score)
        valid = [t for t in self.trials if np.isfinite(t.score)
                 and (self.final_budget is None or t.budget == self.final_budget)]
        if not valid:
            raise ValueError("no finite trials")
        return max(valid, key=key)

    @property
    def best_params(self) -> dict:
        return self.best.params

    @property
    def best_score(self) -> float:
        return self.best.score


def grid_search(objective: Callable[[dict], float], space: dict[str, Sequence[Any]],
                mode: str = "max") -> TuningResult:
    keys = list(space)
    res = TuningResult(mode=mode)
    for combo in itertools.product(*(space[k] for k in keys)):
        params = dict(zip(keys, combo))
        res.trials.append(Trial(params, float(objective(params))))
    return res


def sample_space(space: dict[str, Any], rng: np.random.Generator) -> dict:
    out = {}
    for k, v in space.items():
        if isinstance(v, Distribution):
            out[k] = v.sample(rng)
        elif isinstance(v, (list, tuple)):
            out[k] = v[int(rng.integers(len(v)))]
        else:
            out[k] = v
    return out


def random_search(objective: Callable[[dict], float], space: dict[str, Any], n_trials: int = 20,
                  seed: int = 0, mode: str = "max") -> TuningResult:
    rng = np.random.default_rng(seed)
    res = TuningResult(mode=mode)
    for _ in range(n_trials):
        params = sample_space(space, rng)
        res.trials.append(Trial(params, float(objective(params))))
    return res


def successive_halving(objective: Callable[[dict, float], float], space: dict[str, Any],
                       n_configs: int = 27, min_budget: float = 1, max_budget: float = 27, eta: int = 3,
                       seed: int = 0, mode: str = "max") -> TuningResult:
    """Successive halving (Jamieson & Talwalkar 2016): evaluate n configs at a small
    budget, keep the top 1/eta, multiply budget by eta, repeat until max_budget.

    # TODO(ml): Hyperband brackets, ASHA (async), resuming from checkpoints instead
    # of re-training from scratch at each rung.
    """
    rng = np.random.default_rng(seed)
    configs = [sample_space(space, rng) for _ in range(n_configs)]
    res = TuningResult(mode=mode)
    budget = float(min_budget)
    sign = 1.0 if mode == "max" else -1.0
    while configs:
        scored = []
        for p in configs:
            s = float(objective(p, budget))
            res.trials.append(Trial(p, s, budget))
            scored.append((sign * s if np.isfinite(s) else -math.inf, p))
        if budget >= max_budget or len(configs) == 1:
            break
        scored.sort(key=lambda t: t[0], reverse=True)
        k = max(1, len(configs) // eta)
        configs = [p for _, p in scored[:k]]
        budget = min(budget * eta, float(max_budget))
    res.final_budget = budget  # best is chosen among survivors at the last rung
    return res
