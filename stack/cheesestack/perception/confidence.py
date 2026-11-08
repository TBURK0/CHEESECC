"""Confidence scoring: calibration of raw model scores and fusion with sensor
health and environment visibility.

- Calibrators (fit on held-out (score/logit, label) pairs):
  `TemperatureScaler` (softmax / sigmoid logits; 1-D Newton on NLL over log T),
  `PlattScaler` (logistic regression a*s + b, Newton/IRLS),
  `HistogramBinningCalibrator` (non-parametric, monotone-smoothed bins).
- Metrics: `expected_calibration_error`, `brier_score`.
- Aggregation: `aggregate(values, weights, method)` with mean / geometric /
  min / noisy_or / harmonic.
- `SensorHealth` -> health score from drop rate, latency, staleness and faults.
- `ConfidenceScorer.score(raw, health, visibility)` -> `ScoredConfidence`
  with the final value in [0,1] and a per-factor breakdown.

# TODO(perception): per-class calibrators, online recalibration from
#   tracker-confirmed outcomes, conformal prediction intervals for regressed
#   quantities (lane offset, distance), uncertainty from ensembles/MC-dropout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

EPS = 1e-12


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, 1 / (1 + np.exp(-np.abs(x))), np.exp(-np.abs(x)) / (1 + np.exp(-np.abs(x))))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def softmax(z, axis: int = -1):
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


# ----------------------------------------------------------------------------- calibrators


class Calibrator:
    fitted: bool = False

    def fit(self, scores, labels) -> "Calibrator":  # pragma: no cover - interface
        raise NotImplementedError

    def __call__(self, scores):  # pragma: no cover - interface
        raise NotImplementedError


class IdentityCalibrator(Calibrator):
    fitted = True

    def fit(self, scores, labels):
        return self

    def __call__(self, scores):
        return np.clip(np.asarray(scores, float), 0, 1)


class TemperatureScaler(Calibrator):
    """p = softmax(z / T) for (N, C) logits with integer labels, or sigmoid(z / T) for (N,) logits
    with binary labels. T fit by Newton on NLL w.r.t. log T (numerical derivatives, convex in 1/T)."""

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    def _nll(self, z: np.ndarray, y: np.ndarray, T: float) -> float:
        if z.ndim == 1:
            p = sigmoid(z / T)
            return float(-np.mean(y * np.log(p + EPS) + (1 - y) * np.log(1 - p + EPS)))
        p = softmax(z / T, axis=1)
        return float(-np.mean(np.log(p[np.arange(len(y)), y.astype(int)] + EPS)))

    def fit(self, logits, labels, iters: int = 50) -> "TemperatureScaler":
        z, y = np.asarray(logits, float), np.asarray(labels)
        s = math.log(self.temperature)
        h = 1e-4
        for _ in range(iters):
            f0 = self._nll(z, y, math.exp(s))
            fp, fm = self._nll(z, y, math.exp(s + h)), self._nll(z, y, math.exp(s - h))
            g, H = (fp - fm) / (2 * h), (fp - 2 * f0 + fm) / h ** 2
            step = -g / H if H > 1e-12 else -0.5 * np.sign(g)
            step = float(np.clip(step, -1.0, 1.0))
            s += step
            if abs(step) < 1e-7:
                break
        self.temperature = float(math.exp(np.clip(s, -5, 5)))
        self.fitted = True
        return self

    def __call__(self, logits):
        z = np.asarray(logits, float)
        return sigmoid(z / self.temperature) if z.ndim <= 1 else softmax(z / self.temperature, axis=-1)


class PlattScaler(Calibrator):
    """p = sigmoid(a * s + b) fit by Newton-Raphson (IRLS) with Platt's smoothed targets.
    Input `s` may be a probability (`on_logit=True` converts to logit first) or a raw score."""

    def __init__(self, a: float = 1.0, b: float = 0.0, on_logit: bool = True) -> None:
        self.a, self.b, self.on_logit = float(a), float(b), on_logit

    def _x(self, s):
        return logit(s) if self.on_logit else np.asarray(s, float)

    def fit(self, scores, labels, iters: int = 100, l2: float = 1e-6) -> "PlattScaler":
        x, y = self._x(scores), np.asarray(labels, float)
        npos, nneg = y.sum(), len(y) - y.sum()
        t = np.where(y > 0.5, (npos + 1) / (npos + 2), 1 / (nneg + 2))  # Platt targets
        w = np.array([self.a, self.b])
        X = np.column_stack([x, np.ones_like(x)])
        def obj(w):
            m = X @ w  # stable cross-entropy: log(1+e^m) - t*m
            return float(np.sum(np.logaddexp(0, m) - t * m) + 0.5 * l2 * w @ w)

        f = obj(w)
        for _ in range(iters):
            p = sigmoid(X @ w)
            g = X.T @ (p - t) + l2 * w
            Hm = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(2)
            step = np.linalg.solve(Hm + 1e-9 * np.eye(2), g)
            alpha = 1.0  # backtracking line search (Newton alone diverges from saturated starts)
            while alpha > 1e-8:
                wn = w - alpha * step
                fn = obj(wn)
                if fn <= f - 1e-4 * alpha * float(g @ step):
                    break
                alpha *= 0.5
            if alpha <= 1e-8:
                break
            w, f = wn, fn
            if np.max(np.abs(alpha * step)) < 1e-9:
                break
        self.a, self.b = float(w[0]), float(w[1])
        self.fitted = True
        return self

    def __call__(self, scores):
        return sigmoid(self.a * self._x(scores) + self.b)


class HistogramBinningCalibrator(Calibrator):
    """Equal-width bins on [0,1]; each bin maps to its empirical positive rate (Laplace-smoothed),
    then made monotone non-decreasing (pool-adjacent-violators on bin means)."""

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins
        self.values = np.linspace(0.5 / n_bins, 1 - 0.5 / n_bins, n_bins)

    def fit(self, scores, labels) -> "HistogramBinningCalibrator":
        s, y = np.clip(np.asarray(scores, float), 0, 1), np.asarray(labels, float)
        idx = np.minimum((s * self.n_bins).astype(int), self.n_bins - 1)
        cnt = np.bincount(idx, minlength=self.n_bins).astype(float)
        pos = np.bincount(idx, weights=y, minlength=self.n_bins)
        prior = self.values
        vals = (pos + prior) / (cnt + 1)  # one pseudo-count at the bin center
        self.values = _pava(vals, cnt + 1)
        self.fitted = True
        return self

    def __call__(self, scores):
        s = np.clip(np.asarray(scores, float), 0, 1)
        idx = np.minimum((s * self.n_bins).astype(int), self.n_bins - 1)
        return self.values[idx]


def _pava(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators -> non-decreasing sequence."""
    blocks = [[float(v[i]), float(w[i]), 1] for i in range(len(v))]
    out: list[list] = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            b2, b1 = out.pop(), out.pop()
            tw = b1[1] + b2[1]
            out.append([(b1[0] * b1[1] + b2[0] * b2[1]) / tw, tw, b1[2] + b2[2]])
    return np.concatenate([[m] * n for m, _, n in out])


# ----------------------------------------------------------------------------- metrics


def expected_calibration_error(probs, labels, n_bins: int = 10) -> float:
    """ECE for binary probabilities (probability of the positive class)."""
    p, y = np.clip(np.asarray(probs, float), 0, 1), np.asarray(labels, float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def brier_score(probs, labels) -> float:
    p, y = np.asarray(probs, float), np.asarray(labels, float)
    return float(np.mean((p - y) ** 2))


# ----------------------------------------------------------------------------- aggregation


def aggregate(values: Sequence[float], weights: Optional[Sequence[float]] = None, method: str = "geometric") -> float:
    """Combine confidences in [0,1].
    - mean: weighted arithmetic mean
    - geometric: weighted geometric mean (one weak factor pulls hard)
    - harmonic: weighted harmonic mean
    - min: weakest link (weights ignored)
    - noisy_or: 1 - prod(1 - w*v) (independent evidence for the same hypothesis)
    """
    v = np.clip(np.asarray(values, float), 0, 1)
    if v.size == 0:
        return 0.0
    w = np.ones_like(v) if weights is None else np.asarray(weights, float)
    if method == "min":
        return float(v.min())
    if method == "noisy_or":
        return clamp01(1 - np.prod(1 - np.clip(w, 0, 1) * v))
    ws = w.sum()
    if ws <= 0:
        return 0.0
    if method == "mean":
        return clamp01((w * v).sum() / ws)
    if method == "geometric":
        return clamp01(math.exp((w * np.log(v + EPS)).sum() / ws))
    if method == "harmonic":
        return clamp01(ws / (w / (v + EPS)).sum())
    raise ValueError(f"unknown method {method}")


# ----------------------------------------------------------------------------- health + scorer


@dataclass
class SensorHealth:
    """Inputs typically from camera_ingest.IngestStats and the safety/fault system."""

    drop_rate: float = 0.0  # [0,1]
    latency: float = 0.0  # s
    age: float = 0.0  # s since last good frame
    fault: bool = False  # hard fault (e.g. Severity >= DEGRADED)
    calibration_ok: bool = True


@dataclass
class ConfidenceConfig:
    max_drop_rate: float = 0.3  # health -> 0 at this drop rate
    latency_soft: float = 0.05  # s, no penalty below
    latency_hard: float = 0.3  # s, health -> 0
    age_soft: float = 0.1
    age_hard: float = 0.5
    uncalibrated_factor: float = 0.5
    visibility_exponent: float = 1.0  # weight of environment visibility
    method: str = "product"  # "product" or any `aggregate` method
    floor: float = 0.0
    ceiling: float = 1.0


@dataclass
class ScoredConfidence:
    value: float
    factors: dict[str, float] = field(default_factory=dict)


def _soft_hard(x: float, soft: float, hard: float) -> float:
    if x <= soft:
        return 1.0
    if x >= hard:
        return 0.0
    return 1.0 - (x - soft) / (hard - soft)


class ConfidenceScorer:
    """final = calibrate(raw) * health * visibility^k (product mode), clipped to [floor, ceiling]."""

    def __init__(self, config: Optional[ConfidenceConfig] = None, calibrator: Optional[Calibrator] = None) -> None:
        self.config = config or ConfidenceConfig()
        self.calibrator = calibrator or IdentityCalibrator()

    def health_score(self, h: Optional[SensorHealth]) -> float:
        if h is None:
            return 1.0
        if h.fault:
            return 0.0
        c = self.config
        s = clamp01(1 - h.drop_rate / c.max_drop_rate) if c.max_drop_rate > 0 else 1.0
        s *= _soft_hard(h.latency, c.latency_soft, c.latency_hard)
        s *= _soft_hard(h.age, c.age_soft, c.age_hard)
        if not h.calibration_ok:
            s *= c.uncalibrated_factor
        return clamp01(s)

    def score(self, raw: float, health: Optional[SensorHealth] = None, visibility: float = 1.0) -> ScoredConfidence:
        c = self.config
        cal = float(np.asarray(self.calibrator(np.array([raw]))).ravel()[0])
        hs = self.health_score(health)
        vis = clamp01(visibility) ** c.visibility_exponent
        f = {"calibrated": cal, "health": hs, "visibility": vis}
        if c.method == "product":
            v = cal * hs * vis
        else:
            v = aggregate(list(f.values()), method=c.method)
        return ScoredConfidence(float(np.clip(v, c.floor, c.ceiling)), f)

    def score_many(self, raws: Sequence[float], health: Optional[SensorHealth] = None, visibility: float = 1.0) -> np.ndarray:
        """Vectorized product-mode scoring for many outputs sharing one sensor/environment."""
        cal = np.asarray(self.calibrator(np.asarray(raws, float)), float).ravel()
        k = self.health_score(health) * clamp01(visibility) ** self.config.visibility_exponent
        return np.clip(cal * k, self.config.floor, self.config.ceiling)
