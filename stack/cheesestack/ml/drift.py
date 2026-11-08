"""Data / embedding drift detection.

- ``psi``  Population Stability Index over reference-quantile bins:
  sum((c - r) * ln(c / r)). Rule of thumb: <0.1 stable, 0.1-0.25 moderate, >0.25 major.
- ``ks_2samp``  two-sample Kolmogorov-Smirnov statistic D = sup|F1 - F2| with
  the asymptotic Kolmogorov p-value (Stephens' small-sample correction).
- ``embedding_mean_shift``  distance between embedding means, in units of the
  reference per-dimension std (a cheap MMD-with-linear-kernel proxy), plus cosine distance.

``DriftDetector`` fits per-feature references and emits ``core.Fault`` alerts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from cheesestack.core import Fault, Severity


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    ref = np.asarray(reference, float).ravel()
    cur = np.asarray(current, float).ravel()
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:  # constant reference: compare "equal to it" vs not
        r = np.array([1.0, 0.0])
        c = np.array([np.mean(cur == ref[0]), np.mean(cur != ref[0])])
    else:
        inner = edges[1:-1]
        r = np.bincount(np.searchsorted(inner, ref, side="right"), minlength=len(edges) - 1) / len(ref)
        c = np.bincount(np.searchsorted(inner, cur, side="right"), minlength=len(edges) - 1) / max(len(cur), 1)
    r = np.clip(r, eps, None)
    c = np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def kolmogorov_sf(x: float, terms: int = 100) -> float:
    """Survival function of the Kolmogorov distribution: 2 sum (-1)^(k-1) exp(-2 k^2 x^2)."""
    if x <= 0:
        return 1.0
    k = np.arange(1, terms + 1)
    return float(np.clip(2 * np.sum((-1.0) ** (k - 1) * np.exp(-2 * k ** 2 * x ** 2)), 0.0, 1.0))


def ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (D, p-value)."""
    a = np.sort(np.asarray(a, float).ravel())
    b = np.sort(np.asarray(b, float).ravel())
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0, 1.0
    allv = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, allv, side="right") / n
    cdf_b = np.searchsorted(b, allv, side="right") / m
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    en = np.sqrt(n * m / (n + m))
    return d, kolmogorov_sf((en + 0.12 + 0.11 / en) * d)


@dataclass
class EmbeddingShift:
    normalized_shift: float  # ||mu_c - mu_r|| / sqrt(mean var_r)
    cosine_distance: float
    per_dim_z: np.ndarray  # (mu_c - mu_r) / (std_r / sqrt(n_c))


def embedding_mean_shift(reference: np.ndarray, current: np.ndarray) -> EmbeddingShift:
    R = np.asarray(reference, float)
    C = np.asarray(current, float)
    mr, mc = R.mean(0), C.mean(0)
    sr = R.std(0) + 1e-9
    shift = float(np.linalg.norm(mc - mr) / np.sqrt(np.mean(sr ** 2)))
    cos = 1.0 - float(mr @ mc / (np.linalg.norm(mr) * np.linalg.norm(mc) + 1e-12))
    z = (mc - mr) / (sr / np.sqrt(len(C)))
    return EmbeddingShift(shift, cos, z)


@dataclass
class DriftConfig:
    psi_warn: float = 0.1
    psi_alert: float = 0.25
    ks_pvalue: float = 0.01
    ks_min_d: float = 0.1  # ignore statistically significant but tiny shifts on big samples
    embedding_shift_alert: float = 0.5
    bins: int = 10
    source: str = "ml.drift"


@dataclass
class FeatureDrift:
    name: str
    psi: float
    ks_d: float
    ks_p: float
    severity: Severity


@dataclass
class DriftReport:
    features: list[FeatureDrift]
    embedding: Optional[EmbeddingShift] = None
    faults: list[Fault] = field(default_factory=list)

    @property
    def drifted(self) -> list[str]:
        return [f.name for f in self.features if f.severity >= Severity.WARNING]


class DriftDetector:
    """Per-feature PSI + KS against a frozen reference window.

    # TODO(ml): sequential tests (ADWIN / Page-Hinkley) for streaming, image-level
    # stats (brightness histograms, blur) as standard features, MMD with RBF kernel.
    """

    def __init__(self, config: Optional[DriftConfig] = None) -> None:
        self.config = config or DriftConfig()
        self.reference: Optional[np.ndarray] = None
        self.names: list[str] = []
        self.ref_embeddings: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, feature_names: Optional[Sequence[str]] = None,
            embeddings: Optional[np.ndarray] = None) -> "DriftDetector":
        X = np.asarray(X, float)
        X = X[:, None] if X.ndim == 1 else X
        self.reference = X
        self.names = list(feature_names) if feature_names else [f"f{i}" for i in range(X.shape[1])]
        self.ref_embeddings = None if embeddings is None else np.asarray(embeddings, float)
        return self

    def check(self, X: np.ndarray, stamp: float = 0.0, embeddings: Optional[np.ndarray] = None) -> DriftReport:
        if self.reference is None:
            raise RuntimeError("DriftDetector.fit must be called first")
        cfg = self.config
        X = np.asarray(X, float)
        X = X[:, None] if X.ndim == 1 else X
        feats, faults = [], []
        for i, name in enumerate(self.names):
            p = psi(self.reference[:, i], X[:, i], cfg.bins)
            d, pv = ks_2samp(self.reference[:, i], X[:, i])
            ks_hit = pv < cfg.ks_pvalue and d >= cfg.ks_min_d
            if p >= cfg.psi_alert:
                sev = Severity.DEGRADED
            elif p >= cfg.psi_warn or ks_hit:
                sev = Severity.WARNING
            else:
                sev = Severity.INFO
            feats.append(FeatureDrift(name, p, d, pv, sev))
            if sev >= Severity.WARNING:
                faults.append(Fault(f"ML_DRIFT_{name.upper()}", sev, cfg.source, stamp,
                                    f"feature '{name}' drift psi={p:.3f} ks={d:.3f} p={pv:.2g}",
                                    {"psi": p, "ks_d": d, "ks_p": pv}))
        emb = None
        if embeddings is not None and self.ref_embeddings is not None:
            emb = embedding_mean_shift(self.ref_embeddings, embeddings)
            if emb.normalized_shift >= cfg.embedding_shift_alert:
                faults.append(Fault("ML_DRIFT_EMBEDDING", Severity.DEGRADED, cfg.source, stamp,
                                    f"embedding mean shift {emb.normalized_shift:.3f}",
                                    {"shift": emb.normalized_shift, "cosine": emb.cosine_distance}))
        return DriftReport(feats, emb, faults)
