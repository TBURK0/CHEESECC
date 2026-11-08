"""Risk scoring: fuse TTC, collision probability, vulnerability and uncertainty into [0, 1].

Model (all terms in [0, 1]):

* TTC term ``r_ttc``: 1 below ``ttc_critical``, 0 above ``ttc_safe``, smooth
  (cosine) ramp between.
* Collision term ``p``: predicted collision probability.
* Hazard ``h = 1 - (1 - p)(1 - w_ttc * r_ttc)`` (noisy-OR of the two cues).
* Vulnerability ``vuln`` per class (pedestrian 1.0 ... static 0.3): scales
  severity, ``sev = vuln_floor + (1 - vuln_floor) * vuln``.
* Uncertainty ``u`` (e.g. from track covariance via ``uncertainty_from_cov``)
  inflates risk: ``risk = clip(h * sev * (1 + k_u * u), 0, 1)``.

The score is a heuristic for ranking / gating, not a calibrated probability of harm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cheesestack.core import ObjectClass


def _default_vulnerability() -> dict:
    return {
        ObjectClass.PEDESTRIAN: 1.0,
        ObjectClass.BICYCLE: 0.95,
        ObjectClass.MOTORCYCLE: 0.9,
        ObjectClass.ANIMAL: 0.7,
        ObjectClass.CAR: 0.6,
        ObjectClass.UNKNOWN: 0.7,
        ObjectClass.BUS: 0.7,  # many occupants, large mass
        ObjectClass.TRUCK: 0.7,
        ObjectClass.STATIC_OBSTACLE: 0.4,
        ObjectClass.TRAFFIC_SIGN: 0.3,
        ObjectClass.TRAFFIC_LIGHT: 0.3,
    }


@dataclass
class RiskScorerConfig:
    ttc_critical: float = 1.5  # s
    ttc_safe: float = 6.0  # s
    w_ttc: float = 0.9
    vuln_floor: float = 0.5
    k_uncertainty: float = 0.5
    uncertainty_scale: float = 2.0  # m; std-dev that maps to u ≈ 0.63
    vulnerability: dict = field(default_factory=_default_vulnerability)
    levels: tuple = (("low", 0.0), ("medium", 0.3), ("high", 0.6), ("critical", 0.85))


@dataclass
class RiskAssessment:
    risk: float
    level: str
    ttc_term: float
    collision_term: float
    hazard: float
    severity: float
    uncertainty: float
    track_id: int = -1


def uncertainty_from_cov(cov: Optional[np.ndarray], scale: float = 2.0) -> float:
    """Map a position covariance (>= 2x2) to [0, 1): 1 - exp(-sqrt(trace(P_xy)) / scale)."""
    if cov is None:
        return 0.0
    c = np.asarray(cov, dtype=float)
    tr = max(float(c[0, 0] + c[1, 1]), 0.0)
    return 1.0 - math.exp(-math.sqrt(tr) / scale)


class RiskScorer:
    def __init__(self, config: Optional[RiskScorerConfig] = None):
        self.cfg = config or RiskScorerConfig()

    def ttc_term(self, ttc: float) -> float:
        c = self.cfg
        if not math.isfinite(ttc) or ttc >= c.ttc_safe:
            return 0.0
        if ttc <= c.ttc_critical:
            return 1.0
        x = (ttc - c.ttc_critical) / (c.ttc_safe - c.ttc_critical)
        return 0.5 * (1.0 + math.cos(math.pi * x))

    def level(self, risk: float) -> str:
        name = self.cfg.levels[0][0]
        for n, thr in self.cfg.levels:
            if risk >= thr:
                name = n
        return name

    def score(self, ttc: float = math.inf, collision_probability: float = 0.0,
              cls: ObjectClass = ObjectClass.UNKNOWN, uncertainty: float = 0.0,
              track_id: int = -1) -> RiskAssessment:
        c = self.cfg
        r_ttc = self.ttc_term(ttc)
        p = float(np.clip(collision_probability, 0.0, 1.0))
        u = float(np.clip(uncertainty, 0.0, 1.0))
        hazard = 1.0 - (1.0 - p) * (1.0 - c.w_ttc * r_ttc)
        vuln = c.vulnerability.get(cls, c.vulnerability.get(ObjectClass.UNKNOWN, 0.7))
        sev = c.vuln_floor + (1.0 - c.vuln_floor) * vuln
        risk = float(np.clip(hazard * sev * (1.0 + c.k_uncertainty * u), 0.0, 1.0))
        # TODO(prediction): calibrate against logged near-misses; add relative-speed severity
        # (impact energy) and occlusion/visibility terms.
        return RiskAssessment(risk, self.level(risk), r_ttc, p, hazard, sev, u, track_id)

    def aggregate(self, assessments: list) -> float:
        """Scene risk = noisy-OR over per-object risks."""
        q = 1.0
        for a in assessments:
            q *= 1.0 - a.risk
        return 1.0 - q
