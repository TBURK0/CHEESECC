"""System health scoring.

Each monitor contributes a component score in [0, 1] (1 = healthy) with a
weight. The overall score is

    S = (sum_i w_i * s_i / sum_i w_i) * prod_f penalty(severity_f)

where the product runs over active faults (a fault already reflected in a
component still counts - deliberately pessimistic). A component flagged
`critical` with score below `critical_floor` caps S at `critical_cap`.

Mode recommendation = the more restrictive of
  * score thresholds: S >= nominal_min -> NOMINAL, >= degraded_min -> DEGRADED, else MRM
  * worst active fault severity via `severity_to_mode`.
Non-finite component scores are treated as 0 (unknown = unhealthy).

# TODO(safety): calibrate weights/penalties against fault-injection campaigns;
# per-ODD thresholds; hysteresis on the score itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

from cheesestack.core import Fault, Severity

from .fallback import SafetyMode, severity_to_mode


@dataclass
class HealthConfig:
    weights: dict = field(default_factory=lambda: {
        "watchdog": 3.0, "heartbeat": 2.0, "sensors": 3.0, "perception": 2.0,
        "redundancy": 2.0, "integrity": 1.0, "driver": 1.0,
    })
    default_weight: float = 1.0
    critical: tuple = ("watchdog", "sensors")
    critical_floor: float = 0.5
    critical_cap: float = 0.4
    severity_penalty: dict = field(default_factory=lambda: {
        Severity.INFO: 1.0, Severity.WARNING: 0.97, Severity.DEGRADED: 0.85,
        Severity.CRITICAL: 0.5, Severity.FATAL: 0.0,
    })
    nominal_min: float = 0.8
    degraded_min: float = 0.5


@dataclass
class HealthReport:
    score: float
    components: dict
    recommended_mode: SafetyMode
    worst_severity: Severity
    limiting: str  # what drove the recommendation


class HealthScorer:
    def __init__(self, config: Optional[HealthConfig] = None) -> None:
        self.config = config or HealthConfig()

    def evaluate(self, components: dict[str, float], faults: Iterable[Fault] = ()) -> HealthReport:
        c = self.config
        comps = {k: (min(max(float(v), 0.0), 1.0) if math.isfinite(float(v)) else 0.0) for k, v in components.items()}
        faults = list(faults)
        if comps:
            wsum = sum(c.weights.get(k, c.default_weight) for k in comps)
            base = sum(c.weights.get(k, c.default_weight) * v for k, v in comps.items()) / wsum if wsum > 0 else 0.0
        else:
            base = 1.0
        pen = 1.0
        for f in faults:
            pen *= c.severity_penalty.get(f.severity, 0.0)
        score = base * pen
        limiting = "score"
        for k in c.critical:
            if k in comps and comps[k] < c.critical_floor and score > c.critical_cap:
                score, limiting = c.critical_cap, f"critical component {k}"
        worst = max((f.severity for f in faults), default=Severity.INFO)
        if score >= c.nominal_min:
            m_score = SafetyMode.NOMINAL
        elif score >= c.degraded_min:
            m_score = SafetyMode.DEGRADED
        else:
            m_score = SafetyMode.MINIMAL_RISK_MANEUVER
        m_fault = severity_to_mode(worst)
        if m_fault > m_score:
            mode, limiting = m_fault, f"fault severity {worst.name}"
        else:
            mode = m_score
        return HealthReport(score, comps, mode, worst, limiting)
