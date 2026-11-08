"""Vehicle health monitoring: aggregate subsystem health into one report.

Subsystems (steering, wheel speed, IMU, GPS, actuators, ...) are registered with
a criticality weight and an optional staleness timeout. Each reports a
`Severity` (or a `NodeStatus`, mapped to severity) and optional faults.
Aggregation:
- overall severity = max over subsystems, where a non-critical subsystem's
  severity is capped at DEGRADED and a stale subsystem counts as its
  `stale_severity`;
- score in [0, 1] = weighted mean of per-subsystem scores (1 - severity/FATAL);
- `ok_to_drive` iff overall < CRITICAL; `degraded` iff overall >= DEGRADED.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Union

from cheesestack.core import Fault, NodeStatus, Severity

_NODE_TO_SEV = {
    NodeStatus.INIT: Severity.WARNING,
    NodeStatus.OK: Severity.INFO,
    NodeStatus.DEGRADED: Severity.DEGRADED,
    NodeStatus.FAILED: Severity.CRITICAL,
    NodeStatus.STOPPED: Severity.CRITICAL,
}


@dataclass
class SubsystemSpec:
    name: str
    weight: float = 1.0
    critical: bool = True
    timeout_s: float = math.inf
    stale_severity: Severity = Severity.CRITICAL


@dataclass
class SubsystemHealth:
    name: str
    severity: Severity = Severity.WARNING  # unknown until first report
    stamp: float = -math.inf
    message: str = ""
    faults: list = field(default_factory=list)
    stale: bool = False

    @property
    def score(self) -> float:
        return 1.0 - float(self.severity) / float(Severity.FATAL)


@dataclass
class VehicleHealthReport:
    stamp: float
    overall: Severity
    score: float
    ok_to_drive: bool
    degraded: bool
    subsystems: dict
    faults: list
    worst: Optional[str] = None


@dataclass
class HealthConfig:
    source: str = "vehicle.health"
    never_reported_severity: Severity = Severity.WARNING


class HealthMonitor:
    def __init__(self, specs: Optional[Iterable[SubsystemSpec]] = None, config: Optional[HealthConfig] = None) -> None:
        self.config = config or HealthConfig()
        self.specs: dict[str, SubsystemSpec] = {}
        self.state: dict[str, SubsystemHealth] = {}
        for s in specs or []:
            self.register(s)

    def register(self, spec: SubsystemSpec) -> None:
        self.specs[spec.name] = spec
        self.state[spec.name] = SubsystemHealth(spec.name, self.config.never_reported_severity)

    def update(self, name: str, t: float, status: Union[Severity, NodeStatus], message: str = "",
               faults: Optional[list[Fault]] = None) -> None:
        if name not in self.specs:
            self.register(SubsystemSpec(name))
        sev = _NODE_TO_SEV[status] if isinstance(status, NodeStatus) else Severity(status)
        faults = list(faults or [])
        if faults:
            sev = max(sev, max(f.severity for f in faults))
        self.state[name] = SubsystemHealth(name, sev, t, message, faults)

    def update_from_faults(self, name: str, t: float, faults: list[Fault]) -> None:
        """Severity = worst fault severity (INFO if none)."""
        self.update(name, t, Severity.INFO, "", faults)

    def report(self, t: float) -> VehicleHealthReport:
        subs: dict[str, SubsystemHealth] = {}
        overall = Severity.INFO
        worst, worst_sev = None, -1
        wsum = ssum = 0.0
        faults: list[Fault] = []
        for name, spec in self.specs.items():
            h = self.state[name]
            stale = h.stamp > -math.inf and (t - h.stamp) > spec.timeout_s
            sev = max(h.severity, spec.stale_severity) if stale else h.severity
            eff = sev if spec.critical else min(sev, Severity.DEGRADED)
            sh = SubsystemHealth(name, sev, h.stamp, h.message, h.faults, stale)
            subs[name] = sh
            if stale:
                faults.append(Fault(f"HEALTH_{name.upper()}_STALE", eff, self.config.source, t,
                                    f"{name} last reported {t - h.stamp:.2f}s ago"))
            faults.extend(h.faults)
            overall = max(overall, eff)
            if eff > worst_sev:
                worst, worst_sev = name, eff
            wsum += spec.weight
            ssum += spec.weight * sh.score
        score = ssum / wsum if wsum > 0 else 1.0
        # TODO(vehicle): hysteresis on ok_to_drive; per-ODD capability mapping (e.g. GPS loss => lane-keep only).
        return VehicleHealthReport(t, Severity(overall), score, overall < Severity.CRITICAL,
                                   overall >= Severity.DEGRADED, subs, faults, worst)
