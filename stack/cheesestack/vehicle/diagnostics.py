"""Generic DTC-style diagnostic state machines with counter-based debouncing.

Each `DiagnosticTroubleCode` follows an AUTOSAR-DEM-like lifecycle:

    NO_FAULT --fail--> PENDING --counter>=confirm--> CONFIRMED
       ^                  |                              |
       |              counter<=heal                  counter<=heal
       |                  v                              v
       +------------- (NO_FAULT)                       HEALED --aging cycles--> NO_FAULT
                                                         |
                                                       fail -> PENDING/CONFIRMED (re-occurrence)

The debounce counter moves +`fail_step` on a failed test and -`pass_step` on a
passed one, saturating at [`heal_threshold`, `confirm_threshold`]. Aging counts
*operation cycles* (`end_cycle()`) spent HEALED without a failure.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

from cheesestack.core import Fault, Severity


class DTCStatus(enum.Enum):
    NO_FAULT = "no_fault"
    PENDING = "pending"
    CONFIRMED = "confirmed"
    HEALED = "healed"


@dataclass
class DTCConfig:
    confirm_threshold: int = 10
    heal_threshold: int = -10
    fail_step: int = 2
    pass_step: int = 1
    aging_cycles: int = 3
    severity: Severity = Severity.WARNING


@dataclass
class DTCTransition:
    code: str
    old: DTCStatus
    new: DTCStatus
    stamp: float


class DiagnosticTroubleCode:
    def __init__(self, code: str, config: Optional[DTCConfig] = None, description: str = "") -> None:
        self.code = code
        self.description = description
        self.config = config or DTCConfig()
        self.status = DTCStatus.NO_FAULT
        self.counter = 0
        self.occurrences = 0
        self.first_failed: Optional[float] = None
        self.last_failed: Optional[float] = None
        self.freeze_frame: dict = {}
        self.healed_cycles = 0
        self._failed_this_cycle = False

    @property
    def active(self) -> bool:
        return self.status in (DTCStatus.PENDING, DTCStatus.CONFIRMED)

    def report(self, failed: bool, stamp: float, freeze_frame: Optional[dict] = None) -> Optional[DTCTransition]:
        """Feed one test result. Returns the transition if the status changed."""
        c = self.config
        old = self.status
        if failed:
            self.counter = min(c.confirm_threshold, max(0, self.counter) + c.fail_step)
            self._failed_this_cycle = True
            self.last_failed = stamp
            if self.first_failed is None:
                self.first_failed = stamp
            if self.status in (DTCStatus.NO_FAULT, DTCStatus.HEALED):
                self.status = DTCStatus.PENDING
                self.healed_cycles = 0
            if self.status == DTCStatus.PENDING and self.counter >= c.confirm_threshold:
                self.status = DTCStatus.CONFIRMED
                self.occurrences += 1
                self.freeze_frame = dict(freeze_frame or {}, stamp=stamp)
        else:
            self.counter = max(c.heal_threshold, min(0, self.counter) - c.pass_step) if self.counter <= 0 \
                else max(0, self.counter - c.pass_step)
            if self.counter <= c.heal_threshold:
                if self.status == DTCStatus.CONFIRMED:
                    self.status = DTCStatus.HEALED
                elif self.status == DTCStatus.PENDING:
                    self.status = DTCStatus.NO_FAULT
        if self.status != old:
            return DTCTransition(self.code, old, self.status, stamp)
        return None

    def end_cycle(self) -> Optional[DTCTransition]:
        """Close an operation (ignition/drive) cycle; ages HEALED codes."""
        old = self.status
        if self.status == DTCStatus.HEALED and not self._failed_this_cycle:
            self.healed_cycles += 1
            if self.healed_cycles >= self.config.aging_cycles:
                self.status = DTCStatus.NO_FAULT
                self.healed_cycles = 0
        self._failed_this_cycle = False
        return DTCTransition(self.code, old, self.status, float("nan")) if self.status != old else None

    def clear(self) -> None:
        """Service-tool clear: back to NO_FAULT, history wiped."""
        self.__init__(self.code, self.config, self.description)

    def to_fault(self, stamp: float, source: str = "vehicle.diagnostics") -> Fault:
        sev = self.config.severity if self.status == DTCStatus.CONFIRMED else Severity.INFO
        return Fault(self.code, sev, source, stamp, self.description,
                     {"status": self.status.value, "counter": self.counter, "occurrences": self.occurrences,
                      "freeze_frame": dict(self.freeze_frame)})


@dataclass
class DiagnosticManager:
    """Registry of DTCs. Unknown codes are auto-registered with `default_config`."""

    default_config: DTCConfig = field(default_factory=DTCConfig)
    source: str = "vehicle.diagnostics"

    def __post_init__(self) -> None:
        self.dtcs: dict[str, DiagnosticTroubleCode] = {}
        self.history: list[DTCTransition] = []

    def register(self, code: str, config: Optional[DTCConfig] = None, description: str = "") -> DiagnosticTroubleCode:
        dtc = DiagnosticTroubleCode(code, config or self.default_config, description)
        self.dtcs[code] = dtc
        return dtc

    def report(self, code: str, failed: bool, stamp: float, freeze_frame: Optional[dict] = None) -> Optional[DTCTransition]:
        dtc = self.dtcs.get(code) or self.register(code)
        tr = dtc.report(failed, stamp, freeze_frame)
        if tr:
            self.history.append(tr)
        return tr

    def end_cycle(self) -> list[DTCTransition]:
        out = [tr for d in self.dtcs.values() if (tr := d.end_cycle())]
        self.history.extend(out)
        return out

    def status(self, code: str) -> DTCStatus:
        d = self.dtcs.get(code)
        return d.status if d else DTCStatus.NO_FAULT

    def confirmed(self) -> list[str]:
        return [c for c, d in self.dtcs.items() if d.status == DTCStatus.CONFIRMED]

    def active_faults(self, stamp: float) -> list[Fault]:
        """Faults for CONFIRMED codes (PENDING ones are reported as INFO)."""
        return [d.to_fault(stamp, self.source) for d in self.dtcs.values() if d.active]
    # TODO(vehicle): persistence of DTC memory across power cycles; UDS-style status byte export.
