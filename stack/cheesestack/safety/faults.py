"""Fault classification: a registry of fault codes, a classifier that maps raw
fault evidence (strings, exceptions, dicts) onto `core.Fault`, and an active
fault table with latching and debounced healing.

Every monitor in the safety layer emits `core.Fault` objects whose `code` is a
key of the `FaultRegistry`. The registry is the single source of truth for a
code's default severity, whether it latches, and its recovery hint.

Conservative defaults:
  * Unknown codes and unclassifiable raw faults map to `Severity.CRITICAL`
    (configurable) - an unexplained fault in a driving system must lead to a
    minimal-risk maneuver, not be ignored.
  * Latching faults stay active until `reset()` is called explicitly.
  * Non-latching faults only heal after they have not been re-reported for
    `heal_time_s` (debounce against flapping).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from cheesestack.core import Fault, Severity


@dataclass(frozen=True)
class FaultSpec:
    """Static description of a fault code."""

    code: str
    severity: Severity
    description: str = ""
    latching: bool = False
    recovery: str = "none"  # hint for RecoveryManager: none|retry|reset|degrade|escalate


# Canonical codes. Prefixes: WDG watchdog, HB heartbeat, SNS sensor, MDL model,
# RED redundancy, E2E data integrity, CMD/TRAJ/STATE input validation,
# DRV driver, AST assertion, REC recovery, SYS system.
_DEFAULT_SPECS: tuple[FaultSpec, ...] = (
    FaultSpec("WDG_LATE", Severity.WARNING, "node missed its deadline", recovery="none"),
    FaultSpec("WDG_TIMEOUT", Severity.DEGRADED, "node exceeded watchdog timeout", recovery="reset"),
    FaultSpec("WDG_EXPIRED", Severity.CRITICAL, "node unresponsive beyond escalation limit", recovery="reset"),
    FaultSpec("WDG_NODE_FAILED", Severity.CRITICAL, "node reported FAILED status", recovery="reset"),
    FaultSpec("HB_JITTER", Severity.WARNING, "heartbeat jitter above limit"),
    FaultSpec("HB_RATE", Severity.WARNING, "heartbeat rate below expected"),
    FaultSpec("HB_STALE", Severity.DEGRADED, "heartbeat stale", recovery="retry"),
    FaultSpec("HB_DEAD", Severity.CRITICAL, "heartbeat dead", recovery="reset"),
    FaultSpec("SNS_STALE", Severity.DEGRADED, "sensor data too old", recovery="retry"),
    FaultSpec("SNS_NONFINITE", Severity.DEGRADED, "sensor value NaN/Inf", recovery="retry"),
    FaultSpec("SNS_RANGE", Severity.DEGRADED, "sensor value out of range", recovery="retry"),
    FaultSpec("SNS_RATE", Severity.WARNING, "sensor rate-of-change implausible"),
    FaultSpec("SNS_FROZEN", Severity.DEGRADED, "sensor value frozen", recovery="reset"),
    FaultSpec("SNS_TIME", Severity.DEGRADED, "sensor timestamp invalid (future / non-monotonic)"),
    FaultSpec("MDL_SCHEMA", Severity.CRITICAL, "model output schema/shape/dtype mismatch", recovery="reset"),
    FaultSpec("MDL_RANGE", Severity.DEGRADED, "model output value out of range"),
    FaultSpec("MDL_IMPLAUSIBLE", Severity.WARNING, "physically implausible detection"),
    FaultSpec("MDL_JUMP", Severity.WARNING, "track jumped implausibly"),
    FaultSpec("MDL_COUNT_JUMP", Severity.WARNING, "sudden change in detection count"),
    FaultSpec("MDL_OOD", Severity.DEGRADED, "model input out of distribution"),
    FaultSpec("RED_DISAGREE", Severity.DEGRADED, "redundant channels disagree", recovery="degrade"),
    FaultSpec("RED_NO_MAJORITY", Severity.CRITICAL, "no majority among redundant channels", latching=True),
    FaultSpec("E2E_CRC", Severity.DEGRADED, "E2E CRC mismatch"),
    FaultSpec("E2E_SEQUENCE", Severity.DEGRADED, "E2E counter sequence error"),
    FaultSpec("E2E_REPEATED", Severity.WARNING, "E2E repeated counter"),
    FaultSpec("E2E_TIMEOUT", Severity.DEGRADED, "E2E no new data"),
    FaultSpec("SEQ_GAP", Severity.WARNING, "message sequence gap"),
    FaultSpec("CMD_INVALID", Severity.CRITICAL, "control command invalid (non-finite)"),
    FaultSpec("CMD_CLAMPED", Severity.WARNING, "control command clamped"),
    FaultSpec("CMD_STALE", Severity.CRITICAL, "control command stale"),
    FaultSpec("TRAJ_INVALID", Severity.CRITICAL, "trajectory invalid"),
    FaultSpec("TRAJ_LIMIT", Severity.WARNING, "trajectory exceeds dynamic limits"),
    FaultSpec("STATE_INVALID", Severity.CRITICAL, "vehicle state invalid"),
    FaultSpec("DRV_INATTENTIVE", Severity.WARNING, "driver inattentive"),
    FaultSpec("DRV_HANDS_OFF", Severity.WARNING, "driver hands off wheel"),
    FaultSpec("DRV_MONITOR_LOST", Severity.DEGRADED, "driver monitoring input stale"),
    FaultSpec("DRV_UNRESPONSIVE", Severity.CRITICAL, "driver did not respond to takeover request", latching=True),
    FaultSpec("AST_FAILED", Severity.DEGRADED, "runtime assertion failed"),
    FaultSpec("REC_EXHAUSTED", Severity.CRITICAL, "recovery attempts exhausted", latching=True),
    FaultSpec("SYS_HEALTH_LOW", Severity.DEGRADED, "system health score low"),
    FaultSpec("SYS_FATAL", Severity.FATAL, "fatal system fault", latching=True),
    FaultSpec("UNKNOWN", Severity.CRITICAL, "unclassified fault"),
)


class FaultRegistry:
    """Code -> FaultSpec mapping. Codes are case-sensitive, upper-snake."""

    def __init__(self, specs: Iterable[FaultSpec] = _DEFAULT_SPECS, unknown_severity: Severity = Severity.CRITICAL) -> None:
        self._specs: dict[str, FaultSpec] = {}
        self.unknown_severity = unknown_severity
        for s in specs:
            self.register(s)

    def register(self, spec: FaultSpec, overwrite: bool = False) -> None:
        if spec.code in self._specs and not overwrite:
            raise KeyError(f"fault code already registered: {spec.code}")
        self._specs[spec.code] = spec

    def get(self, code: str) -> FaultSpec:
        spec = self._specs.get(code)
        if spec is None:
            return FaultSpec(code, self.unknown_severity, "unregistered fault code")
        return spec

    def __contains__(self, code: str) -> bool:
        return code in self._specs

    def codes(self) -> list[str]:
        return sorted(self._specs)

    def make(self, code: str, source: str, stamp: float, message: str = "",
             severity: Optional[Severity] = None, **data: Any) -> Fault:
        """Build a `core.Fault` with the registry's default severity unless overridden."""
        spec = self.get(code)
        sev = spec.severity if severity is None else Severity(severity)
        if code not in self._specs:
            data.setdefault("unregistered", True)
        return Fault(code=code, severity=sev, source=source, stamp=float(stamp),
                     message=message or spec.description, data=dict(data))


DEFAULT_REGISTRY = FaultRegistry()


def make_fault(code: str, source: str, stamp: float, message: str = "",
               severity: Optional[Severity] = None, registry: Optional[FaultRegistry] = None,
               **data: Any) -> Fault:
    """Convenience wrapper around `DEFAULT_REGISTRY.make`."""
    return (registry or DEFAULT_REGISTRY).make(code, source, stamp, message, severity, **data)


# --------------------------------------------------------------------------- classifier

Rule = Callable[[Any], Optional[str]]


@dataclass
class FaultClassifierConfig:
    # Substring (lowercase) -> code. First match wins; checked in order.
    keyword_rules: tuple[tuple[str, str], ...] = (
        ("watchdog", "WDG_TIMEOUT"),
        ("heartbeat", "HB_STALE"),
        ("timeout", "WDG_TIMEOUT"),
        ("stale", "SNS_STALE"),
        ("nan", "SNS_NONFINITE"),
        ("inf", "SNS_NONFINITE"),
        ("frozen", "SNS_FROZEN"),
        ("range", "SNS_RANGE"),
        ("crc", "E2E_CRC"),
        ("checksum", "E2E_CRC"),
        ("sequence", "E2E_SEQUENCE"),
        ("counter", "E2E_SEQUENCE"),
        ("shape", "MDL_SCHEMA"),
        ("schema", "MDL_SCHEMA"),
        ("disagree", "RED_DISAGREE"),
        ("driver", "DRV_INATTENTIVE"),
    )


class FaultClassifier:
    """Maps raw fault evidence to a `core.Fault`.

    Accepted raw inputs:
      * `core.Fault`: returned unchanged.
      * `str`: a registered code, or free text matched against keyword rules.
      * `BaseException`: mapped by exception type, then by message keywords.
      * `dict`: with optional keys code/message/severity/data.
    Anything unmatched becomes `UNKNOWN` (CRITICAL by default - conservative).

    # TODO(safety): replace keyword heuristics with explicit, reviewed mapping tables
    # per producer (ISO 26262 FMEA-derived); free-text classification is a fallback only.
    """

    _EXC_MAP: tuple[tuple[type, str], ...] = (
        (TimeoutError, "WDG_TIMEOUT"),
        (FloatingPointError, "SNS_NONFINITE"),
        (OverflowError, "SNS_RANGE"),
        (ZeroDivisionError, "AST_FAILED"),
        (AssertionError, "AST_FAILED"),
        (MemoryError, "SYS_FATAL"),
    )

    def __init__(self, registry: Optional[FaultRegistry] = None,
                 config: Optional[FaultClassifierConfig] = None) -> None:
        self.registry = registry or DEFAULT_REGISTRY
        self.config = config or FaultClassifierConfig()
        self.rules: list[Rule] = []

    def add_rule(self, rule: Rule) -> None:
        """Custom rule: raw -> code or None. Custom rules run before built-ins."""
        self.rules.append(rule)

    def _code_from_text(self, text: str) -> Optional[str]:
        if text in self.registry:
            return text
        low = text.lower()
        for kw, code in self.config.keyword_rules:
            if kw in low:
                return code
        return None

    def classify(self, raw: Any, source: str, stamp: float) -> Fault:
        if isinstance(raw, Fault):
            return raw
        for rule in self.rules:
            code = rule(raw)
            if code:
                return self.registry.make(code, source, stamp, str(raw))
        if isinstance(raw, BaseException):
            for exc_t, code in self._EXC_MAP:
                if isinstance(raw, exc_t):
                    return self.registry.make(code, source, stamp, f"{type(raw).__name__}: {raw}",
                                              exception=type(raw).__name__)
            code = self._code_from_text(str(raw)) or "UNKNOWN"
            return self.registry.make(code, source, stamp, f"{type(raw).__name__}: {raw}",
                                      exception=type(raw).__name__)
        if isinstance(raw, dict):
            code = raw.get("code") or self._code_from_text(str(raw.get("message", ""))) or "UNKNOWN"
            sev = raw.get("severity")
            return self.registry.make(code, source, stamp, str(raw.get("message", "")),
                                      None if sev is None else Severity(sev), **dict(raw.get("data", {})))
        if isinstance(raw, str):
            code = self._code_from_text(raw) or "UNKNOWN"
            return self.registry.make(code, source, stamp, raw if code != raw else "")
        return self.registry.make("UNKNOWN", source, stamp, repr(raw))


# --------------------------------------------------------------------------- active table

@dataclass
class ActiveFault:
    fault: Fault
    first_seen: float
    last_seen: float
    count: int = 1
    latched: bool = False


@dataclass
class FaultTableConfig:
    heal_time_s: float = 1.0  # non-latching fault clears after this long without re-report
    history_len: int = 1000


class FaultTable:
    """Set of currently active faults keyed by (code, source).

    Re-reporting refreshes `last_seen`; severity can only rise while active
    (a fault never silently downgrades). Latching faults require `reset()`.
    """

    def __init__(self, config: Optional[FaultTableConfig] = None,
                 registry: Optional[FaultRegistry] = None) -> None:
        self.config = config or FaultTableConfig()
        self.registry = registry or DEFAULT_REGISTRY
        self._active: dict[tuple[str, str], ActiveFault] = {}
        self.history: deque[Fault] = deque(maxlen=self.config.history_len)

    def report(self, fault: Fault) -> ActiveFault:
        key = (fault.code, fault.source)
        self.history.append(fault)
        latching = self.registry.get(fault.code).latching
        af = self._active.get(key)
        if af is None:
            af = ActiveFault(fault, fault.stamp, fault.stamp, 1, latching)
            self._active[key] = af
        else:
            af.count += 1
            af.last_seen = max(af.last_seen, fault.stamp)
            if fault.severity >= af.fault.severity:
                af.fault = fault
        return af

    def report_all(self, faults: Iterable[Fault]) -> None:
        for f in faults:
            self.report(f)

    def update(self, t: float) -> list[Fault]:
        """Heal non-latching faults not seen for `heal_time_s`. Returns healed faults."""
        healed = []
        for key, af in list(self._active.items()):
            if not af.latched and t - af.last_seen >= self.config.heal_time_s:
                healed.append(af.fault)
                del self._active[key]
        return healed

    def clear(self, code: str, source: Optional[str] = None) -> int:
        """Clear a non-latching fault immediately (e.g. monitor confirmed healthy)."""
        n = 0
        for key, af in list(self._active.items()):
            if key[0] == code and (source is None or key[1] == source) and not af.latched:
                del self._active[key]
                n += 1
        return n

    def reset(self, code: Optional[str] = None, source: Optional[str] = None) -> int:
        """Explicit reset: clears latching faults too. Use only from an operator/maintenance path."""
        n = 0
        for key in list(self._active):
            if (code is None or key[0] == code) and (source is None or key[1] == source):
                del self._active[key]
                n += 1
        return n

    def active(self) -> list[Fault]:
        return [af.fault for af in self._active.values()]

    def entries(self) -> list[ActiveFault]:
        return list(self._active.values())

    def is_active(self, code: str, source: Optional[str] = None) -> bool:
        return any(k[0] == code and (source is None or k[1] == source) for k in self._active)

    def worst_severity(self) -> Severity:
        return max((af.fault.severity for af in self._active.values()), default=Severity.INFO)

    def __len__(self) -> int:
        return len(self._active)
