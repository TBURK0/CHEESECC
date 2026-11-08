"""Per-node deadline watchdog with staged escalation.

Each watched entry has a heartbeat source: either a `core.Node` (its
`heartbeat` attribute, updated on every successful `tick()`), or a manual
`kick(name, t)` for code that is not a Node. On `check(t)` the age of the last
heartbeat is compared against the entry's timeout:

    age <= timeout                       -> OK
    timeout < age <= escalate_factor*T   -> TIMEOUT  (WDG_TIMEOUT, DEGRADED)
    age > escalate_factor*T              -> EXPIRED  (WDG_EXPIRED, CRITICAL)
    node.status == FAILED                -> FAILED   (WDG_NODE_FAILED, CRITICAL)

A node that has never ticked (heartbeat == -inf) is given `startup_grace_s`
from the moment it was registered. Escalation callbacks (e.g. a restart hook)
fire once per transition into a worse state. Recovery to OK is reported, but
the fault table decides when the associated fault heals.

The watchdog reads the time it is given (`check(t)`); it never reads a wall clock.
# TODO(safety): run the watchdog in an independent process / on separate silicon
# (e.g. an MCU window watchdog) so a hung main process cannot also hang its monitor.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from cheesestack.core import Clock, Fault, Node, NodeStatus

from .faults import DEFAULT_REGISTRY, FaultRegistry


class WatchdogState(enum.IntEnum):
    OK = 0
    TIMEOUT = 1
    EXPIRED = 2
    FAILED = 3


@dataclass
class WatchdogConfig:
    default_timeout_s: float = 0.3
    escalate_factor: float = 3.0  # EXPIRED when age > escalate_factor * timeout
    startup_grace_s: float = 1.0
    timeout_from_rate_factor: float = 3.0  # timeout = factor / node.rate_hz when not given


@dataclass
class _Entry:
    name: str
    timeout_s: float
    registered_at: float
    node: Optional[Node] = None
    last_kick: float = float("-inf")
    state: WatchdogState = WatchdogState.OK
    on_escalate: Optional[Callable[[str, WatchdogState], None]] = None
    escalations: int = 0

    def heartbeat(self) -> float:
        hb = self.last_kick
        if self.node is not None:
            hb = max(hb, float(self.node.heartbeat))
        return hb


class Watchdog:
    def __init__(self, config: Optional[WatchdogConfig] = None, clock: Optional[Clock] = None,
                 registry: Optional[FaultRegistry] = None, source: str = "safety.watchdog") -> None:
        self.config = config or WatchdogConfig()
        self.clock = clock
        self.registry = registry or DEFAULT_REGISTRY
        self.source = source
        self._entries: dict[str, _Entry] = {}

    def _now(self, t: Optional[float]) -> float:
        if t is not None:
            return float(t)
        if self.clock is None:
            raise ValueError("Watchdog needs either t or a clock")
        return self.clock.now()

    def watch(self, node: Node, timeout_s: Optional[float] = None, t: Optional[float] = None,
              on_escalate: Optional[Callable[[str, WatchdogState], None]] = None) -> None:
        """Watch a `core.Node`. Default timeout derives from `node.rate_hz`."""
        if timeout_s is None:
            rate = getattr(node, "rate_hz", 0.0) or 0.0
            timeout_s = (self.config.timeout_from_rate_factor / rate) if rate > 0 else self.config.default_timeout_s
        self._add(_Entry(node.name, float(timeout_s), self._now(t), node=node, on_escalate=on_escalate))

    def watch_named(self, name: str, timeout_s: Optional[float] = None, t: Optional[float] = None,
                    on_escalate: Optional[Callable[[str, WatchdogState], None]] = None) -> None:
        """Watch a non-Node component that calls `kick(name, t)`."""
        self._add(_Entry(name, float(timeout_s or self.config.default_timeout_s), self._now(t),
                         on_escalate=on_escalate))

    def _add(self, e: _Entry) -> None:
        if e.timeout_s <= 0 or not math.isfinite(e.timeout_s):
            raise ValueError("watchdog timeout must be positive and finite")
        self._entries[e.name] = e

    def unwatch(self, name: str) -> None:
        self._entries.pop(name, None)

    def kick(self, name: str, t: float) -> None:
        e = self._entries[name]
        e.last_kick = max(e.last_kick, float(t))

    def state(self, name: str) -> WatchdogState:
        return self._entries[name].state

    def states(self) -> dict[str, WatchdogState]:
        return {n: e.state for n, e in self._entries.items()}

    def names(self) -> list[str]:
        return list(self._entries)

    def _evaluate(self, e: _Entry, t: float) -> tuple[WatchdogState, float]:
        if e.node is not None and e.node.status == NodeStatus.FAILED:
            return WatchdogState.FAILED, t - e.heartbeat()
        hb = e.heartbeat()
        if hb == float("-inf"):
            # Never ticked: measure from registration, plus startup grace.
            age = t - e.registered_at - self.config.startup_grace_s
            age = max(age, 0.0)
        else:
            age = t - hb
        if age <= e.timeout_s:
            return WatchdogState.OK, age
        if age <= self.config.escalate_factor * e.timeout_s:
            return WatchdogState.TIMEOUT, age
        return WatchdogState.EXPIRED, age

    def check(self, t: Optional[float] = None) -> list[Fault]:
        """Evaluate all entries. Returns faults for every entry not OK (re-reported each check)."""
        t = self._now(t)
        faults: list[Fault] = []
        for e in self._entries.values():
            new_state, age = self._evaluate(e, t)
            if new_state > e.state:
                e.escalations += 1
                if e.on_escalate is not None:
                    e.on_escalate(e.name, new_state)
            e.state = new_state
            if new_state == WatchdogState.OK:
                continue
            code = {WatchdogState.TIMEOUT: "WDG_TIMEOUT", WatchdogState.EXPIRED: "WDG_EXPIRED",
                    WatchdogState.FAILED: "WDG_NODE_FAILED"}[new_state]
            faults.append(self.registry.make(code, f"{self.source}/{e.name}", t,
                                             f"{e.name}: {new_state.name.lower()} (age {age:.3f}s, timeout {e.timeout_s:.3f}s)",
                                             node=e.name, age_s=age, timeout_s=e.timeout_s))
        return faults

    def health(self) -> float:
        """Fraction-style score in [0, 1]: OK=1, TIMEOUT=0.5, EXPIRED/FAILED=0."""
        if not self._entries:
            return 1.0
        w = {WatchdogState.OK: 1.0, WatchdogState.TIMEOUT: 0.5, WatchdogState.EXPIRED: 0.0, WatchdogState.FAILED: 0.0}
        return sum(w[e.state] for e in self._entries.values()) / len(self._entries)
