"""Per-fault recovery strategies with rate limiting and exponential backoff.

A `RecoveryPolicy` per fault code says what to try:

  NONE      log only
  RETRY     call the registered handler again (e.g. re-request sensor data)
  RESET     reset/restart the faulty component (handler does the reset)
  DEGRADE   no repair attempt; request DEGRADED mode
  ESCALATE  no repair attempt; escalate immediately

Attempts are rate limited per (code, source): at most `max_attempts` inside a
sliding `window_s`, and consecutive attempts are separated by
`backoff_s * 2**(n-1)` (capped at `max_backoff_s`). When attempts are
exhausted, or the handler raises, the manager returns `escalate=True` and emits
REC_EXHAUSTED (latching, CRITICAL). Recovery never lowers a fault's severity by
itself - it only reports whether a repair was attempted / succeeded; the
reporting monitor must observe the healthy condition before the fault heals.

# TODO(safety): persist attempt counters across process restarts; coordinate
# recoveries that share a dependency (don't reset camera and perception at once).
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from cheesestack.core import Fault

from .faults import DEFAULT_REGISTRY, FaultRegistry


class RecoveryStrategy(enum.Enum):
    NONE = "none"
    RETRY = "retry"
    RESET = "reset"
    DEGRADE = "degrade"
    ESCALATE = "escalate"


@dataclass
class RecoveryPolicy:
    strategy: RecoveryStrategy = RecoveryStrategy.NONE
    max_attempts: int = 3
    window_s: float = 60.0
    backoff_s: float = 0.5
    max_backoff_s: float = 10.0


@dataclass
class RecoveryAction:
    fault: Fault
    strategy: RecoveryStrategy
    attempted: bool = False
    succeeded: bool = False
    escalate: bool = False
    request_degrade: bool = False
    reason: str = ""
    attempt: int = 0


Handler = Callable[[Fault], bool]


class _History:
    def __init__(self) -> None:
        self.attempts: deque[float] = deque()
        self.last_attempt: float = float("-inf")
        self.consecutive: int = 0
        self.exhausted: bool = False


class RecoveryManager:
    def __init__(self, registry: Optional[FaultRegistry] = None, source: str = "safety.recovery") -> None:
        self.registry = registry or DEFAULT_REGISTRY
        self.source = source
        self._policies: dict[str, RecoveryPolicy] = {}
        self._handlers: dict[str, Handler] = {}
        self._hist: dict[tuple[str, str], _History] = {}
        self.log: deque[RecoveryAction] = deque(maxlen=500)

    def set_policy(self, code: str, policy: RecoveryPolicy, handler: Optional[Handler] = None) -> None:
        self._policies[code] = policy
        if handler is not None:
            self._handlers[code] = handler

    def policy_for(self, code: str) -> RecoveryPolicy:
        if code in self._policies:
            return self._policies[code]
        hint = self.registry.get(code).recovery
        try:
            return RecoveryPolicy(RecoveryStrategy(hint))
        except ValueError:
            return RecoveryPolicy(RecoveryStrategy.ESCALATE)

    def handle(self, fault: Fault, t: float) -> RecoveryAction:
        pol = self.policy_for(fault.code)
        act = RecoveryAction(fault, pol.strategy)
        if pol.strategy == RecoveryStrategy.NONE:
            act.reason = "log only"
        elif pol.strategy == RecoveryStrategy.DEGRADE:
            act.request_degrade, act.reason = True, "policy: degrade"
        elif pol.strategy == RecoveryStrategy.ESCALATE:
            act.escalate, act.reason = True, "policy: escalate"
        else:
            self._attempt(fault, pol, act, t)
        self.log.append(act)
        return act

    def _attempt(self, fault: Fault, pol: RecoveryPolicy, act: RecoveryAction, t: float) -> None:
        key = (fault.code, fault.source)
        h = self._hist.setdefault(key, _History())
        while h.attempts and t - h.attempts[0] > pol.window_s:
            h.attempts.popleft()
        if h.exhausted or len(h.attempts) >= pol.max_attempts:
            h.exhausted = True
            act.escalate, act.reason = True, f"exhausted {pol.max_attempts} attempts in {pol.window_s}s"
            return
        if h.consecutive > 0:
            wait = min(pol.backoff_s * 2 ** (h.consecutive - 1), pol.max_backoff_s)
            if t - h.last_attempt < wait:
                act.reason = f"backoff ({wait:.2f}s)"
                act.request_degrade = True  # meanwhile operate degraded
                return
        handler = self._handlers.get(fault.code)
        h.attempts.append(t)
        h.last_attempt = t
        h.consecutive += 1
        act.attempted, act.attempt = True, h.consecutive
        if handler is None:
            act.reason = "no handler registered"
            act.request_degrade = True
            return
        try:
            ok = bool(handler(fault))
        except Exception as e:  # handler failure is itself a fault -> escalate
            act.escalate, act.reason = True, f"handler raised {type(e).__name__}: {e}"
            return
        act.succeeded = ok
        act.reason = "handler ok" if ok else "handler failed"
        if not ok:
            act.request_degrade = True

    def notify_healthy(self, code: str, source: str) -> None:
        """Monitor confirmed the fault condition is gone: reset backoff (not the window count)."""
        h = self._hist.get((code, source))
        if h is not None:
            h.consecutive = 0

    def reset(self, code: Optional[str] = None) -> None:
        """Explicit operator reset of attempt counters (including exhaustion)."""
        for key in list(self._hist):
            if code is None or key[0] == code:
                del self._hist[key]

    def exhausted_fault(self, act: RecoveryAction, t: float) -> Optional[Fault]:
        if not act.escalate:
            return None
        return self.registry.make("REC_EXHAUSTED", self.source, t,
                                  f"{act.fault.code}@{act.fault.source}: {act.reason}",
                                  fault_code=act.fault.code, fault_source=act.fault.source)
