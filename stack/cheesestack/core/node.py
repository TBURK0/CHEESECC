"""Base class for a stack component that runs on a tick."""

from __future__ import annotations

import enum
from typing import Optional

from .bus import MessageBus
from .time import Clock, SimClock


class NodeStatus(enum.Enum):
    INIT = "init"
    OK = "ok"
    DEGRADED = "degraded"
    FAILED = "failed"
    STOPPED = "stopped"


class Node:
    """A component with a name, a clock, an optional bus, and a `step()`.

    Subclasses override `step(t)`; the orchestrator (sim or runtime) calls it
    at `rate_hz`. `heartbeat` is updated on every successful step so the safety
    layer's watchdog can detect stalls.
    """

    rate_hz: float = 10.0

    def __init__(self, name: str, bus: Optional[MessageBus] = None, clock: Optional[Clock] = None) -> None:
        self.name = name
        self.bus = bus or MessageBus()
        self.clock = clock or SimClock()
        self.status = NodeStatus.INIT
        self.heartbeat: float = float("-inf")
        self.step_count = 0

    def tick(self) -> None:
        t = self.clock.now()
        try:
            self.step(t)
            self.heartbeat = t
            self.step_count += 1
            if self.status == NodeStatus.INIT:
                self.status = NodeStatus.OK
        except Exception:
            self.status = NodeStatus.FAILED
            raise

    def step(self, t: float) -> None:  # pragma: no cover - interface
        raise NotImplementedError
