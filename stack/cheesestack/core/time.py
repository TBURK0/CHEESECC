"""Time sources and stamped messages.

All timestamps in the stack are float seconds on a monotonic timeline.
Use a Clock instead of time.time() so simulation and replay can drive time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


class Clock:
    """Abstract monotonic clock (seconds)."""

    def now(self) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class WallClock(Clock):
    def now(self) -> float:
        return time.monotonic()


class SimClock(Clock):
    """Manually advanced clock for simulation, replay and tests."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)

    def now(self) -> float:
        return self._t

    def advance(self, dt: float) -> float:
        if dt < 0:
            raise ValueError("SimClock cannot go backwards")
        self._t += dt
        return self._t

    def set(self, t: float) -> None:
        if t < self._t:
            raise ValueError("SimClock cannot go backwards")
        self._t = float(t)


@dataclass(frozen=True)
class Header:
    """Metadata carried by every message."""

    stamp: float
    frame_id: str = "base_link"
    seq: int = 0
    source: str = ""


@dataclass(frozen=True)
class Stamped(Generic[T]):
    header: Header
    data: T

    @property
    def stamp(self) -> float:
        return self.header.stamp
