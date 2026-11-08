"""In-process pub/sub message bus. Synchronous, deterministic (good for sim/replay)."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable


Callback = Callable[[str, Any], None]


@dataclass
class Subscription:
    topic: str
    callback: Callback
    _bus: "MessageBus"

    def cancel(self) -> None:
        self._bus._unsubscribe(self)


class MessageBus:
    """Topic-based pub/sub. Topics are strings like "/perception/lanes".

    A subscriber to "*" receives everything (used by loggers/recorders).
    The last message per topic is cached and available via `latest()`.
    """

    def __init__(self, history: int = 1) -> None:
        self._subs: dict[str, list[Subscription]] = defaultdict(list)
        self._history: dict[str, deque] = defaultdict(lambda: deque(maxlen=max(1, history)))
        self.publish_count: dict[str, int] = defaultdict(int)

    def subscribe(self, topic: str, callback: Callback) -> Subscription:
        sub = Subscription(topic, callback, self)
        self._subs[topic].append(sub)
        return sub

    def _unsubscribe(self, sub: Subscription) -> None:
        if sub in self._subs.get(sub.topic, []):
            self._subs[sub.topic].remove(sub)

    def publish(self, topic: str, msg: Any) -> None:
        self._history[topic].append(msg)
        self.publish_count[topic] += 1
        for sub in list(self._subs.get(topic, [])) + list(self._subs.get("*", [])):
            sub.callback(topic, msg)

    def latest(self, topic: str, default: Any = None) -> Any:
        h = self._history.get(topic)
        return h[-1] if h else default

    def topics(self) -> list[str]:
        return sorted(self._history.keys())
