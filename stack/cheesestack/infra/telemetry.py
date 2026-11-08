"""Channel-based telemetry sink with decimation and uplink batching.

A ``TelemetrySink`` accepts ``record(channel, value, t)`` calls. Each channel
has a ``ChannelSpec`` limiting its rate: either keep every N-th sample
(``decimate``) or at most ``max_rate_hz`` (time-based). Accepted samples are
buffered and ``flush()`` produces ``UplinkPacket``s whose encoded JSON payload
stays under ``max_packet_bytes`` (greedy packing, priority channels first).

The sink can also tap a ``MessageBus`` ("*" subscription) and pull a scalar via
an extractor per topic. Time comes from a ``core.Clock`` - never wall time.

TODO(infra): binary encoding (CBOR/protobuf), link-budget aware scheduling,
store-and-forward when the link is down, ack/retransmit (comms layer).
"""

from __future__ import annotations

import json
import zlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from cheesestack.core import Clock, MessageBus, SimClock


@dataclass
class ChannelSpec:
    decimate: int = 1  # keep every N-th sample
    max_rate_hz: Optional[float] = None  # additionally rate-limit by time
    priority: int = 0  # higher goes into packets first


@dataclass
class TelemetryConfig:
    max_buffer: int = 10_000  # samples; oldest dropped when exceeded
    max_packet_bytes: int = 1200  # uncompressed bound; fits a typical UDP MTU payload
    compress: bool = False
    default_channel: ChannelSpec = field(default_factory=ChannelSpec)


@dataclass
class TelemetrySample:
    channel: str
    t: float
    value: Any


@dataclass
class UplinkPacket:
    seq: int
    t_first: float
    t_last: float
    samples: list[TelemetrySample]
    payload: bytes

    @staticmethod
    def decode(payload: bytes, compressed: bool = False) -> list[TelemetrySample]:
        raw = zlib.decompress(payload) if compressed else payload
        obj = json.loads(raw.decode())
        return [TelemetrySample(c, t, v) for c, t, v in obj["s"]]


def _jsonable(v: Any) -> Any:
    if hasattr(v, "tolist"):
        return v.tolist()
    if hasattr(v, "item"):
        return v.item()
    return v


class TelemetrySink:
    def __init__(self, config: Optional[TelemetryConfig] = None, clock: Optional[Clock] = None) -> None:
        self.config = config or TelemetryConfig()
        self.clock = clock or SimClock()
        self.channels: dict[str, ChannelSpec] = {}
        self._count: dict[str, int] = {}
        self._last_t: dict[str, float] = {}
        self._buf: deque[TelemetrySample] = deque()
        self._seq = 0
        self.dropped = 0
        self.decimated = 0
        self.accepted = 0

    def add_channel(self, name: str, spec: Optional[ChannelSpec] = None) -> None:
        self.channels[name] = spec or ChannelSpec()

    def _spec(self, name: str) -> ChannelSpec:
        return self.channels.get(name, self.config.default_channel)

    def record(self, channel: str, value: Any, t: Optional[float] = None) -> bool:
        """Offer a sample; returns True if kept after decimation/rate limiting."""
        t = self.clock.now() if t is None else float(t)
        spec = self._spec(channel)
        n = self._count.get(channel, 0)
        self._count[channel] = n + 1
        if n % max(1, spec.decimate) != 0:
            self.decimated += 1
            return False
        if spec.max_rate_hz:
            last = self._last_t.get(channel)
            if last is not None and (t - last) < (1.0 / spec.max_rate_hz) - 1e-9:
                self.decimated += 1
                return False
        self._last_t[channel] = t
        self._buf.append(TelemetrySample(channel, t, _jsonable(value)))
        self.accepted += 1
        while len(self._buf) > self.config.max_buffer:
            self._buf.popleft()
            self.dropped += 1
        return True

    def attach(self, bus: MessageBus, extractors: dict[str, Callable[[Any], Any]]) -> None:
        """Record topic -> extractor(msg) for each topic in ``extractors``."""
        def cb(topic: str, msg: Any) -> None:
            fn = extractors.get(topic)
            if fn is not None:
                self.record(topic, fn(msg))
        bus.subscribe("*", cb)

    def pending(self) -> int:
        return len(self._buf)

    def _encode(self, samples: list[TelemetrySample]) -> bytes:
        raw = json.dumps({"s": [[s.channel, s.t, s.value] for s in samples]},
                         separators=(",", ":")).encode()
        return zlib.compress(raw) if self.config.compress else raw

    def flush(self, max_packets: Optional[int] = None) -> list[UplinkPacket]:
        """Drain the buffer into size-bounded packets (priority, then time order).

        Packing is greedy on the *uncompressed* size, so with ``compress=True``
        packets are conservatively small. A single sample bigger than the limit
        is sent alone (oversized).
        """
        order = sorted(self._buf, key=lambda s: (-self._spec(s.channel).priority, s.t))
        limit = self.config.max_packet_bytes
        packets: list[UplinkPacket] = []
        cur: list[TelemetrySample] = []
        cur_size = len(b'{"s":[]}')  # envelope bytes
        for s in order:
            if max_packets is not None and len(packets) >= max_packets:
                cur = []
                break
            # Uncompressed JSON size is additive: item bytes + 1 comma separator.
            item = len(json.dumps([s.channel, s.t, s.value], separators=(",", ":")).encode())
            add = item + (1 if cur else 0)
            if cur and cur_size + add > limit:
                packets.append(self._make_packet(cur))
                cur, cur_size, add = [], len(b'{"s":[]}'), item
            cur.append(s)
            cur_size += add
        if cur and (max_packets is None or len(packets) < max_packets):
            packets.append(self._make_packet(cur))
        sent = {id(s) for p in packets for s in p.samples}
        self._buf = deque(s for s in self._buf if id(s) not in sent)
        return packets

    def _make_packet(self, samples: list[TelemetrySample]) -> UplinkPacket:
        self._seq += 1
        ts = [s.t for s in samples]
        return UplinkPacket(self._seq, min(ts), max(ts), list(samples), self._encode(samples))
