"""Replay engine: record bus traffic, save/load JSONL, replay on a SimClock.

Recording: `BusRecorder` subscribes to "*" on a `core.MessageBus` and stores
(stamp, topic, msg) using the supplied clock.

Serialization: `encode`/`decode` turn dataclasses (incl. frozen ones such as
`Pose2D`/`Header`), enums, numpy arrays, tuples and dicts into tagged JSON so
that core message types round-trip exactly. Decoding only instantiates classes
from modules under `allowed_prefixes` (default: "cheesestack.") - a log file can
never import arbitrary code.

Replay: `Replayer.play(bus, clock, speed)` advances the SimClock to each record's
stamp and publishes. A `sleep_fn` makes playback wall-clock paced at `speed`x
(speed = 2.0 -> twice real time); without one it runs as fast as possible.
`step_to(t)` gives incremental playback for interleaving with nodes under test.

# TODO(sim): binary/columnar format (MCAP) for images, indexing + seek, schema
# versioning/migration, compression, multi-file splits.
"""

from __future__ import annotations

import dataclasses
import enum
import importlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

import numpy as np

from cheesestack.core import Clock, MessageBus, SimClock


# ------------------------------------------------------------ serialization
def _qualname(obj_type: type) -> str:
    return f"{obj_type.__module__}:{obj_type.__qualname__}"


def encode(obj: Any) -> Any:
    """Convert a message to JSON-compatible tagged structures."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, enum.Enum):
        return {"__enum__": _qualname(type(obj)), "value": encode(obj.value)}
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if math.isfinite(f):
            return f
        return {"__float__": repr(f)}
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": encode(obj.tolist()), "dtype": str(obj.dtype), "shape": list(obj.shape)}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        fields = {f.name: encode(getattr(obj, f.name)) for f in dataclasses.fields(obj) if f.init}
        return {"__dataclass__": _qualname(type(obj)), "fields": fields}
    if isinstance(obj, tuple):
        return {"__tuple__": [encode(x) for x in obj]}
    if isinstance(obj, list):
        return [encode(x) for x in obj]
    if isinstance(obj, dict):
        if all(isinstance(k, str) for k in obj):
            return {"__dict__": {k: encode(v) for k, v in obj.items()}}
        return {"__items__": [[encode(k), encode(v)] for k, v in obj.items()]}
    raise TypeError(f"cannot serialize {type(obj).__name__}")


def _resolve(qual: str, allowed_prefixes: tuple[str, ...]) -> type:
    module, _, name = qual.partition(":")
    if not any(module == p.rstrip(".") or module.startswith(p) for p in allowed_prefixes):
        raise ValueError(f"refusing to load type from module {module!r}")
    obj: Any = importlib.import_module(module)
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def decode(data: Any, allowed_prefixes: tuple[str, ...] = ("cheesestack.",)) -> Any:
    if isinstance(data, list):
        return [decode(x, allowed_prefixes) for x in data]
    if not isinstance(data, dict):
        return data
    if "__enum__" in data:
        return _resolve(data["__enum__"], allowed_prefixes)(decode(data["value"], allowed_prefixes))
    if "__float__" in data:
        return float(data["__float__"])
    if "__ndarray__" in data:
        arr = np.array(decode(data["__ndarray__"], allowed_prefixes), dtype=data["dtype"])
        return arr.reshape(data["shape"])
    if "__dataclass__" in data:
        cls = _resolve(data["__dataclass__"], allowed_prefixes)
        return cls(**{k: decode(v, allowed_prefixes) for k, v in data["fields"].items()})
    if "__tuple__" in data:
        return tuple(decode(x, allowed_prefixes) for x in data["__tuple__"])
    if "__dict__" in data:
        return {k: decode(v, allowed_prefixes) for k, v in data["__dict__"].items()}
    if "__items__" in data:
        return {decode(k, allowed_prefixes): decode(v, allowed_prefixes) for k, v in data["__items__"]}
    raise ValueError(f"untagged object in log: {sorted(data)[:5]}")


# ------------------------------------------------------------------ records
@dataclass
class Record:
    stamp: float
    topic: str
    msg: Any


class BusRecorder:
    """Records every message on a bus ("*" subscription)."""

    def __init__(self, bus: MessageBus, clock: Clock, topics: Optional[Iterable[str]] = None) -> None:
        self.clock = clock
        self.records: list[Record] = []
        self._filter = set(topics) if topics is not None else None
        self._sub = bus.subscribe("*", self._on_msg)

    def _on_msg(self, topic: str, msg: Any) -> None:
        if self._filter is None or topic in self._filter:
            self.records.append(Record(self.clock.now(), topic, msg))

    def stop(self) -> None:
        self._sub.cancel()

    def save(self, path: str) -> int:
        return save_jsonl(path, self.records)


def save_jsonl(path: str, records: Iterable[Record]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({"t": r.stamp, "topic": r.topic, "msg": encode(r.msg)}, separators=(",", ":")))
            f.write("\n")
            n += 1
    return n


def load_jsonl(path: str, allowed_prefixes: tuple[str, ...] = ("cheesestack.",)) -> list[Record]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out.append(Record(float(d["t"]), d["topic"], decode(d["msg"], allowed_prefixes)))
    return out


# ------------------------------------------------------------------- replay
class Replayer:
    """Publishes recorded messages in stamp order onto a bus, driving a SimClock."""

    def __init__(self, records: list[Record], topics: Optional[Iterable[str]] = None,
                 topic_remap: Optional[dict[str, str]] = None) -> None:
        recs = [r for r in records if topics is None or r.topic in set(topics)]
        self.records = sorted(recs, key=lambda r: r.stamp)  # stable: preserves same-stamp order
        self.remap = topic_remap or {}
        self._i = 0

    @classmethod
    def from_file(cls, path: str, **kw) -> "Replayer":
        return cls(load_jsonl(path), **kw)

    @property
    def done(self) -> bool:
        return self._i >= len(self.records)

    @property
    def start_time(self) -> float:
        return self.records[0].stamp if self.records else 0.0

    def rewind(self) -> None:
        self._i = 0

    def step_to(self, t: float, bus: MessageBus, clock: Optional[SimClock] = None) -> int:
        """Publish all records with stamp <= t. Returns number published."""
        n = 0
        while self._i < len(self.records) and self.records[self._i].stamp <= t + 1e-12:
            r = self.records[self._i]
            if clock is not None and r.stamp > clock.now():
                clock.set(r.stamp)
            bus.publish(self.remap.get(r.topic, r.topic), r.msg)
            self._i += 1
            n += 1
        if clock is not None and t > clock.now():
            clock.set(t)
        return n

    def play(self, bus: MessageBus, clock: SimClock, speed: float = 1.0,
             sleep_fn: Optional[Callable[[float], None]] = None, until: Optional[float] = None) -> int:
        """Replay everything (or up to `until`). With `sleep_fn`, pace at `speed`x real time."""
        if speed <= 0:
            raise ValueError("speed must be > 0")
        n = 0
        prev = None
        while not self.done:
            r = self.records[self._i]
            if until is not None and r.stamp > until:
                break
            if sleep_fn is not None and prev is not None and r.stamp > prev:
                sleep_fn((r.stamp - prev) / speed)
            prev = r.stamp
            n += self.step_to(r.stamp, bus, clock)
        return n
