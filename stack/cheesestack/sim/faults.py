"""Fault injection on bus topics, scheduled by (sim) time.

`FaultInjectingBus` is a drop-in `core.MessageBus` whose `publish` passes each
message through the faults active on that topic at the current clock time:

  dropout  message is discarded (with probability `p`, default 1)
  delay    message is held for `delay` s and released by `pump()` (original stamp kept)
  noise    zero-mean Gaussian noise (`std`) added to numeric `fields`
  bias     constant `offset` added to numeric `fields`
  stuck    the first message seen in the window is re-published instead of new ones
  corrupt  with probability `p`, a random one of `fields` becomes NaN / scaled by `scale`

Fields are dotted attribute paths ("pose.x", "v") or dict keys; frozen
dataclasses are rebuilt with `dataclasses.replace`, so the original message is
never mutated. numpy array fields are perturbed element-wise.

Every injection is logged as a `core.Fault` (INFO severity, ground truth for
evaluating the safety layer's detectors) in `bus.injection_log` and optionally
published on `log_topic`.

# TODO(sim): sensor-internal faults (calibration drift, partial FOV loss),
# clock skew between nodes, bus-wide latency spikes, fault trees / FMEA linkage.
"""

from __future__ import annotations

import copy
import dataclasses
import heapq
import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from cheesestack.core import Clock, Fault, MessageBus, Severity, SimClock

FAULT_KINDS = ("dropout", "delay", "noise", "bias", "stuck", "corrupt")


@dataclass
class FaultSpec:
    kind: str
    topic: str
    t_start: float = 0.0
    t_end: float = math.inf
    fields: tuple[str, ...] = ()
    p: float = 1.0
    std: float = 0.0
    offset: float = 0.0
    delay: float = 0.0
    scale: float = 1e6
    name: str = ""

    def __post_init__(self) -> None:
        if self.kind not in FAULT_KINDS:
            raise ValueError(f"unknown fault kind {self.kind!r}; expected one of {FAULT_KINDS}")

    def active(self, t: float) -> bool:
        return self.t_start <= t < self.t_end


# ----------------------------------------------------------- field helpers
def get_field(msg: Any, path: str) -> Any:
    obj = msg
    for part in path.split("."):
        obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
    return obj


def set_field(msg: Any, path: str, value: Any) -> Any:
    """Return a copy of msg with `path` replaced (works for frozen dataclasses and dicts)."""
    head, _, rest = path.partition(".")
    if isinstance(msg, dict):
        child = msg[head]
        new = dict(msg)
        new[head] = set_field(child, rest, value) if rest else value
        return new
    child = getattr(msg, head)
    new_val = set_field(child, rest, value) if rest else value
    if dataclasses.is_dataclass(msg):
        return dataclasses.replace(msg, **{head: new_val})
    out = copy.copy(msg)
    setattr(out, head, new_val)
    return out


def _perturb(value: Any, fn) -> Any:
    if isinstance(value, np.ndarray):
        return fn(value.astype(float))
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return float(fn(np.array(float(value))))
    return value


# ------------------------------------------------------------------ the bus
class FaultInjectingBus(MessageBus):
    def __init__(self, clock: Clock, faults: tuple[FaultSpec, ...] | list[FaultSpec] = (), seed: int = 0,
                 history: int = 1, log_topic: Optional[str] = None) -> None:
        super().__init__(history=history)
        self.clock = clock
        self.faults: list[FaultSpec] = list(faults)
        self.rng = np.random.default_rng(seed)
        self.log_topic = log_topic
        self.injection_log: list[Fault] = []
        self._delayed: list[tuple[float, int, str, Any]] = []
        self._counter = itertools.count()
        self._stuck: dict[int, Any] = {}

    def add_fault(self, spec: FaultSpec) -> None:
        self.faults.append(spec)

    def _log(self, spec: FaultSpec, t: float, topic: str) -> None:
        f = Fault(code=f"SIM_INJECT_{spec.kind.upper()}", severity=Severity.INFO, source="sim.faults", stamp=t,
                  message=spec.name or f"{spec.kind} on {topic}", data={"topic": topic})
        self.injection_log.append(f)
        if self.log_topic and topic != self.log_topic:
            MessageBus.publish(self, self.log_topic, f)

    def publish(self, topic: str, msg: Any) -> None:
        t = self.clock.now()
        delay = 0.0
        for i, spec in enumerate(self.faults):
            if spec.topic not in (topic, "*") or topic == self.log_topic:
                continue
            if not spec.active(t):
                self._stuck.pop(i, None)
                continue
            k = spec.kind
            if k == "dropout":
                if self.rng.random() < spec.p:
                    self._log(spec, t, topic)
                    return
            elif k == "stuck":
                if i not in self._stuck:
                    self._stuck[i] = msg
                else:
                    msg = self._stuck[i]
                    self._log(spec, t, topic)
            elif k == "delay":
                delay += spec.delay
                self._log(spec, t, topic)
            elif k == "noise":
                for fp in spec.fields:
                    msg = set_field(msg, fp, _perturb(get_field(msg, fp),
                                                      lambda v, s=spec: v + self.rng.normal(0.0, s.std, v.shape)))
                self._log(spec, t, topic)
            elif k == "bias":
                for fp in spec.fields:
                    msg = set_field(msg, fp, _perturb(get_field(msg, fp), lambda v, s=spec: v + s.offset))
                self._log(spec, t, topic)
            elif k == "corrupt":
                if spec.fields and self.rng.random() < spec.p:
                    fp = spec.fields[int(self.rng.integers(len(spec.fields)))]
                    if self.rng.random() < 0.5:
                        fn = lambda v: np.full_like(v, np.nan)
                    else:
                        fn = lambda v, s=spec: v * s.scale + s.scale
                    msg = set_field(msg, fp, _perturb(get_field(msg, fp), fn))
                    self._log(spec, t, topic)
        if delay > 0:
            heapq.heappush(self._delayed, (t + delay, next(self._counter), topic, msg))
            return
        super().publish(topic, msg)

    def pump(self, t: Optional[float] = None) -> int:
        """Deliver delayed messages whose release time has passed. Call once per sim tick."""
        t = self.clock.now() if t is None else t
        n = 0
        while self._delayed and self._delayed[0][0] <= t + 1e-12:
            _, _, topic, msg = heapq.heappop(self._delayed)
            MessageBus.publish(self, topic, msg)
            n += 1
        return n

    @property
    def pending(self) -> int:
        return len(self._delayed)


def fault_schedule(*specs: FaultSpec) -> list[FaultSpec]:
    """Convenience: sorted schedule by start time."""
    return sorted(specs, key=lambda s: s.t_start)
