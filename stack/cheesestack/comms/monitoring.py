"""Network monitoring and timing validation.

`TimingValidator` checks each cyclic CAN ID against an expected period (from
DBC `GenMsgCycleTime` or config): mean period deviation, jitter (std of
inter-arrival times), missed cycles, and RX timeouts (gap > `timeout_factor`
x period). `NetworkMonitor` adds an ID inventory (rates, DLC mismatches vs.
DBC, unknown IDs) and per-node liveness based on message senders.

Both emit `core.Fault`s so the safety layer can consume them unchanged.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from cheesestack.core import Fault, Severity
from .can import CanFrame
from .dbc import Database

IdKey = tuple[int, bool]  # (arbitration_id, is_extended)


@dataclass
class TimingConfig:
    expected_periods: dict[int, float] = field(default_factory=dict)  # frame_id -> s
    period_tolerance: float = 0.10  # |mean - T| / T
    jitter_tolerance: float = 0.25  # std(dt) / T
    timeout_factor: float = 3.0  # gap > factor * T -> RX timeout
    missed_cycle_factor: float = 1.5  # a gap > 1.5 T counts round(gap/T) - 1 missed cycles
    window: int = 50  # inter-arrival samples kept per ID
    min_samples: int = 5
    startup_grace_s: float = 1.0  # time before "never received" is reported
    timeout_severity: Severity = Severity.DEGRADED
    source: str = "comms.timing"


@dataclass
class IdTiming:
    frame_id: int
    expected_period: Optional[float]
    count: int = 0
    first_t: Optional[float] = None
    last_t: Optional[float] = None
    missed_cycles: int = 0
    max_gap: float = 0.0
    dts: deque = field(default_factory=lambda: deque(maxlen=50))

    @property
    def mean_period(self) -> float:
        return float(np.mean(self.dts)) if self.dts else float("nan")

    @property
    def jitter_std(self) -> float:
        return float(np.std(self.dts)) if len(self.dts) > 1 else 0.0

    @property
    def max_abs_jitter(self) -> float:
        """Largest |dt - T| in the window (or vs. the mean when T is unknown)."""
        if not self.dts:
            return 0.0
        ref = self.expected_period or self.mean_period
        return float(np.max(np.abs(np.asarray(self.dts) - ref)))


class TimingValidator:
    # TODO(comms): event-triggered / on-change send types (GenMsgSendType), per-ID
    #   tolerances from DBC attributes, hardware timestamps for sub-ms accuracy.

    def __init__(self, config: Optional[TimingConfig] = None, start_t: float = 0.0) -> None:
        self.config = config or TimingConfig()
        self.start_t = start_t
        self.ids: dict[int, IdTiming] = {
            fid: IdTiming(fid, T, dts=deque(maxlen=self.config.window))
            for fid, T in self.config.expected_periods.items()
        }

    @classmethod
    def from_database(cls, db: Database, start_t: float = 0.0, **overrides) -> "TimingValidator":
        cfg = TimingConfig(expected_periods=db.cycle_times(), **overrides)
        return cls(cfg, start_t)

    def observe(self, frame: CanFrame) -> None:
        if frame.is_error or frame.is_remote:
            return
        st = self.ids.get(frame.arbitration_id)
        if st is None:
            st = self.ids[frame.arbitration_id] = IdTiming(frame.arbitration_id, None,
                                                           dts=deque(maxlen=self.config.window))
        t = frame.timestamp
        if st.last_t is not None:
            dt = t - st.last_t
            st.dts.append(dt)
            st.max_gap = max(st.max_gap, dt)
            T = st.expected_period
            if T and dt > self.config.missed_cycle_factor * T:
                st.missed_cycles += int(round(dt / T)) - 1
        else:
            st.first_t = t
        st.last_t = t
        st.count += 1

    __call__ = observe

    def check(self, now: float) -> list[Fault]:
        c = self.config
        faults: list[Fault] = []
        for fid, st in sorted(self.ids.items()):
            T = st.expected_period
            if not T:
                continue
            data = {"id": fid, "expected_period": T}
            if st.last_t is None:
                if now - self.start_t > max(c.startup_grace_s, c.timeout_factor * T):
                    faults.append(Fault("CAN_RX_NEVER", c.timeout_severity, c.source, now,
                                        f"0x{fid:X} never received", data))
                continue
            gap = now - st.last_t
            if gap > c.timeout_factor * T:
                faults.append(Fault("CAN_RX_TIMEOUT", c.timeout_severity, c.source, now,
                                    f"0x{fid:X} silent for {gap * 1000:.0f} ms", {**data, "gap": gap}))
            if len(st.dts) >= c.min_samples:
                mean = st.mean_period
                if abs(mean - T) / T > c.period_tolerance:
                    faults.append(Fault("CAN_PERIOD_DEVIATION", Severity.WARNING, c.source, now,
                                        f"0x{fid:X} period {mean * 1000:.1f} ms vs {T * 1000:.1f} ms",
                                        {**data, "mean_period": mean}))
                jit = st.jitter_std
                if jit / T > c.jitter_tolerance:
                    faults.append(Fault("CAN_JITTER", Severity.WARNING, c.source, now,
                                        f"0x{fid:X} jitter {jit * 1000:.2f} ms", {**data, "jitter_std": jit}))
        return faults

    def report(self) -> dict[int, dict]:
        return {fid: {"count": s.count, "expected_period": s.expected_period, "mean_period": s.mean_period,
                      "jitter_std": s.jitter_std, "max_gap": s.max_gap, "missed_cycles": s.missed_cycles}
                for fid, s in sorted(self.ids.items())}


@dataclass
class NetworkMonitorConfig:
    rate_window_s: float = 1.0
    node_timeout_s: float = 0.5
    report_unknown_ids: bool = True
    source: str = "comms.network"


@dataclass
class IdInventory:
    frame_id: int
    is_extended: bool
    name: Optional[str]
    count: int = 0
    last_t: float = float("nan")
    last_data: bytes = b""
    dlc_mismatches: int = 0
    times: deque = field(default_factory=deque)


class NetworkMonitor:
    """ID inventory + node liveness + timing validation in one tap.

    # TODO(comms): data-plausibility checks (frozen payloads), gateway routing
    #   validation, multi-channel correlation.
    """

    def __init__(self, db: Optional[Database] = None, config: Optional[NetworkMonitorConfig] = None,
                 timing: Optional[TimingValidator] = None, start_t: float = 0.0) -> None:
        self.db = db
        self.config = config or NetworkMonitorConfig()
        self.timing = timing or (TimingValidator.from_database(db, start_t) if db else TimingValidator(start_t=start_t))
        self.start_t = start_t
        self.ids: dict[IdKey, IdInventory] = {}
        self.node_last_seen: dict[str, float] = {}
        self.unknown_ids: set[IdKey] = set()

    def observe(self, frame: CanFrame) -> None:
        if frame.is_error:
            return
        key = (frame.arbitration_id, frame.is_extended)
        msg = self.db.get_message_by_id(*key) if self.db else None
        inv = self.ids.get(key)
        if inv is None:
            inv = self.ids[key] = IdInventory(key[0], key[1], msg.name if msg else None)
            if self.db and msg is None:
                self.unknown_ids.add(key)
        inv.count += 1
        inv.last_t = frame.timestamp
        inv.last_data = frame.data
        inv.times.append(frame.timestamp)
        while inv.times and inv.times[0] <= frame.timestamp - self.config.rate_window_s:
            inv.times.popleft()
        if msg is not None:
            if not frame.is_remote and len(frame.data) < msg.length:
                inv.dlc_mismatches += 1
            if msg.sender:
                self.node_last_seen[msg.sender] = frame.timestamp
        self.timing.observe(frame)

    __call__ = observe

    def observe_many(self, frames: Iterable[CanFrame]) -> None:
        for f in frames:
            self.observe(f)

    def rate_hz(self, frame_id: int, is_extended: bool = False) -> float:
        inv = self.ids.get((frame_id, is_extended))
        return len(inv.times) / self.config.rate_window_s if inv else 0.0

    def node_alive(self, node: str, now: float) -> bool:
        t = self.node_last_seen.get(node)
        return t is not None and now - t <= self.config.node_timeout_s

    def check(self, now: float) -> list[Fault]:
        c = self.config
        faults = self.timing.check(now)
        if self.db:
            senders = sorted({m.sender for m in self.db.messages if m.sender and m.cycle_time_s})
            for node in senders:
                if now - self.start_t > c.node_timeout_s and not self.node_alive(node, now):
                    faults.append(Fault("CAN_NODE_SILENT", Severity.DEGRADED, c.source, now,
                                        f"node {node} silent", {"node": node}))
        for key, inv in sorted(self.ids.items()):
            if inv.dlc_mismatches:
                faults.append(Fault("CAN_DLC_MISMATCH", Severity.WARNING, c.source, now,
                                    f"0x{key[0]:X} short DLC x{inv.dlc_mismatches}", {"id": key[0]}))
        if c.report_unknown_ids and self.unknown_ids:
            faults.append(Fault("CAN_UNKNOWN_ID", Severity.INFO, c.source, now,
                                f"{len(self.unknown_ids)} IDs not in DBC",
                                {"ids": sorted(k[0] for k in self.unknown_ids)}))
        return faults

    def summary(self) -> list[dict]:
        return [{"id": k[0], "ext": k[1], "name": v.name, "count": v.count,
                 "rate_hz": len(v.times) / self.config.rate_window_s, "last_t": v.last_t}
                for k, v in sorted(self.ids.items())]
