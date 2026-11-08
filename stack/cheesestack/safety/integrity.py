"""Data integrity: CRC8 / CRC32, alive counters, sequence gap detection and an
E2E-protection wrapper (in the spirit of AUTOSAR E2E Profile 1/4, simplified).

CRCs
  * `crc8(data, poly=0x1D, init=0xFF, xorout=0xFF)` - SAE J1850 by default
    (check value for b"123456789" is 0x4B). `crc8_autosar` uses poly 0x2F (check 0xDF).
  * `crc32(data)` - IEEE 802.3 reflected, table driven (check 0xCBF43926;
    identical to zlib.crc32).

E2E frame layout (big endian):
    [crc (1 or 4 bytes)][counter (1 byte)][payload ...]
The CRC covers data_id (2 bytes) + counter + payload, so a frame from the
wrong sender (different data_id) fails the CRC even if bytes are intact.

`E2EChecker.check(frame)` returns an `E2EStatus`:
    OK, OK_SOME_LOST (gap within max_delta), REPEATED, WRONG_SEQUENCE (gap too
    large / backwards), ERROR (CRC / length), NO_NEW_DATA (frame is None).
After a WRONG_SEQUENCE or ERROR the checker needs `min_ok_to_resync`
consecutive OK frames before its `valid` flag is restored.

# TODO(safety): full AUTOSAR E2E profiles (P4/P5/P22 with length field), CRC
# selection per bus, hardware CRC offload, timeout state machine per signal.
"""

from __future__ import annotations

import enum
import struct
from dataclasses import dataclass, field
from typing import Optional

from cheesestack.core import Fault

from .faults import DEFAULT_REGISTRY, FaultRegistry


# --------------------------------------------------------------------------- CRC

def _crc8_table(poly: int) -> list[int]:
    tbl = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = ((c << 1) ^ poly) & 0xFF if c & 0x80 else (c << 1) & 0xFF
        tbl.append(c)
    return tbl


_CRC8_TABLES: dict[int, list[int]] = {}


def crc8(data: bytes, poly: int = 0x1D, init: int = 0xFF, xorout: int = 0xFF) -> int:
    tbl = _CRC8_TABLES.get(poly)
    if tbl is None:
        tbl = _CRC8_TABLES[poly] = _crc8_table(poly)
    c = init & 0xFF
    for b in bytes(data):
        c = tbl[c ^ b]
    return c ^ xorout


def crc8_autosar(data: bytes) -> int:
    return crc8(data, poly=0x2F, init=0xFF, xorout=0xFF)


def _crc32_table() -> list[int]:
    tbl = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ 0xEDB88320 if c & 1 else c >> 1
        tbl.append(c)
    return tbl


_CRC32_TABLE = _crc32_table()


def crc32(data: bytes, init: int = 0) -> int:
    """IEEE CRC-32. `init` is a previous crc32 result for incremental use (like zlib)."""
    c = (init ^ 0xFFFFFFFF) & 0xFFFFFFFF
    for b in bytes(data):
        c = _CRC32_TABLE[(c ^ b) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


# --------------------------------------------------------------------------- counters

class CounterStatus(enum.Enum):
    OK = "ok"
    FIRST = "first"
    REPEATED = "repeated"
    LOST = "lost"  # gap within tolerance
    WRONG_SEQUENCE = "wrong_sequence"


@dataclass
class AliveCounterConfig:
    modulo: int = 16  # 4-bit alive counter
    max_delta: int = 2  # accepted counter increment (1 = no loss allowed)


class AliveCounterMonitor:
    """Wrapping alive counter check (e.g. 4-bit CAN rolling counter)."""

    def __init__(self, config: Optional[AliveCounterConfig] = None) -> None:
        self.config = config or AliveCounterConfig()
        self.last: Optional[int] = None
        self.lost = 0
        self.errors = 0

    def check(self, counter: int) -> CounterStatus:
        m = self.config.modulo
        counter = int(counter)
        if not 0 <= counter < m:
            self.errors += 1
            return CounterStatus.WRONG_SEQUENCE
        if self.last is None:
            self.last = counter
            return CounterStatus.FIRST
        delta = (counter - self.last) % m
        if delta == 0:
            return CounterStatus.REPEATED
        if delta > self.config.max_delta:
            self.errors += 1
            self.last = counter  # resync to the new counter
            return CounterStatus.WRONG_SEQUENCE
        self.last = counter
        if delta == 1:
            return CounterStatus.OK
        self.lost += delta - 1
        return CounterStatus.LOST


@dataclass
class SequenceEvent:
    status: CounterStatus
    gap: int = 0  # number of missing seq numbers (LOST)


class SequenceGapDetector:
    """Unbounded sequence numbers (e.g. `core.Header.seq`): gaps, duplicates, reordering."""

    def __init__(self, reorder_window: int = 0) -> None:
        self.expected: Optional[int] = None
        self.reorder_window = reorder_window
        self.total_lost = 0
        self.duplicates = 0
        self.reordered = 0

    def observe(self, seq: int) -> SequenceEvent:
        seq = int(seq)
        if self.expected is None:
            self.expected = seq + 1
            return SequenceEvent(CounterStatus.FIRST)
        if seq == self.expected:
            self.expected += 1
            return SequenceEvent(CounterStatus.OK)
        if seq > self.expected:
            gap = seq - self.expected
            self.total_lost += gap
            self.expected = seq + 1
            return SequenceEvent(CounterStatus.LOST, gap)
        # seq < expected: duplicate of last, a late (reordered) packet, or a reset
        if seq == self.expected - 1:
            self.duplicates += 1
            return SequenceEvent(CounterStatus.REPEATED)
        if self.expected - seq <= self.reorder_window + 1:
            self.reordered += 1
            self.total_lost = max(0, self.total_lost - 1)
            return SequenceEvent(CounterStatus.OK)
        self.expected = seq + 1  # producer restarted
        return SequenceEvent(CounterStatus.WRONG_SEQUENCE)

    def faults(self, ev: SequenceEvent, source: str, t: float,
               registry: Optional[FaultRegistry] = None) -> list[Fault]:
        reg = registry or DEFAULT_REGISTRY
        if ev.status == CounterStatus.LOST:
            return [reg.make("SEQ_GAP", source, t, f"{ev.gap} message(s) lost", gap=ev.gap)]
        if ev.status == CounterStatus.WRONG_SEQUENCE:
            return [reg.make("E2E_SEQUENCE", source, t, "sequence reset / backwards jump")]
        return []


# --------------------------------------------------------------------------- E2E

class E2EStatus(enum.Enum):
    OK = "ok"
    OK_SOME_LOST = "ok_some_lost"
    REPEATED = "repeated"
    WRONG_SEQUENCE = "wrong_sequence"
    ERROR = "error"
    NO_NEW_DATA = "no_new_data"


@dataclass
class E2EConfig:
    data_id: int = 0x0001  # 16 bit
    crc_bits: int = 8  # 8 or 32
    counter_modulo: int = 256
    max_delta: int = 2
    min_ok_to_resync: int = 2
    max_no_new_data: int = 3  # consecutive NO_NEW_DATA before timeout fault


def _crc(cfg: E2EConfig, body: bytes) -> bytes:
    covered = struct.pack(">H", cfg.data_id & 0xFFFF) + body
    if cfg.crc_bits == 8:
        return bytes([crc8(covered)])
    if cfg.crc_bits == 32:
        return struct.pack(">I", crc32(covered))
    raise ValueError("crc_bits must be 8 or 32")


class E2EProtector:
    """Sender side: adds counter and CRC."""

    def __init__(self, config: Optional[E2EConfig] = None) -> None:
        self.config = config or E2EConfig()
        self.counter = 0

    def protect(self, payload: bytes) -> bytes:
        body = bytes([self.counter % self.config.counter_modulo]) + bytes(payload)
        self.counter = (self.counter + 1) % self.config.counter_modulo
        return _crc(self.config, body) + body


@dataclass
class E2EResult:
    status: E2EStatus
    payload: Optional[bytes] = None
    counter: Optional[int] = None

    @property
    def usable(self) -> bool:
        return self.status in (E2EStatus.OK, E2EStatus.OK_SOME_LOST)


class E2EChecker:
    """Receiver side: verifies CRC and counter progression."""

    def __init__(self, config: Optional[E2EConfig] = None) -> None:
        self.config = config or E2EConfig()
        self.last_counter: Optional[int] = None
        self.ok_streak = 0
        self.valid = False
        self.no_new_data = 0
        self.stats: dict[E2EStatus, int] = {s: 0 for s in E2EStatus}

    def check(self, frame: Optional[bytes]) -> E2EResult:
        res = self._check(frame)
        self.stats[res.status] += 1
        if res.status in (E2EStatus.OK, E2EStatus.OK_SOME_LOST):
            self.ok_streak += 1
            if self.ok_streak >= self.config.min_ok_to_resync:
                self.valid = True
        elif res.status in (E2EStatus.ERROR, E2EStatus.WRONG_SEQUENCE):
            self.ok_streak = 0
            self.valid = False
        if res.status == E2EStatus.NO_NEW_DATA:
            self.no_new_data += 1
        else:
            self.no_new_data = 0
        return res

    def _check(self, frame: Optional[bytes]) -> E2EResult:
        cfg = self.config
        if frame is None:
            return E2EResult(E2EStatus.NO_NEW_DATA)
        n = 1 if cfg.crc_bits == 8 else 4
        if len(frame) < n + 1:
            return E2EResult(E2EStatus.ERROR)
        crc, body = frame[:n], frame[n:]
        if _crc(cfg, body) != crc:
            return E2EResult(E2EStatus.ERROR)
        counter, payload = body[0], bytes(body[1:])
        if counter >= cfg.counter_modulo:
            return E2EResult(E2EStatus.ERROR)
        if self.last_counter is None:
            self.last_counter = counter
            return E2EResult(E2EStatus.OK, payload, counter)
        delta = (counter - self.last_counter) % cfg.counter_modulo
        if delta == 0:
            return E2EResult(E2EStatus.REPEATED, None, counter)
        self.last_counter = counter
        if delta == 1:
            return E2EResult(E2EStatus.OK, payload, counter)
        if delta <= cfg.max_delta:
            return E2EResult(E2EStatus.OK_SOME_LOST, payload, counter)
        return E2EResult(E2EStatus.WRONG_SEQUENCE, None, counter)

    def faults(self, res: E2EResult, source: str, t: float,
               registry: Optional[FaultRegistry] = None) -> list[Fault]:
        reg = registry or DEFAULT_REGISTRY
        code = {E2EStatus.ERROR: "E2E_CRC", E2EStatus.WRONG_SEQUENCE: "E2E_SEQUENCE",
                E2EStatus.REPEATED: "E2E_REPEATED"}.get(res.status)
        out = [reg.make(code, source, t, res.status.value)] if code else []
        if res.status == E2EStatus.NO_NEW_DATA and self.no_new_data >= self.config.max_no_new_data:
            out.append(reg.make("E2E_TIMEOUT", source, t, f"{self.no_new_data} cycles without data"))
        return out
