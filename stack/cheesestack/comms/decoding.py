"""Message decoding: CAN frames -> physical signal values using a DBC `Database`.

`decode_signals` is the pure function; `FrameDecoder` adds database lookup,
optional VAL_ label substitution, DLC checks, counter/checksum validation and
per-ID error statistics.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any, Optional

from .can import CanFrame
from .dbc import Database, MessageDef, SignalDef
from .serialization import checksum_sum_xor, extract_raw, to_signed


class DecodeError(ValueError):
    pass


def raw_to_physical(sig: SignalDef, raw: int) -> float:
    if sig.is_float:
        if sig.length == 32:
            return struct.unpack("<f", struct.pack("<I", raw))[0]
        return struct.unpack("<d", struct.pack("<Q", raw))[0]
    if sig.signed:
        raw = to_signed(raw, sig.length)
    return raw * sig.scale + sig.offset


def decode_signals(msg: MessageDef, data: bytes, decode_choices: bool = False,
                   allow_truncated: bool = False) -> dict[str, Any]:
    """Decode all signals that are present (respecting the multiplexer).

    `allow_truncated` decodes the signals that fit in a short payload instead
    of raising.
    """
    if len(data) < msg.length and not allow_truncated:
        raise DecodeError(f"{msg.name}: payload {len(data)} B < DLC {msg.length} B")
    mux = msg.multiplexer
    mux_val: Optional[int] = None
    if mux is not None:
        mux_val = extract_raw(data, mux)
    out: dict[str, Any] = {}
    for sig in msg.signals_for_mux(mux_val):
        try:
            raw = extract_raw(data, sig)
        except ValueError:
            if allow_truncated:
                continue
            raise
        if decode_choices and sig.choices:
            r = to_signed(raw, sig.length) if sig.signed else raw
            if r in sig.choices:
                out[sig.name] = sig.choices[r]
                continue
        val = raw_to_physical(sig, raw)
        if sig.scale == int(sig.scale) and sig.offset == int(sig.offset) and not sig.is_float:
            val = int(val)
        out[sig.name] = val
    return out


@dataclass
class DecodedMessage:
    name: str
    frame_id: int
    timestamp: float
    signals: dict[str, Any]
    valid: bool = True
    errors: list[str] = field(default_factory=list)


@dataclass
class DecoderConfig:
    decode_choices: bool = False
    check_counter: bool = True
    check_checksum: bool = True
    counter_signal: str = "Counter"
    checksum_signal: str = "Checksum"
    counter_modulo: int = 16
    allow_truncated: bool = False


@dataclass
class DecoderStats:
    decoded: int = 0
    unknown_id: int = 0
    dlc_errors: int = 0
    checksum_errors: int = 0
    counter_errors: int = 0


class FrameDecoder:
    """Stateful decoder: database lookup + integrity checks.

    Checksum check uses the CHEESECC rule (`serialization.checksum_sum_xor`).
    # TODO(comms): pluggable E2E profiles per message (from DBC attributes).
    """

    def __init__(self, db: Database, config: Optional[DecoderConfig] = None) -> None:
        self.db = db
        self.config = config or DecoderConfig()
        self.stats = DecoderStats()
        self._last_counter: dict[tuple[int, bool], int] = {}

    def decode(self, frame: CanFrame) -> Optional[DecodedMessage]:
        """Decode a frame; returns None for unknown IDs, error and remote frames."""
        if frame.is_error or frame.is_remote:
            return None
        msg = self.db.get_message_by_id(frame.arbitration_id, frame.is_extended)
        if msg is None:
            self.stats.unknown_id += 1
            return None
        cfg = self.config
        errors: list[str] = []
        if len(frame.data) < msg.length:
            self.stats.dlc_errors += 1
            if not cfg.allow_truncated:
                return DecodedMessage(msg.name, msg.frame_id, frame.timestamp, {}, False,
                                      [f"dlc {len(frame.data)} < {msg.length}"])
            errors.append("truncated")
        sigs = decode_signals(msg, frame.data, cfg.decode_choices, cfg.allow_truncated)
        names = {s.name for s in msg.signals}
        if cfg.check_checksum and cfg.checksum_signal in names and len(frame.data) >= msg.length:
            csig = msg.signal(cfg.checksum_signal)
            expected = checksum_sum_xor(frame.data[: msg.length], exclude_byte=csig.start_bit // 8)
            if extract_raw(frame.data, csig) != expected:
                self.stats.checksum_errors += 1
                errors.append("checksum")
        if cfg.check_counter and cfg.counter_signal in sigs:
            key = (frame.arbitration_id, frame.is_extended)
            c = int(sigs[cfg.counter_signal])
            last = self._last_counter.get(key)
            if last is not None and c != (last + 1) % cfg.counter_modulo:
                self.stats.counter_errors += 1
                errors.append(f"counter {last}->{c}")
            self._last_counter[key] = c
        self.stats.decoded += 1
        return DecodedMessage(msg.name, msg.frame_id, frame.timestamp, sigs, not errors, errors)

    def decode_many(self, frames) -> list[DecodedMessage]:
        return [d for f in frames if (d := self.decode(f)) is not None]
