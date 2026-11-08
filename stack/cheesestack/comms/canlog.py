"""Logging frameworks: candump-format CAN logs and a structured event logger.

candump `-L` (log file) line format, as written/read by can-utils:

    (1436509052.249713) vcan0 044#2A366C2BBA          classic, 11-bit
    (1436509052.249713) vcan0 18FF0310#0102           classic, 29-bit (8 hex digits)
    (1436509052.249713) vcan0 123#R                   remote (optional length: 123#R4)
    (1436509052.249713) vcan0 123##1AABBCC...          CAN-FD, flags nibble (1=BRS, 2=ESI)
    (1436509052.249713) vcan0 20000080#0000000000000000  error frame (CAN_ERR_FLAG)

`StructuredLogger` writes JSON-lines records stamped with the stack clock so
logs replay deterministically alongside CAN traces.

# TODO(comms): Vector ASC/BLF and PCAP-ng (SocketCAN link type) writers; log rotation;
#   compression.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Iterable, Iterator, Optional, TextIO, Union

from cheesestack.core import Clock, Severity, SimClock
from .can import CAN_ERR_FLAG, CANFD_BRS, CANFD_ESI, CanFrame, EXT_ID_MAX, STD_ID_MAX

_LINE_RE = re.compile(r"^\((\d+(?:\.\d+)?)\)\s+(\S+)\s+([0-9A-Fa-f]+)#(#?)(\S*)\s*$")


def format_candump(frame: CanFrame, channel: Optional[str] = None) -> str:
    ch = channel or frame.channel or "can0"
    if frame.is_error:
        ident = f"{(frame.arbitration_id | CAN_ERR_FLAG):08X}"
    elif frame.is_extended:
        ident = f"{frame.arbitration_id:08X}"
    else:
        ident = f"{frame.arbitration_id:03X}"
    if frame.is_remote:
        body = "#R" + (str(frame.remote_dlc) if frame.remote_dlc else "")
    elif frame.is_fd:
        flags = (CANFD_BRS if frame.bitrate_switch else 0) | (CANFD_ESI if frame.error_state_indicator else 0)
        body = f"##{flags:X}{frame.data.hex().upper()}"
    else:
        body = "#" + frame.data.hex().upper()
    return f"({frame.timestamp:.6f}) {ch} {ident}{body}"


def parse_candump(line: str) -> CanFrame:
    m = _LINE_RE.match(line.strip())
    if not m:
        raise ValueError(f"not a candump -L line: {line!r}")
    ts, ch, ident, fd_mark, body = m.groups()
    raw_id = int(ident, 16)
    ext = len(ident) > 3
    err = ext and bool(raw_id & CAN_ERR_FLAG)
    arb = raw_id & (EXT_ID_MAX if ext else STD_ID_MAX)
    if fd_mark:
        if not body:
            raise ValueError(f"CAN-FD line without flags: {line!r}")
        flags = int(body[0], 16)
        return CanFrame(arb, bytes.fromhex(body[1:]), float(ts), is_extended=ext, is_fd=True,
                        bitrate_switch=bool(flags & CANFD_BRS), error_state_indicator=bool(flags & CANFD_ESI),
                        channel=ch)
    if body[:1] in ("R", "r"):
        return CanFrame(arb, b"", float(ts), is_extended=ext, is_remote=True, channel=ch,
                        remote_dlc=int(body[1:]) if body[1:] else 0)
    return CanFrame(arb, bytes.fromhex(body), float(ts), is_extended=ext and not (err and arb <= STD_ID_MAX),
                    is_error=err, channel=ch)


class CandumpWriter:
    """Append frames to a text stream / file in candump -L format.

    Usable directly as a `VirtualCanNetwork` tap: `net.add_tap(writer)`.
    """

    def __init__(self, sink: Union[str, Path, TextIO], channel: Optional[str] = None) -> None:
        self._own = not hasattr(sink, "write")
        self._f: TextIO = open(sink, "w", encoding="ascii") if self._own else sink  # type: ignore[arg-type]
        self.channel = channel
        self.count = 0

    def write(self, frame: CanFrame) -> None:
        self._f.write(format_candump(frame, self.channel) + "\n")
        self.count += 1

    __call__ = write

    def write_many(self, frames: Iterable[CanFrame]) -> None:
        for f in frames:
            self.write(f)

    def flush(self) -> None:
        self._f.flush()

    def close(self) -> None:
        if self._own:
            self._f.close()

    def __enter__(self) -> "CandumpWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class CandumpReader:
    """Iterate frames from a candump -L file / stream. Bad lines are skipped
    (counted in `errors`) unless `strict=True`."""

    def __init__(self, source: Union[str, Path, TextIO, Iterable[str]], strict: bool = False) -> None:
        self.source = source
        self.strict = strict
        self.errors = 0

    def _lines(self) -> Iterator[str]:
        if isinstance(self.source, (str, Path)) and not str(self.source).lstrip().startswith("("):
            with open(self.source, encoding="ascii", errors="replace") as f:
                yield from f
        elif isinstance(self.source, str):
            yield from self.source.splitlines()
        else:
            yield from self.source

    def __iter__(self) -> Iterator[CanFrame]:
        for line in self._lines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                yield parse_candump(line)
            except ValueError:
                if self.strict:
                    raise
                self.errors += 1

    def read_all(self) -> list[CanFrame]:
        return list(self)


# --- structured logger ----------------------------------------------------

@dataclass
class LogRecord:
    t: float
    level: Severity
    component: str
    event: str
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"t": self.t, "level": self.level.name, "component": self.component,
                "event": self.event, **self.fields}


@dataclass
class StructuredLoggerConfig:
    min_level: Severity = Severity.INFO
    ring_size: int = 10000  # records kept in memory for inspection / upload


class StructuredLogger:
    """Clock-stamped JSON-lines event logger with bound context.

    `log.bind(node="can_gw").warning("rx_timeout", id=0x100)` ->
    `{"t": 1.23, "level": "WARNING", "component": "comms", "event": "rx_timeout", "node": "can_gw", "id": 256}`

    # TODO(comms): bridge to stdlib `logging` and to the infra layer's log shipper;
    #   rate limiting of repeated events.
    """

    def __init__(self, component: str = "comms", clock: Optional[Clock] = None,
                 sink: Optional[IO[str]] = None, config: Optional[StructuredLoggerConfig] = None,
                 _context: Optional[dict] = None, _shared: Optional[dict] = None) -> None:
        self.component = component
        self.clock = clock or SimClock()
        self.config = config or StructuredLoggerConfig()
        self._context = dict(_context or {})
        if _shared is None:
            from collections import deque
            _shared = {"ring": deque(maxlen=self.config.ring_size), "sink": sink}
        self._shared = _shared

    def bind(self, **context: Any) -> "StructuredLogger":
        return StructuredLogger(self.component, self.clock, None, self.config,
                                {**self._context, **context}, self._shared)

    def log(self, level: Severity, event: str, **fields: Any) -> Optional[LogRecord]:
        if level < self.config.min_level:
            return None
        rec = LogRecord(self.clock.now(), Severity(level), self.component, event, {**self._context, **fields})
        self._shared["ring"].append(rec)
        sink = self._shared["sink"]
        if sink is not None:
            sink.write(json.dumps(rec.to_dict(), default=_json_default, sort_keys=False) + "\n")
        return rec

    def info(self, event: str, **f: Any):
        return self.log(Severity.INFO, event, **f)

    def warning(self, event: str, **f: Any):
        return self.log(Severity.WARNING, event, **f)

    def degraded(self, event: str, **f: Any):
        return self.log(Severity.DEGRADED, event, **f)

    def critical(self, event: str, **f: Any):
        return self.log(Severity.CRITICAL, event, **f)

    @property
    def records(self) -> list[LogRecord]:
        return list(self._shared["ring"])

    def query(self, event: Optional[str] = None, min_level: Severity = Severity.INFO, **match: Any) -> list[LogRecord]:
        return [r for r in self._shared["ring"]
                if (event is None or r.event == event) and r.level >= min_level
                and all(r.fields.get(k) == v for k, v in match.items())]


def _json_default(o: Any) -> Any:
    if isinstance(o, (bytes, bytearray)):
        return o.hex()
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "name"):
        return o.name
    return str(o)


def read_jsonl(source: Union[str, TextIO]) -> list[dict]:
    """Parse a JSON-lines log (path, text, or stream)."""
    if isinstance(source, str) and "\n" not in source and not source.lstrip().startswith("{"):
        with open(source, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    text = source if isinstance(source, str) else source.read()
    return [json.loads(l) for l in io.StringIO(text) if l.strip()]
