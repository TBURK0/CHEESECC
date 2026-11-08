"""DBC processing: signal/message definitions and a parser for a real subset of
the Vector DBC format.

Supported statements:
  VERSION, BU_ (nodes), BO_ (messages, incl. extended IDs via bit 31),
  SG_ (signals incl. multiplexer switch `M` and multiplexed `mN`),
  VAL_ (value tables), CM_ (network/node/message/signal comments, multi-line),
  BA_DEF_ / BA_DEF_DEF_ / BA_ (attributes; `GenMsgCycleTime` drives timing
  validation), SIG_VALTYPE_ (IEEE float/double signals).

Bit numbering follows the DBC convention: `start_bit` is the LSB position for
Intel (`@1`) signals and the MSB position for Motorola (`@0`) signals, with bit
`b` living in byte `b // 8`, bit `b % 8` (0 = LSB of the byte).

# TODO(comms): extended multiplexing (SG_MUL_VAL_), signal groups, EV_, BO_TX_BU_,
#   environment variables, VAL_TABLE_ references, J1939 PGN helpers, DBC writer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Union

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_DBC = DATA_DIR / "cheesecc.dbc"


@dataclass
class SignalDef:
    name: str
    start_bit: int
    length: int
    little_endian: bool = True  # Intel (@1) when True, Motorola (@0) when False
    signed: bool = False
    scale: float = 1.0
    offset: float = 0.0
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    unit: str = ""
    receivers: list[str] = field(default_factory=list)
    is_multiplexer: bool = False
    multiplex_id: Optional[int] = None  # signal present only when switch == this value
    is_float: bool = False  # IEEE754 (length 32 or 64), via SIG_VALTYPE_
    choices: dict[int, str] = field(default_factory=dict)
    comment: str = ""

    def __post_init__(self) -> None:
        if self.length <= 0 or self.length > 64:
            raise ValueError(f"signal {self.name}: invalid length {self.length}")
        if self.scale == 0:
            raise ValueError(f"signal {self.name}: scale must be non-zero")

    @property
    def raw_min(self) -> int:
        return -(1 << (self.length - 1)) if self.signed else 0

    @property
    def raw_max(self) -> int:
        return (1 << (self.length - 1)) - 1 if self.signed else (1 << self.length) - 1

    def choice_value(self, label: str) -> int:
        for k, v in self.choices.items():
            if v == label:
                return k
        raise KeyError(f"signal {self.name}: no choice {label!r}")


@dataclass
class MessageDef:
    frame_id: int
    name: str
    length: int
    sender: str = ""
    is_extended: bool = False
    signals: list[SignalDef] = field(default_factory=list)
    comment: str = ""
    cycle_time_s: Optional[float] = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def is_fd(self) -> bool:
        return self.length > 8

    def signal(self, name: str) -> SignalDef:
        for s in self.signals:
            if s.name == name:
                return s
        raise KeyError(f"message {self.name}: no signal {name!r}")

    @property
    def multiplexer(self) -> Optional[SignalDef]:
        return next((s for s in self.signals if s.is_multiplexer), None)

    def signals_for_mux(self, mux_value: Optional[int]) -> list[SignalDef]:
        """Signals that are present for a given multiplexer switch value."""
        return [s for s in self.signals if s.multiplex_id is None or s.multiplex_id == mux_value]


@dataclass
class Database:
    messages: list[MessageDef] = field(default_factory=list)
    nodes: list[str] = field(default_factory=list)
    version: str = ""
    comment: str = ""
    attribute_defaults: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._reindex()

    def _reindex(self) -> None:
        self._by_id = {(m.frame_id, m.is_extended): m for m in self.messages}
        self._by_name = {m.name: m for m in self.messages}

    def add_message(self, msg: MessageDef) -> None:
        self.messages.append(msg)
        self._reindex()

    def get_message_by_id(self, frame_id: int, is_extended: Optional[bool] = None) -> Optional[MessageDef]:
        if is_extended is None:
            return self._by_id.get((frame_id, False)) or self._by_id.get((frame_id, True))
        return self._by_id.get((frame_id, is_extended))

    def get_message_by_name(self, name: str) -> MessageDef:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"no message named {name!r}") from None

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def cycle_times(self) -> dict[int, float]:
        """frame_id -> expected period (s) for messages with GenMsgCycleTime > 0."""
        return {m.frame_id: m.cycle_time_s for m in self.messages if m.cycle_time_s}


# --- parser ---------------------------------------------------------------

class DbcParseError(ValueError):
    pass


_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_BO_RE = re.compile(r"^BO_\s+(\d+)\s+(\w+)\s*:\s*(\d+)\s+(\w+)")
_SG_RE = re.compile(
    rf"^SG_\s+(\w+)\s*(M|m\d+)?\s*:\s*(\d+)\|(\d+)@([01])([+-])\s*"
    rf"\(\s*({_NUM})\s*,\s*({_NUM})\s*\)\s*\[\s*({_NUM})\s*\|\s*({_NUM})\s*\]\s*"
    r'"([^"]*)"\s*(.*)$'
)
_STR_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


def _statements(text: str) -> list[str]:
    """Split DBC text into logical statements.

    Most statements are one line; CM_/VAL_/BA_* statements end with ';' and may
    span lines (comments often contain newlines). Quoted strings are respected.
    """
    out: list[str] = []
    buf: list[str] = []
    in_str = False
    multiline_kw = ("CM_", "VAL_", "BA_", "BA_DEF_", "BA_DEF_DEF_", "SIG_VALTYPE_", "VAL_TABLE_")
    in_ns = False
    for line in text.splitlines():
        if not buf:
            stripped = line.strip()
            if in_ns:
                # NS_ block: indented symbol list (contains bare "CM_", "BA_" ...)
                if not stripped or line[:1] in (" ", "\t"):
                    continue
                in_ns = False
            if not stripped or stripped.startswith("//"):
                continue
            if stripped.startswith("NS_"):
                in_ns = True
                continue
            first = stripped.split(None, 1)[0]
            if first not in multiline_kw:
                out.append(stripped)
                continue
            buf.append(stripped)
        else:
            buf.append(line)
        joined = "\n".join(buf)
        # statement complete when an unquoted ';' terminates it
        in_str = False
        done = False
        i = 0
        while i < len(joined):
            c = joined[i]
            if c == "\\" and in_str:
                i += 2
                continue
            if c == '"':
                in_str = not in_str
            elif c == ";" and not in_str:
                done = True
            i += 1
        if done:
            out.append(joined)
            buf = []
    if buf:
        raise DbcParseError(f"unterminated statement: {buf[0][:60]!r}")
    return out


def _num(s: str) -> Union[int, float]:
    f = float(s)
    return int(f) if f.is_integer() and "." not in s and "e" not in s.lower() else f


def parse_dbc(text: str) -> Database:
    """Parse DBC text into a `Database`."""
    db = Database()
    current: Optional[MessageDef] = None
    msgs: dict[int, MessageDef] = {}  # raw DBC id (with bit 31) -> msg
    attr_defs: dict[str, str] = {}
    attr_values: list[tuple[str, int, Any]] = []
    float_sigs: list[tuple[int, str, int]] = []

    def msg_for(raw_id: int) -> MessageDef:
        if raw_id not in msgs:
            raise DbcParseError(f"reference to unknown message id {raw_id}")
        return msgs[raw_id]

    for stmt in _statements(text):
        kw = stmt.split(None, 1)[0].rstrip(":")
        if kw == "VERSION":
            m = _STR_RE.search(stmt)
            db.version = m.group(1) if m else ""
            current = None
        elif kw == "BU_":
            db.nodes = stmt.split(":", 1)[1].split() if ":" in stmt else []
            current = None
        elif kw == "BO_":
            m = _BO_RE.match(stmt)
            if not m:
                raise DbcParseError(f"bad BO_: {stmt!r}")
            raw_id = int(m.group(1))
            ext = bool(raw_id & 0x80000000)
            current = MessageDef(
                frame_id=raw_id & 0x1FFFFFFF, name=m.group(2), length=int(m.group(3)),
                sender=m.group(4), is_extended=ext,
            )
            msgs[raw_id] = current
            db.messages.append(current)
        elif kw == "SG_":
            if current is None:
                raise DbcParseError(f"SG_ outside of BO_: {stmt!r}")
            m = _SG_RE.match(stmt)
            if not m:
                raise DbcParseError(f"bad SG_: {stmt!r}")
            name, mux, start, length, order, sign, scale, offset, mn, mx, unit, rx = m.groups()
            sig = SignalDef(
                name=name, start_bit=int(start), length=int(length),
                little_endian=order == "1", signed=sign == "-",
                scale=float(scale), offset=float(offset),
                minimum=float(mn), maximum=float(mx), unit=unit,
                receivers=[r for r in re.split(r"[,\s]+", rx.strip()) if r and r != "Vector__XXX"],
                is_multiplexer=mux == "M",
                multiplex_id=int(mux[1:]) if mux and mux.startswith("m") else None,
            )
            if sig.minimum == 0 and sig.maximum == 0:
                sig.minimum = sig.maximum = None  # DBC "[0|0]" means unbounded
            current.signals.append(sig)
        elif kw == "VAL_":
            body = stmt[4:].strip().rstrip(";").strip()
            parts = body.split(None, 2)
            if len(parts) < 2 or not parts[0].isdigit():
                continue  # VAL_ for environment variables: unsupported
            msg = msg_for(int(parts[0]))
            sig = msg.signal(parts[1])
            rest = parts[2] if len(parts) > 2 else ""
            for v, label in re.findall(rf"({_NUM})\s+\"((?:[^\"\\]|\\.)*)\"", rest):
                sig.choices[int(float(v))] = label
        elif kw == "CM_":
            body = stmt[3:].strip()
            sm = _STR_RE.search(body)
            comment = sm.group(1).replace('\\"', '"') if sm else ""
            head = body[: sm.start()].split() if sm else body.split()
            if not head:
                db.comment = comment
            elif head[0] == "BO_":
                msg_for(int(head[1])).comment = comment
            elif head[0] == "SG_":
                msg_for(int(head[1])).signal(head[2]).comment = comment
            # BU_/EV_ comments are accepted and ignored.
        elif kw == "BA_DEF_":
            m = re.match(r'BA_DEF_\s+(?:(BU_|BO_|SG_|EV_)\s+)?"(\w+)"\s+(\w+)', stmt)
            if m:
                attr_defs[m.group(2)] = m.group(3)
        elif kw == "BA_DEF_DEF_":
            m = re.match(rf'BA_DEF_DEF_\s+"(\w+)"\s+(?:"([^"]*)"|({_NUM}))', stmt)
            if m:
                db.attribute_defaults[m.group(1)] = m.group(2) if m.group(2) is not None else _num(m.group(3))
        elif kw == "BA_":
            m = re.match(rf'BA_\s+"(\w+)"\s+(?:(BO_)\s+(\d+)\s+)?(?:"([^"]*)"|({_NUM}))', stmt)
            if m and m.group(2) == "BO_":
                val = m.group(4) if m.group(4) is not None else _num(m.group(5))
                attr_values.append((m.group(1), int(m.group(3)), val))
        elif kw == "SIG_VALTYPE_":
            m = re.match(r"SIG_VALTYPE_\s+(\d+)\s+(\w+)\s*:\s*(\d)", stmt)
            if m:
                float_sigs.append((int(m.group(1)), m.group(2), int(m.group(3))))
        # NS_, BS_, VAL_TABLE_, BO_TX_BU_, EV_, SG_MUL_VAL_, SIG_GROUP_ ...: ignored.
        # TODO(comms): support SG_MUL_VAL_ (extended multiplexing) and VAL_TABLE_.

    for name, raw_id, val in attr_values:
        if raw_id in msgs:
            msgs[raw_id].attributes[name] = val
    for raw_id, sname, vt in float_sigs:
        if raw_id in msgs and vt in (1, 2):
            sig = msgs[raw_id].signal(sname)
            if sig.length != (32 if vt == 1 else 64):
                raise DbcParseError(f"float signal {sname} must be {32 if vt == 1 else 64} bits")
            sig.is_float = True
    default_cycle = db.attribute_defaults.get("GenMsgCycleTime", 0)
    for msg in db.messages:
        ct = msg.attributes.get("GenMsgCycleTime", default_cycle)
        msg.cycle_time_s = float(ct) / 1000.0 if ct else None
        _validate_layout(msg)
    db._reindex()
    return db


def signal_bit_positions(sig: SignalDef) -> list[int]:
    """DBC bit positions occupied by a signal, LSB first."""
    if sig.little_endian:
        return list(range(sig.start_bit, sig.start_bit + sig.length))
    # Motorola: walk from MSB (start_bit) towards LSB in "sawtooth" order.
    pos = []
    b = sig.start_bit
    for _ in range(sig.length):
        pos.append(b)
        b = b - 1 if b % 8 else b + 15
    return pos[::-1]


def _validate_layout(msg: MessageDef) -> None:
    """Check signals fit in the frame and do not overlap (per mux branch)."""
    nbits = msg.length * 8
    occupied: dict[Optional[int], dict[int, str]] = {}
    for s in msg.signals:
        bits = signal_bit_positions(s)
        if min(bits) < 0 or max(bits) >= nbits:
            raise DbcParseError(f"{msg.name}.{s.name} does not fit in {msg.length} bytes")
        branch = occupied.setdefault(s.multiplex_id, {})
        for b in bits:
            other = branch.get(b) or (occupied.get(None, {}).get(b) if s.multiplex_id is not None else None)
            if other and other != s.name:
                raise DbcParseError(f"{msg.name}: {s.name} overlaps {other} at bit {b}")
            branch[b] = s.name


def load_dbc(path: Union[str, Path]) -> Database:
    return parse_dbc(Path(path).read_text(encoding="utf-8", errors="replace"))


def load_default_database() -> Database:
    """The CHEESECC drive-by-wire DBC shipped in `comms/data/cheesecc.dbc`."""
    return load_dbc(DEFAULT_DBC)
