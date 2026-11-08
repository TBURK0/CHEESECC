"""Message serialization.

Two halves:

1. **CAN signal encoding**: pack physical signal values into frame payloads
   following DBC rules (Intel/Motorola byte order, signed two's complement,
   `raw = (phys - offset) / scale`, IEEE float signals, multiplexing), plus
   rolling-counter / checksum helpers and `DriveCommandCodec`, which maps the
   app's DriveCommand JSON fields (and `core.ControlCommand`) onto the
   CHEESECC DBC.

2. **Internal message serialization**: `MessageSerializer` turns stack
   dataclasses (core types, numpy arrays, enums, nested containers) into a
   compact self-describing binary format or JSON, and back.
"""

from __future__ import annotations

import dataclasses
import enum
import io
import json
import math
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Union

import numpy as np

from cheesestack import core
from .can import CanFrame, fd_padded_length
from .dbc import Database, MessageDef, SignalDef


class EncodeError(ValueError):
    pass


# --- bit-level helpers ----------------------------------------------------

def _motorola_msb_index(start_bit: int) -> int:
    """DBC Motorola start bit -> index in big-endian linear bit order (0 = MSB of byte 0)."""
    return (start_bit // 8) * 8 + (7 - start_bit % 8)


def insert_raw(data: bytearray, sig: SignalDef, raw: int) -> None:
    """Write an unsigned raw bit pattern for `sig` into `data` in place."""
    mask = (1 << sig.length) - 1
    raw &= mask
    nbytes = len(data)
    if sig.little_endian:
        word = int.from_bytes(data, "little")
        word &= ~(mask << sig.start_bit)
        word |= raw << sig.start_bit
        data[:] = word.to_bytes(nbytes, "little")
    else:
        nbits = nbytes * 8
        shift = nbits - (_motorola_msb_index(sig.start_bit) + sig.length)
        if shift < 0:
            raise EncodeError(f"signal {sig.name} does not fit in {nbytes} bytes")
        word = int.from_bytes(data, "big")
        word &= ~(mask << shift)
        word |= raw << shift
        data[:] = word.to_bytes(nbytes, "big")


def extract_raw(data: bytes, sig: SignalDef) -> int:
    """Read the unsigned raw bit pattern for `sig` from `data`."""
    mask = (1 << sig.length) - 1
    if sig.little_endian:
        if sig.start_bit + sig.length > len(data) * 8:
            raise ValueError(f"signal {sig.name} beyond payload ({len(data)} bytes)")
        return (int.from_bytes(data, "little") >> sig.start_bit) & mask
    shift = len(data) * 8 - (_motorola_msb_index(sig.start_bit) + sig.length)
    if shift < 0:
        raise ValueError(f"signal {sig.name} beyond payload ({len(data)} bytes)")
    return (int.from_bytes(data, "big") >> shift) & mask


def to_signed(raw: int, length: int) -> int:
    return raw - (1 << length) if raw & (1 << (length - 1)) else raw


def physical_to_raw(sig: SignalDef, value: Union[float, int, str], clamp: bool = True) -> int:
    """Convert a physical value (or a VAL_ label) into the raw bit pattern."""
    if isinstance(value, str):
        return sig.choice_value(value)
    if sig.is_float:
        fmt = "<f" if sig.length == 32 else "<d"
        ifmt = "<I" if sig.length == 32 else "<Q"
        return struct.unpack(ifmt, struct.pack(fmt, float(value)))[0]
    v = float(value)
    if not math.isfinite(v):
        raise EncodeError(f"signal {sig.name}: non-finite value {value}")
    if clamp:
        if sig.minimum is not None:
            v = max(v, sig.minimum)
        if sig.maximum is not None:
            v = min(v, sig.maximum)
    raw = int(round((v - sig.offset) / sig.scale))
    if raw < sig.raw_min or raw > sig.raw_max:
        if not clamp:
            raise EncodeError(f"signal {sig.name}: raw {raw} outside [{sig.raw_min}, {sig.raw_max}]")
        raw = min(max(raw, sig.raw_min), sig.raw_max)
    return raw & ((1 << sig.length) - 1)


@dataclass
class EncoderConfig:
    clamp: bool = True  # clamp to [min, max] and raw range instead of raising
    strict: bool = False  # require every (active) signal to be supplied
    fill_byte: int = 0x00
    counter_signal: str = "Counter"
    checksum_signal: str = "Checksum"


def encode_signals(msg: MessageDef, values: Mapping[str, Any], config: Optional[EncoderConfig] = None) -> bytes:
    """Encode a dict of physical values into a payload of `msg.length` bytes.

    Multiplexed messages: the multiplexer value selects which `mN` signals are
    written; supplying a multiplexed signal of another branch raises.
    """
    cfg = config or EncoderConfig()
    data = bytearray([cfg.fill_byte]) * msg.length
    mux = msg.multiplexer
    mux_val: Optional[int] = None
    if mux is not None:
        if mux.name not in values:
            raise EncodeError(f"{msg.name}: multiplexer {mux.name} value required")
        mux_val = physical_to_raw(mux, values[mux.name], clamp=cfg.clamp)
    active = {s.name: s for s in msg.signals_for_mux(mux_val)}
    for name in values:
        if name not in active:
            known = any(s.name == name for s in msg.signals)
            raise EncodeError(
                f"{msg.name}: signal {name!r} " + ("not active for mux value %s" % mux_val if known else "unknown")
            )
    for name, sig in active.items():
        if name in values:
            insert_raw(data, sig, physical_to_raw(sig, values[name], clamp=cfg.clamp))
        elif cfg.strict:
            raise EncodeError(f"{msg.name}: missing signal {name}")
        else:
            insert_raw(data, sig, physical_to_raw(sig, 0.0, clamp=True))  # physical 0, clamped
    return bytes(data)


def encode_frame(db: Database, name_or_msg: Union[str, MessageDef], values: Mapping[str, Any],
                 timestamp: float = 0.0, config: Optional[EncoderConfig] = None) -> CanFrame:
    msg = db.get_message_by_name(name_or_msg) if isinstance(name_or_msg, str) else name_or_msg
    data = encode_signals(msg, values, config)
    is_fd = msg.is_fd
    if is_fd:
        data = data.ljust(fd_padded_length(len(data)), b"\0")
    return CanFrame(msg.frame_id, data, timestamp=timestamp, is_extended=msg.is_extended, is_fd=is_fd)


# --- integrity helpers ----------------------------------------------------

def checksum_sum_xor(data: bytes, exclude_byte: int = 7) -> int:
    """CHEESECC checksum: (sum of all bytes except `exclude_byte`) & 0xFF, XOR 0xFF."""
    return (sum(b for i, b in enumerate(data) if i != exclude_byte) & 0xFF) ^ 0xFF


def crc8_sae_j1850(data: bytes, init: int = 0xFF, xor_out: int = 0xFF) -> int:
    """CRC-8 SAE-J1850 (poly 0x1D), as used by AUTOSAR E2E profile 1/2."""
    crc = init
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1D) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc ^ xor_out


class RollingCounter:
    def __init__(self, modulo: int = 16) -> None:
        self.modulo = modulo
        self.value = -1

    def next(self) -> int:
        self.value = (self.value + 1) % self.modulo
        return self.value


# --- DriveCommand mapping -------------------------------------------------

DRIVE_SOURCES = {"app": 0, "control": 1, "teleop": 2, "replay": 3, "sim": 4}


def car_id_hash(car_id: str, bits: int = 12) -> int:
    return zlib.crc32(car_id.encode("utf-8")) & ((1 << bits) - 1)


class DriveCommandCodec:
    """Maps the app's DriveCommand (JSON line / dict with camelCase keys) and
    `core.ControlCommand` onto CHEESECC DBC frames, adding rolling counters and
    checksums. DriveCommand.carId is carried as a 12-bit CRC hash only.

    # TODO(comms): E2E protection (AUTOSAR profile) instead of sum/xor checksum;
    #   full 64-bit command id; carId transport via ISO-TP.
    """

    def __init__(self, db: Database, config: Optional[EncoderConfig] = None) -> None:
        self.db = db
        self.config = config or EncoderConfig()
        self._counters: dict[str, RollingCounter] = {}

    def _finish(self, name: str, values: dict, t: float) -> CanFrame:
        msg = self.db.get_message_by_name(name)
        names = {s.name for s in msg.signals}
        if self.config.counter_signal in names:
            values[self.config.counter_signal] = self._counters.setdefault(name, RollingCounter(16)).next()
        frame = encode_frame(self.db, msg, values, t, self.config)
        if self.config.checksum_signal in names:
            sig = msg.signal(self.config.checksum_signal)
            data = bytearray(frame.data)
            insert_raw(data, sig, checksum_sum_xor(bytes(data), exclude_byte=sig.start_bit // 8))
            frame = dataclasses.replace(frame, data=bytes(data))
        return frame

    def encode_drive_command(self, cmd: Mapping[str, Any], t: float = 0.0, drive_mode: str = "Auto") -> list[CanFrame]:
        if isinstance(cmd, str):
            cmd = json.loads(cmd)
        frames = [
            self._finish("SteeringCommand", {
                "SteerWheelDeg": cmd.get("steerWheelDeg", 0.0),
                "Curvature": cmd.get("curvature", 0.0),
                "SteerEnable": 1,
            }, t),
            self._finish("SpeedCommand", {
                "TargetSpeedMps": cmd.get("targetSpeedMps", 0.0),
                "Throttle": cmd.get("throttle", 0.0),
                "Brake": cmd.get("brake", 0.0),
                "DriveMode": drive_mode,
            }, t),
            self._finish("DriveCommandMeta", {
                "CommandId": int(cmd.get("id", 0)) & 0xFFFFFFFF,
                "LookaheadS": cmd.get("lookaheadS", 0.0),
                "Source": DRIVE_SOURCES.get(str(cmd.get("source", "")), 15),
                "CarIdHash": car_id_hash(str(cmd.get("carId", ""))),
            }, t),
        ]
        for mux, key in enumerate(("latitude", "longitude", "fromLatitude", "fromLongitude")):
            if key in cmd:
                sig_name = key[0].upper() + key[1:]
                frames.append(self._finish("DriveCommandGeo", {"GeoMux": mux, sig_name: cmd[key]}, t))
        return frames

    def encode_control_command(self, cmd: core.ControlCommand) -> list[CanFrame]:
        t = cmd.header.stamp
        return [
            self._finish("SteeringCommand", {
                "SteerWheelDeg": cmd.steer_wheel_deg, "Curvature": cmd.curvature, "SteerEnable": 1}, t),
            self._finish("SpeedCommand", {
                "TargetSpeedMps": cmd.target_speed_mps, "Throttle": cmd.throttle,
                "Brake": cmd.brake, "DriveMode": "Auto"}, t),
        ]


# --- internal message serializer -----------------------------------------

class SerializeError(ValueError):
    pass


_MAGIC = b"CS"
_VERSION = 1
# type tags
_T_NONE, _T_FALSE, _T_TRUE, _T_INT, _T_FLOAT, _T_STR, _T_BYTES, _T_LIST, _T_DICT, _T_ND, _T_OBJ, _T_ENUM, _T_TUPLE = range(13)


def _zigzag(n: int) -> int:
    if not -(1 << 63) <= n < (1 << 63):
        raise SerializeError("int out of 64-bit range")
    return (n << 1) ^ (n >> 63)


def _write_varint(buf: io.BytesIO, n: int) -> None:
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            buf.write(bytes([b | 0x80]))
        else:
            buf.write(bytes([b]))
            return


def _read_varint(buf: io.BytesIO) -> int:
    shift = n = 0
    while True:
        c = buf.read(1)
        if not c:
            raise SerializeError("truncated varint")
        b = c[0]
        n |= (b & 0x7F) << shift
        if not b & 0x80:
            return n
        shift += 7


@dataclass
class SerializerConfig:
    float32: bool = False  # store python floats as f32 to save space (lossy)
    checksum: bool = True  # append CRC32 to binary blobs


class MessageSerializer:
    """Self-describing binary / JSON serializer for stack dataclasses.

    Supported: None, bool, int (64-bit, zigzag varint), float, str, bytes,
    list, tuple, dict (any serializable keys), numpy arrays, enums and
    dataclasses registered with `register()` (all `cheesestack.core` message
    types are pre-registered). Binary layout:
    `b"CS" | version u8 | value | [crc32 u32le]`.

    # TODO(comms): schema evolution (field defaults on missing keys work; renames
    #   do not), zero-copy ndarray views, versioned type ids instead of names.
    """

    def __init__(self, config: Optional[SerializerConfig] = None) -> None:
        self.config = config or SerializerConfig()
        self._types: dict[str, type] = {}
        for t in (core.Header, core.Pose2D, core.BoundingBox2D, core.BoundingBox3D, core.Detection,
                  core.TrackedObject, core.VehicleState, core.TrajectoryPoint, core.Trajectory,
                  core.ControlCommand, core.Fault, core.Stamped, core.ObjectClass, core.Severity):
            self.register(t)

    def register(self, cls: type, name: Optional[str] = None) -> type:
        if not (dataclasses.is_dataclass(cls) or (isinstance(cls, type) and issubclass(cls, enum.Enum))):
            raise SerializeError(f"{cls} is not a dataclass or Enum")
        self._types[name or cls.__name__] = cls
        return cls

    def _type_name(self, obj: Any) -> str:
        cls = type(obj)
        for k, v in self._types.items():
            if v is cls:
                return k
        raise SerializeError(f"unregistered type {cls.__name__}; call register()")

    # binary
    def dumps(self, obj: Any) -> bytes:
        buf = io.BytesIO()
        buf.write(_MAGIC + bytes([_VERSION]))
        self._enc(buf, obj)
        out = buf.getvalue()
        if self.config.checksum:
            out += struct.pack("<I", zlib.crc32(out))
        return out

    def loads(self, blob: bytes) -> Any:
        if blob[:2] != _MAGIC:
            raise SerializeError("bad magic")
        if blob[2] != _VERSION:
            raise SerializeError(f"unsupported version {blob[2]}")
        if self.config.checksum:
            body, (crc,) = blob[:-4], struct.unpack("<I", blob[-4:])
            if zlib.crc32(body) != crc:
                raise SerializeError("checksum mismatch")
        else:
            body = blob
        buf = io.BytesIO(body[3:])
        obj = self._dec(buf)
        if buf.read(1):
            raise SerializeError("trailing bytes")
        return obj

    def _enc_str(self, buf: io.BytesIO, s: str) -> None:
        b = s.encode("utf-8")
        _write_varint(buf, len(b))
        buf.write(b)

    def _enc(self, buf: io.BytesIO, obj: Any) -> None:
        w = buf.write
        if obj is None:
            w(bytes([_T_NONE]))
        elif obj is True or obj is False or isinstance(obj, np.bool_):
            w(bytes([_T_TRUE if obj else _T_FALSE]))
        elif isinstance(obj, enum.Enum):
            w(bytes([_T_ENUM]))
            self._enc_str(buf, self._type_name(obj))
            self._enc(buf, obj.value)
        elif isinstance(obj, (int, np.integer)):
            w(bytes([_T_INT]))
            _write_varint(buf, _zigzag(int(obj)))
        elif isinstance(obj, (float, np.floating)):
            if self.config.float32:
                w(bytes([_T_FLOAT, 4]) + struct.pack("<f", float(obj)))
            else:
                w(bytes([_T_FLOAT, 8]) + struct.pack("<d", float(obj)))
        elif isinstance(obj, str):
            w(bytes([_T_STR]))
            self._enc_str(buf, obj)
        elif isinstance(obj, (bytes, bytearray)):
            w(bytes([_T_BYTES]))
            _write_varint(buf, len(obj))
            w(bytes(obj))
        elif isinstance(obj, np.ndarray):
            if obj.dtype == object:
                raise SerializeError("object arrays not supported")
            w(bytes([_T_ND]))
            self._enc_str(buf, obj.dtype.str)
            _write_varint(buf, obj.ndim)
            for d in obj.shape:
                _write_varint(buf, d)
            raw = np.ascontiguousarray(obj).tobytes()
            _write_varint(buf, len(raw))
            w(raw)
        elif isinstance(obj, (list, tuple)):
            w(bytes([_T_TUPLE if isinstance(obj, tuple) else _T_LIST]))
            _write_varint(buf, len(obj))
            for x in obj:
                self._enc(buf, x)
        elif isinstance(obj, dict):
            w(bytes([_T_DICT]))
            _write_varint(buf, len(obj))
            for k, v in obj.items():
                self._enc(buf, k)
                self._enc(buf, v)
        elif dataclasses.is_dataclass(obj):
            w(bytes([_T_OBJ]))
            self._enc_str(buf, self._type_name(obj))
            fields = [f for f in dataclasses.fields(obj) if f.init]
            _write_varint(buf, len(fields))
            for f in fields:
                self._enc_str(buf, f.name)
                self._enc(buf, getattr(obj, f.name))
        else:
            raise SerializeError(f"cannot serialize {type(obj).__name__}")

    def _dec_str(self, buf: io.BytesIO) -> str:
        n = _read_varint(buf)
        return buf.read(n).decode("utf-8")

    def _dec(self, buf: io.BytesIO) -> Any:
        c = buf.read(1)
        if not c:
            raise SerializeError("truncated")
        t = c[0]
        if t == _T_NONE:
            return None
        if t in (_T_TRUE, _T_FALSE):
            return t == _T_TRUE
        if t == _T_INT:
            z = _read_varint(buf)
            return (z >> 1) ^ -(z & 1)
        if t == _T_FLOAT:
            n = buf.read(1)[0]
            return struct.unpack("<f" if n == 4 else "<d", buf.read(n))[0]
        if t == _T_STR:
            return self._dec_str(buf)
        if t == _T_BYTES:
            return buf.read(_read_varint(buf))
        if t == _T_ND:
            dtype = np.dtype(self._dec_str(buf))
            shape = tuple(_read_varint(buf) for _ in range(_read_varint(buf)))
            raw = buf.read(_read_varint(buf))
            return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
        if t in (_T_LIST, _T_TUPLE):
            items = [self._dec(buf) for _ in range(_read_varint(buf))]
            return tuple(items) if t == _T_TUPLE else items
        if t == _T_DICT:
            out = {}
            for _ in range(_read_varint(buf)):
                k = self._dec(buf)
                out[k] = self._dec(buf)
            return out
        if t == _T_ENUM:
            cls = self._lookup(self._dec_str(buf))
            return cls(self._dec(buf))
        if t == _T_OBJ:
            cls = self._lookup(self._dec_str(buf))
            kwargs = {}
            for _ in range(_read_varint(buf)):
                name = self._dec_str(buf)
                kwargs[name] = self._dec(buf)
            known = {f.name for f in dataclasses.fields(cls) if f.init}
            return cls(**{k: v for k, v in kwargs.items() if k in known})
        raise SerializeError(f"unknown type tag {t}")

    def _lookup(self, name: str) -> type:
        try:
            return self._types[name]
        except KeyError:
            raise SerializeError(f"unregistered type {name!r}") from None

    # JSON
    def to_jsonable(self, obj: Any) -> Any:
        if obj is None or isinstance(obj, (bool, str)):
            return obj
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, enum.Enum):
            return {"__enum__": self._type_name(obj), "value": obj.value}
        if isinstance(obj, (int, np.integer)):
            return int(obj)
        if isinstance(obj, (float, np.floating)):
            f = float(obj)
            return f if math.isfinite(f) else {"__float__": repr(f)}
        if isinstance(obj, (bytes, bytearray)):
            return {"__bytes__": bytes(obj).hex()}
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": obj.tolist(), "dtype": obj.dtype.str, "shape": list(obj.shape)}
        if isinstance(obj, tuple):
            return {"__tuple__": [self.to_jsonable(x) for x in obj]}
        if isinstance(obj, list):
            return [self.to_jsonable(x) for x in obj]
        if isinstance(obj, dict):
            if all(isinstance(k, str) and not k.startswith("__") for k in obj):
                return {k: self.to_jsonable(v) for k, v in obj.items()}
            return {"__dict__": [[self.to_jsonable(k), self.to_jsonable(v)] for k, v in obj.items()]}
        if dataclasses.is_dataclass(obj):
            return {"__type__": self._type_name(obj),
                    **{f.name: self.to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj) if f.init}}
        raise SerializeError(f"cannot serialize {type(obj).__name__}")

    def from_jsonable(self, j: Any) -> Any:
        if isinstance(j, list):
            return [self.from_jsonable(x) for x in j]
        if not isinstance(j, dict):
            return j
        if "__enum__" in j:
            return self._lookup(j["__enum__"])(j["value"])
        if "__float__" in j:
            return float(j["__float__"])
        if "__bytes__" in j:
            return bytes.fromhex(j["__bytes__"])
        if "__ndarray__" in j:
            return np.array(j["__ndarray__"], dtype=np.dtype(j["dtype"])).reshape(j["shape"])
        if "__tuple__" in j:
            return tuple(self.from_jsonable(x) for x in j["__tuple__"])
        if "__dict__" in j:
            return {self._hashable(self.from_jsonable(k)): self.from_jsonable(v) for k, v in j["__dict__"]}
        if "__type__" in j:
            cls = self._lookup(j["__type__"])
            known = {f.name for f in dataclasses.fields(cls) if f.init}
            return cls(**{k: self.from_jsonable(v) for k, v in j.items() if k in known})
        return {k: self.from_jsonable(v) for k, v in j.items()}

    @staticmethod
    def _hashable(x: Any) -> Any:
        return tuple(x) if isinstance(x, list) else x

    def to_json(self, obj: Any, **kw) -> str:
        return json.dumps(self.to_jsonable(obj), sort_keys=True, separators=(",", ":"), **kw)

    def from_json(self, text: str) -> Any:
        return self.from_jsonable(json.loads(text))
