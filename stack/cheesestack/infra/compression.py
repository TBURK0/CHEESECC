"""Data compression utilities.

- ``compress``/``decompress``: zlib or lzma with a 1-byte codec header so the
  reader does not need to know which codec was used.
- ``zigzag`` + LEB128 ``varint`` encoding of signed integers.
- ``encode_timeseries``/``decode_timeseries``: quantize floats to a fixed step,
  delta (order 1 or 2) encode, zigzag + varint pack, optionally zlib. Smooth
  signals (timestamps at fixed rate, slowly varying speeds) compress to ~1 byte
  per sample. Error is bounded by ``step / 2`` per sample (no drift, since
  deltas are taken on the quantized integers).
- ``quantize``/``dequantize``: linear quantization of arrays to uint8/uint16
  over [lo, hi] (e.g. images, occupancy grids, confidences).

TODO(infra): Gorilla XOR float compression, zstd when available, SIMD varint.
"""

from __future__ import annotations

import lzma
import struct
import zlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

_CODECS = {b"\x00": "none", b"\x01": "zlib", b"\x02": "lzma"}
_CODEC_IDS = {v: k for k, v in _CODECS.items()}


@dataclass
class CompressionConfig:
    codec: str = "zlib"  # none | zlib | lzma
    level: int = 6


def compress(data: bytes, config: Optional[CompressionConfig] = None) -> bytes:
    cfg = config or CompressionConfig()
    if cfg.codec == "zlib":
        body = zlib.compress(data, cfg.level)
    elif cfg.codec == "lzma":
        body = lzma.compress(data, preset=min(9, max(0, cfg.level)))
    elif cfg.codec == "none":
        body = bytes(data)
    else:
        raise ValueError(f"unknown codec {cfg.codec}")
    return _CODEC_IDS[cfg.codec] + body


def decompress(blob: bytes) -> bytes:
    codec = _CODECS.get(blob[:1])
    body = blob[1:]
    if codec == "zlib":
        return zlib.decompress(body)
    if codec == "lzma":
        return lzma.decompress(body)
    if codec == "none":
        return bytes(body)
    raise ValueError("unknown codec header")


# -- integer coding ---------------------------------------------------------------
def zigzag_encode(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    return ((x << 1) ^ (x >> 63)).astype(np.uint64)


def zigzag_decode(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.uint64)
    return ((z >> np.uint64(1)).astype(np.int64)) ^ (-(z & np.uint64(1)).astype(np.int64))


def varint_encode(values) -> bytes:
    """LEB128 encode non-negative ints (< 2**64)."""
    out = bytearray()
    for v in np.asarray(values, dtype=np.uint64).tolist():
        v = int(v)
        while True:
            b = v & 0x7F
            v >>= 7
            if v:
                out.append(b | 0x80)
            else:
                out.append(b)
                break
    return bytes(out)


def varint_decode(buf: bytes, count: Optional[int] = None) -> tuple[np.ndarray, int]:
    """Decode up to ``count`` varints; returns (values uint64, bytes consumed)."""
    vals: list[int] = []
    shift = cur = 0
    i = 0
    n = len(buf)
    while i < n and (count is None or len(vals) < count):
        b = buf[i]
        i += 1
        cur |= (b & 0x7F) << shift
        if b & 0x80:
            shift += 7
            if shift > 63:
                raise ValueError("varint too long")
        else:
            vals.append(cur)
            cur = shift = 0
    if shift:
        raise ValueError("truncated varint")
    return np.asarray(vals, dtype=np.uint64), i


# -- timeseries -------------------------------------------------------------------
@dataclass
class TimeseriesCodecConfig:
    step: float = 1e-3  # quantization step (units of the signal)
    order: int = 1  # delta order: 0, 1 or 2
    zlib_level: int = 0  # 0 disables the final zlib pass


_TS_HDR = struct.Struct("<4sBBdI")  # magic, order, zlib flag, step, count
_MAGIC = b"CSTS"


def encode_timeseries(x, config: Optional[TimeseriesCodecConfig] = None) -> bytes:
    cfg = config or TimeseriesCodecConfig()
    if cfg.order not in (0, 1, 2):
        raise ValueError("order must be 0, 1 or 2")
    x = np.asarray(x, dtype=np.float64).ravel()
    if not np.all(np.isfinite(x)):
        raise ValueError("timeseries must be finite")
    q = np.round(x / cfg.step).astype(np.int64)
    d = q.copy()
    for _ in range(cfg.order):
        d[1:] = np.diff(d)  # first element stays as anchor
    # For order 2 the second element is a first difference; decode mirrors this.
    body = varint_encode(zigzag_encode(d))
    if cfg.zlib_level:
        body = zlib.compress(body, cfg.zlib_level)
    return _TS_HDR.pack(_MAGIC, cfg.order, 1 if cfg.zlib_level else 0, cfg.step, len(x)) + body


def decode_timeseries(blob: bytes) -> np.ndarray:
    magic, order, zflag, step, count = _TS_HDR.unpack_from(blob)
    if magic != _MAGIC:
        raise ValueError("not a timeseries blob")
    body = blob[_TS_HDR.size:]
    if zflag:
        body = zlib.decompress(body)
    z, _ = varint_decode(body, count)
    d = zigzag_decode(z)
    for _ in range(order):
        d = np.cumsum(d)
    return d.astype(np.float64) * step


# -- quantization -----------------------------------------------------------------
def quantize(x, lo: float, hi: float, bits: int = 8) -> np.ndarray:
    """Map [lo, hi] linearly to [0, 2**bits - 1] (clipped). bits in {8, 16}."""
    if bits not in (8, 16):
        raise ValueError("bits must be 8 or 16")
    if hi <= lo:
        raise ValueError("hi must exceed lo")
    levels = (1 << bits) - 1
    x = np.clip((np.asarray(x, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)
    return np.round(x * levels).astype(np.uint8 if bits == 8 else np.uint16)


def dequantize(q, lo: float, hi: float, bits: int = 8) -> np.ndarray:
    levels = (1 << bits) - 1
    return lo + np.asarray(q, dtype=np.float64) / levels * (hi - lo)
