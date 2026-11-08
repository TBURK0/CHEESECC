"""Append-only segmented record store for time-series messages.

Layout on disk (``root/``)::

    index.json              segment list: file, t_min, t_max, count, topics
    seg_000000.jsonl        one JSON record per line: {"t", "topic", "data"}
    arrays/<topic>/chunk_000000.npz   numpy chunks: t (N,), data (N, ...)

``RecordStore`` appends records to the open segment and rolls to a new one
when ``segment_max_records`` or ``segment_max_bytes`` is hit. The index keeps
per-segment time bounds so ``query(t0, t1)`` only opens overlapping segments.
Records within a segment need not be time-sorted; query results are sorted by
``t`` (stable). Numpy values in ``data`` are converted to lists.

``ArrayStore`` stores dense numeric streams (e.g. IMU, wheel speeds) as npz
chunks with a time index, returning concatenated arrays for a time range.

Limits: single writer per root; the index is rewritten on every segment roll
and on ``close()`` (a crash loses only the open segment's index entry, which
``RecordStore.reindex()`` rebuilds). TODO(infra): MCAP/rosbag2 export,
per-segment compression, concurrent readers with file locks, retention policy.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np


@dataclass
class RecordStoreConfig:
    segment_max_records: int = 10_000
    segment_max_bytes: int = 16_000_000
    fsync: bool = False


@dataclass
class Record:
    t: float
    topic: str
    data: Any


@dataclass
class SegmentInfo:
    file: str
    t_min: float = float("inf")
    t_max: float = float("-inf")
    count: int = 0
    bytes: int = 0
    topics: list = field(default_factory=list)

    def overlaps(self, t0: float, t1: float) -> bool:
        return self.count > 0 and self.t_max >= t0 and self.t_min <= t1


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if hasattr(x, "__dataclass_fields__"):
        import dataclasses
        return _to_jsonable(dataclasses.asdict(x))
    if hasattr(x, "value") and hasattr(x, "name") and type(x).__module__ != "builtins":
        return x.value  # enums
    return x


class RecordStore:
    INDEX = "index.json"

    def __init__(self, root: str | os.PathLike, config: Optional[RecordStoreConfig] = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config or RecordStoreConfig()
        self.segments: list[SegmentInfo] = []
        self._fh = None
        self._load_index()

    # -- index ------------------------------------------------------------------
    def _load_index(self) -> None:
        p = self.root / self.INDEX
        if p.exists():
            self.segments = [SegmentInfo(**s) for s in json.loads(p.read_text())["segments"]]
        known = {s.file for s in self.segments}
        orphans = sorted(f.name for f in self.root.glob("seg_*.jsonl") if f.name not in known)
        for name in orphans:  # recover segments never written to the index
            self.segments.append(self._scan(name))

    def _write_index(self) -> None:
        tmp = self.root / (self.INDEX + ".tmp")
        tmp.write_text(json.dumps({"segments": [vars(s) for s in self.segments]}, indent=1))
        os.replace(tmp, self.root / self.INDEX)

    def _scan(self, name: str) -> SegmentInfo:
        info = SegmentInfo(file=name)
        topics: set = set()
        with (self.root / name).open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                info.t_min, info.t_max = min(info.t_min, r["t"]), max(info.t_max, r["t"])
                info.count += 1
                info.bytes += len(line.encode())
                topics.add(r["topic"])
        info.topics = sorted(topics)
        return info

    def reindex(self) -> None:
        self.close()
        self.segments = [self._scan(f.name) for f in sorted(self.root.glob("seg_*.jsonl"))]
        self._write_index()

    # -- writing ----------------------------------------------------------------
    def _open_segment(self) -> SegmentInfo:
        cur = self.segments[-1] if self.segments else None
        full = cur is not None and (cur.count >= self.config.segment_max_records
                                    or cur.bytes >= self.config.segment_max_bytes)
        if cur is None or full:
            self.close()
            cur = SegmentInfo(file=f"seg_{len(self.segments):06d}.jsonl")
            self.segments.append(cur)
            self._write_index()
        if self._fh is None:
            self._fh = (self.root / cur.file).open("a", encoding="utf-8")
        return cur

    def append(self, t: float, topic: str, data: Any) -> None:
        seg = self._open_segment()
        line = json.dumps({"t": float(t), "topic": topic, "data": _to_jsonable(data)},
                          separators=(",", ":")) + "\n"
        self._fh.write(line)
        if self.config.fsync:
            self._fh.flush()
            os.fsync(self._fh.fileno())
        seg.count += 1
        seg.bytes += len(line.encode())
        seg.t_min, seg.t_max = min(seg.t_min, float(t)), max(seg.t_max, float(t))
        if topic not in seg.topics:
            seg.topics.append(topic)

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()
        self._write_index()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._write_index()

    def __enter__(self) -> "RecordStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- reading ----------------------------------------------------------------
    def iter_segment(self, seg: SegmentInfo) -> Iterator[Record]:
        if self._fh is not None:
            self._fh.flush()
        with (self.root / seg.file).open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    yield Record(r["t"], r["topic"], r["data"])

    def query(self, t0: float = float("-inf"), t1: float = float("inf"),
              topics: Optional[list[str]] = None) -> list[Record]:
        """Records with t0 <= t <= t1 (optionally filtered by topic), time-sorted."""
        want = set(topics) if topics else None
        out: list[Record] = []
        for seg in self.segments:
            if not seg.overlaps(t0, t1):
                continue
            if want is not None and not want.intersection(seg.topics):
                continue
            for r in self.iter_segment(seg):
                if t0 <= r.t <= t1 and (want is None or r.topic in want):
                    out.append(r)
        out.sort(key=lambda r: r.t)
        return out

    def time_range(self) -> tuple[float, float]:
        segs = [s for s in self.segments if s.count]
        if not segs:
            return (float("nan"), float("nan"))
        return (min(s.t_min for s in segs), max(s.t_max for s in segs))

    def __len__(self) -> int:
        return sum(s.count for s in self.segments)


@dataclass
class ArrayStoreConfig:
    chunk_size: int = 4096  # samples buffered before writing a chunk


class ArrayStore:
    """Dense numeric time series per topic, stored as npz chunks."""

    def __init__(self, root: str | os.PathLike, config: Optional[ArrayStoreConfig] = None) -> None:
        self.root = Path(root) / "arrays"
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config or ArrayStoreConfig()
        self._buf: dict[str, tuple[list, list]] = {}
        self._index: dict[str, list[dict]] = {}
        idx = self.root / "index.json"
        if idx.exists():
            self._index = json.loads(idx.read_text())

    def append(self, topic: str, t: float, value) -> None:
        ts, vs = self._buf.setdefault(topic, ([], []))
        ts.append(float(t))
        vs.append(np.asarray(value))
        if len(ts) >= self.config.chunk_size:
            self._flush_topic(topic)

    def _topic_dir(self, topic: str) -> Path:
        d = self.root / topic.strip("/").replace("/", "__")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _flush_topic(self, topic: str) -> None:
        ts, vs = self._buf.get(topic, ([], []))
        if not ts:
            return
        chunks = self._index.setdefault(topic, [])
        name = f"chunk_{len(chunks):06d}.npz"
        path = self._topic_dir(topic) / name
        t = np.asarray(ts, dtype=np.float64)
        np.savez(path, t=t, data=np.stack(vs))
        chunks.append({"file": str(path.relative_to(self.root)), "t_min": float(t.min()),
                       "t_max": float(t.max()), "count": len(ts)})
        self._buf[topic] = ([], [])
        (self.root / "index.json").write_text(json.dumps(self._index))

    def flush(self) -> None:
        for topic in list(self._buf):
            self._flush_topic(topic)

    def query(self, topic: str, t0: float = float("-inf"), t1: float = float("inf")) -> tuple[np.ndarray, np.ndarray]:
        """Return (t, data) sorted by time for t0 <= t <= t1. Includes unflushed samples."""
        ts, ds = [], []
        for c in self._index.get(topic, []):
            if c["t_max"] < t0 or c["t_min"] > t1:
                continue
            with np.load(self.root / c["file"]) as z:
                ts.append(z["t"])
                ds.append(z["data"])
        bt, bv = self._buf.get(topic, ([], []))
        if bt:
            ts.append(np.asarray(bt))
            ds.append(np.stack(bv))
        if not ts:
            return np.zeros(0), np.zeros((0,))
        t = np.concatenate(ts)
        d = np.concatenate(ds)
        m = (t >= t0) & (t <= t1)
        order = np.argsort(t[m], kind="stable")
        return t[m][order], d[m][order]

    def topics(self) -> list[str]:
        return sorted(set(self._index) | {k for k, v in self._buf.items() if v[0]})
