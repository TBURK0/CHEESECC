"""Dataset versioning: content-hash snapshots, diffs, deterministic splits.

Each sample's *content hash* is SHA-256 over its canonical JSON record (labels +
metadata, image path excluded) plus the raw image bytes. A ``DatasetSnapshot``
maps ``sample_id -> content_hash``; its id is the hash of that sorted mapping, so
identical content always yields the same snapshot id (git-tree style).

``DatasetRegistry`` stores snapshots as JSON under ``<root>/versions/`` with a
copy of the manifest, supports named tags, and diffs two versions.

``split_for`` assigns train/val/test from a hash of the sample id (or a group key
such as drive id, to keep correlated frames together), so a sample never
changes split when the dataset grows.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .dataset import SampleRecord, write_manifest, read_manifest


def content_hash(record: SampleRecord, root: Optional[os.PathLike] = None) -> str:
    d = record.to_dict()
    d.pop("image_path", None)
    d.pop("triggers", None)  # why we collected it does not change content
    h = hashlib.sha256(json.dumps(d, sort_keys=True).encode())
    img = record.image
    if img is None and record.image_path is not None:
        try:
            img = record.load_image(root)
        except (FileNotFoundError, OSError):
            img = None
    if img is not None:
        a = np.ascontiguousarray(img)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


@dataclass
class DatasetSnapshot:
    name: str
    entries: dict[str, str]  # sample_id -> content hash
    parent: Optional[str] = None
    created: float = 0.0  # caller-supplied stamp (no wall clock in library code)
    metadata: dict = field(default_factory=dict)

    @property
    def snapshot_id(self) -> str:
        blob = json.dumps(sorted(self.entries.items())).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {"id": self.snapshot_id, "name": self.name, "entries": self.entries, "parent": self.parent,
                "created": self.created, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetSnapshot":
        return cls(d["name"], dict(d["entries"]), d.get("parent"), d.get("created", 0.0), d.get("metadata", {}))


@dataclass
class DatasetDiff:
    added: list[str]
    removed: list[str]
    modified: list[str]
    unchanged: int

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.modified)


def diff_snapshots(old: DatasetSnapshot, new: DatasetSnapshot) -> DatasetDiff:
    a, b = old.entries, new.entries
    added = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    common = set(a) & set(b)
    modified = sorted(k for k in common if a[k] != b[k])
    return DatasetDiff(added, removed, modified, len(common) - len(modified))


@dataclass
class SplitConfig:
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)
    names: tuple[str, str, str] = ("train", "val", "test")
    salt: str = "cheesestack"
    group_key: Optional[str] = None  # sensors[group_key] used instead of sample_id if present


def hash_unit(key: str, salt: str = "") -> float:
    """Deterministic uniform value in [0, 1) from a string."""
    digest = hashlib.sha256((salt + ":" + key).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2.0 ** 64


def split_for(record_or_id, config: Optional[SplitConfig] = None) -> str:
    cfg = config or SplitConfig()
    if isinstance(record_or_id, SampleRecord):
        key = record_or_id.sample_id
        if cfg.group_key and cfg.group_key in record_or_id.sensors:
            key = str(record_or_id.sensors[cfg.group_key])
    else:
        key = str(record_or_id)
    total = float(sum(cfg.fractions))
    u = hash_unit(key, cfg.salt) * total
    acc = 0.0
    for name, frac in zip(cfg.names, cfg.fractions):
        acc += frac
        if u < acc:
            return name
    return cfg.names[-1]


def split_records(records: Sequence[SampleRecord], config: Optional[SplitConfig] = None) -> dict[str, list[SampleRecord]]:
    cfg = config or SplitConfig()
    out: dict[str, list[SampleRecord]] = {n: [] for n in cfg.names}
    for r in records:
        out[split_for(r, cfg)].append(r)
    return out


class DatasetRegistry:
    """Filesystem store of snapshots.

    Layout: ``root/versions/<id>.json``, ``root/versions/<id>.manifest.jsonl``, ``root/tags.json``.
    # TODO(ml): object-store backend (S3/GCS), image dedup store keyed by content hash, locking.
    """

    def __init__(self, root: os.PathLike) -> None:
        self.root = Path(root)
        (self.root / "versions").mkdir(parents=True, exist_ok=True)
        self._tags_path = self.root / "tags.json"

    def commit(self, records: Sequence[SampleRecord], name: str, parent: Optional[str] = None,
               created: float = 0.0, image_root: Optional[os.PathLike] = None,
               metadata: Optional[dict] = None) -> DatasetSnapshot:
        entries = {r.sample_id: content_hash(r, image_root) for r in records}
        if len(entries) != len(records):
            raise ValueError("duplicate sample_id in commit")
        snap = DatasetSnapshot(name, entries, parent, created, dict(metadata or {}))
        sid = snap.snapshot_id
        with open(self.root / "versions" / f"{sid}.json", "w", encoding="utf-8") as f:
            json.dump(snap.to_dict(), f, sort_keys=True)
        write_manifest(self.root / "versions" / f"{sid}.manifest.jsonl", records)
        self.tag(name, sid)
        return snap

    def load(self, ref: str) -> DatasetSnapshot:
        sid = self.resolve(ref)
        with open(self.root / "versions" / f"{sid}.json", encoding="utf-8") as f:
            return DatasetSnapshot.from_dict(json.load(f))

    def records(self, ref: str) -> list[SampleRecord]:
        return read_manifest(self.root / "versions" / f"{self.resolve(ref)}.manifest.jsonl")

    def resolve(self, ref: str) -> str:
        tags = self.tags()
        if ref in tags:
            return tags[ref]
        if (self.root / "versions" / f"{ref}.json").exists():
            return ref
        raise KeyError(f"unknown dataset version '{ref}'")

    def tags(self) -> dict[str, str]:
        if not self._tags_path.exists():
            return {}
        return json.loads(self._tags_path.read_text(encoding="utf-8"))

    def tag(self, name: str, snapshot_id: str) -> None:
        t = self.tags()
        t[name] = snapshot_id
        self._tags_path.write_text(json.dumps(t, sort_keys=True), encoding="utf-8")

    def list_versions(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "versions").glob("*.json"))

    def diff(self, old_ref: str, new_ref: str) -> DatasetDiff:
        return diff_snapshots(self.load(old_ref), self.load(new_ref))
