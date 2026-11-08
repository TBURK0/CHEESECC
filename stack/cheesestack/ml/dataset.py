"""Dataset collection: sample records, mining triggers, and JSONL manifests.

A ``SampleRecord`` is one logged frame: image (array and/or path), sensor
metadata (speed, pose, weather, camera intrinsics, ...), and optional labels.
The on-vehicle ``DatasetCollector`` offers every frame to a set of triggers;
any trigger that fires keeps the frame (subject to a rate limit and a budget),
writes the image as ``.npy`` and appends a line to ``manifest.jsonl``.

Triggers (active-learning style data mining):
- ``LowConfidenceTrigger``  a prediction score falls in an "uncertain" band
- ``DisagreementTrigger``   two models (or model vs. tracker) disagree (unmatched boxes by IoU)
- ``RareClassTrigger``      a class whose running frequency is below a threshold appears
- ``PeriodicTrigger``       uniform background sampling every N seconds
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Protocol, Sequence

import numpy as np

from .annotation import Annotation, BoxLabel


@dataclass
class SampleRecord:
    sample_id: str
    stamp: float
    image: Optional[np.ndarray] = None  # not serialised; written to image_path
    image_path: Optional[str] = None
    sensors: dict = field(default_factory=dict)  # JSON-able metadata
    annotation: Optional[Annotation] = None
    tags: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id, "stamp": float(self.stamp), "image_path": self.image_path,
            "sensors": self.sensors, "annotation": self.annotation.to_dict() if self.annotation else None,
            "tags": list(self.tags), "triggers": list(self.triggers),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SampleRecord":
        ann = Annotation.from_dict(d["annotation"]) if d.get("annotation") else None
        return cls(d["sample_id"], float(d["stamp"]), None, d.get("image_path"), dict(d.get("sensors", {})),
                   ann, list(d.get("tags", [])), list(d.get("triggers", [])))

    def load_image(self, root: Optional[os.PathLike] = None) -> np.ndarray:
        if self.image is not None:
            return self.image
        if self.image_path is None:
            raise ValueError(f"sample {self.sample_id} has no image")
        p = Path(self.image_path)
        if root is not None and not p.is_absolute():
            p = Path(root) / p
        # TODO(ml): JPEG/PNG decode once an image codec is allowed; .npy only for now.
        self.image = np.load(p)
        return self.image


# --------------------------------------------------------------------------- manifests


def write_manifest(path: os.PathLike, records: Sequence[SampleRecord], append: bool = False) -> None:
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), sort_keys=True) + "\n")


def read_manifest(path: os.PathLike) -> list[SampleRecord]:
    return list(iter_manifest(path))


def iter_manifest(path: os.PathLike) -> Iterator[SampleRecord]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield SampleRecord.from_dict(json.loads(line))


# --------------------------------------------------------------------------- triggers


@dataclass
class TriggerContext:
    """What a trigger sees besides the record: primary and optional secondary predictions."""

    predictions: list[BoxLabel] = field(default_factory=list)
    secondary: Optional[list[BoxLabel]] = None


class Trigger(Protocol):
    name: str

    def __call__(self, record: SampleRecord, ctx: TriggerContext) -> bool: ...


@dataclass
class LowConfidenceTrigger:
    low: float = 0.3
    high: float = 0.6
    name: str = "low_confidence"

    def __call__(self, record: SampleRecord, ctx: TriggerContext) -> bool:
        return any(self.low <= p.score <= self.high for p in ctx.predictions)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy arrays."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


@dataclass
class DisagreementTrigger:
    """Fires when the two prediction sets have >= ``min_unmatched`` boxes without a
    same-class partner at IoU >= ``iou_threshold``."""

    iou_threshold: float = 0.5
    min_score: float = 0.3
    min_unmatched: int = 1
    name: str = "disagreement"

    def __call__(self, record: SampleRecord, ctx: TriggerContext) -> bool:
        if ctx.secondary is None:
            return False
        a = [p for p in ctx.predictions if p.score >= self.min_score]
        b = [p for p in ctx.secondary if p.score >= self.min_score]
        unmatched = 0
        for src, dst in ((a, b), (b, a)):
            if not dst:
                unmatched += len(src)
                continue
            iou = _iou_xyxy(np.array([p.xyxy() for p in src]).reshape(-1, 4),
                            np.array([p.xyxy() for p in dst]).reshape(-1, 4))
            for i, p in enumerate(src):
                same = np.array([q.cls == p.cls for q in dst])
                if not np.any(iou[i][same] >= self.iou_threshold):
                    unmatched += 1
        return unmatched >= self.min_unmatched


@dataclass
class RareClassTrigger:
    """Keeps frames containing a class whose observed frequency is < ``max_frequency``.

    Counts are updated online from every offered frame (prediction classes + label classes).
    ``always_rare`` classes fire regardless of counts (e.g. "animal", "school_bus").
    """

    max_frequency: float = 0.02
    min_observations: int = 50
    always_rare: Sequence[str] = ()
    name: str = "rare_class"
    counts: dict = field(default_factory=dict)
    total: int = 0

    def __call__(self, record: SampleRecord, ctx: TriggerContext) -> bool:
        classes = {p.cls for p in ctx.predictions}
        if record.annotation:
            classes |= {b.cls for b in record.annotation.boxes}
        for c in classes:
            self.counts[c] = self.counts.get(c, 0) + 1
        self.total += 1
        if classes & set(self.always_rare):
            return True
        if self.total < self.min_observations:
            return False
        return any(self.counts[c] / self.total < self.max_frequency for c in classes)


@dataclass
class PeriodicTrigger:
    period_s: float = 10.0
    name: str = "periodic"
    _last: float = float("-inf")

    def __call__(self, record: SampleRecord, ctx: TriggerContext) -> bool:
        if record.stamp - self._last >= self.period_s:
            self._last = record.stamp
            return True
        return False


# --------------------------------------------------------------------------- collector


@dataclass
class CollectorConfig:
    output_dir: str = "dataset"
    manifest_name: str = "manifest.jsonl"
    min_interval_s: float = 0.5  # rate limit between kept frames
    max_samples: int = 10_000  # storage budget
    save_images: bool = True


class DatasetCollector:
    """Runs triggers on offered frames and persists the kept ones.

    # TODO(ml): ring-buffer pre/post-trigger context frames, upload queue with
    # bandwidth limits, PII blurring (faces / plates) before storage.
    """

    def __init__(self, triggers: Sequence[Trigger], config: Optional[CollectorConfig] = None) -> None:
        self.config = config or CollectorConfig()
        self.triggers = list(triggers)
        self.out = Path(self.config.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "images").mkdir(exist_ok=True)
        self.manifest_path = self.out / self.config.manifest_name
        self.kept = 0
        self.offered = 0
        self.trigger_counts: dict[str, int] = {t.name: 0 for t in self.triggers}
        self._last_kept = float("-inf")

    def offer(self, record: SampleRecord, ctx: Optional[TriggerContext] = None) -> bool:
        ctx = ctx or TriggerContext()
        self.offered += 1
        fired = [t.name for t in self.triggers if t(record, ctx)]  # evaluate all (stateful triggers)
        if not fired or self.kept >= self.config.max_samples:
            return False
        if record.stamp - self._last_kept < self.config.min_interval_s:
            return False
        for n in fired:
            self.trigger_counts[n] += 1
        record.triggers = sorted(set(record.triggers) | set(fired))
        if self.config.save_images and record.image is not None:
            rel = f"images/{record.sample_id}.npy"
            np.save(self.out / rel, record.image)
            record.image_path = rel
        write_manifest(self.manifest_path, [record], append=True)
        self._last_kept = record.stamp
        self.kept += 1
        return True

    def records(self) -> list[SampleRecord]:
        return read_manifest(self.manifest_path) if self.manifest_path.exists() else []
