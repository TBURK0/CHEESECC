"""Sanity checks on ML/perception outputs.

Three layers of checks, cheapest first:

1. `OutputSpec` / `check_array`: schema checks on raw tensors - shape (with
   `None` wildcards), dtype kind, finiteness, value range. Fails -> MDL_SCHEMA
   (structural) or MDL_RANGE (values).
2. `DetectionSanityChecker`: per-`core.Detection` physical plausibility -
   score in [0, 1], finite boxes, 2D box positive and inside the image (with
   margin), 3D size within class priors, position within sensor range, detection
   count bounded, and frame-to-frame count jumps.
3. `TrackJumpChecker`: per-track continuity for `core.TrackedObject` - the
   implied speed between consecutive poses must not exceed a class limit;
   class flips are flagged.

Plus an out-of-distribution interface: `OODScorer` (abstract) with a classical
baseline `ImageStatsOOD` that z-scores simple image statistics (mean, contrast,
dark/saturated fraction) against a reference set. A learned OOD detector
(e.g. energy score on model logits, feature-space Mahalanobis) plugs in behind
the same interface.

# TODO(safety): calibrate class priors and speed limits from fleet data; add
# temporal consistency of lane outputs; tie OOD thresholds to validation-set ROC.
"""

from __future__ import annotations

import abc
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from cheesestack.core import Detection, Fault, ObjectClass, TrackedObject

from .faults import DEFAULT_REGISTRY, FaultRegistry


# --------------------------------------------------------------------------- schema

@dataclass(frozen=True)
class OutputSpec:
    """Expected tensor schema. `shape` entries of None match any size."""

    name: str
    shape: tuple[Optional[int], ...]
    dtype_kind: str = "f"  # numpy dtype.kind: f float, i int, u uint, b bool
    lo: float = -math.inf
    hi: float = math.inf
    allow_nonfinite: bool = False


def check_array(arr, spec: OutputSpec, source: str = "safety.model", stamp: float = 0.0,
                registry: Optional[FaultRegistry] = None) -> list[Fault]:
    reg = registry or DEFAULT_REGISTRY
    src = f"{source}/{spec.name}"
    if not isinstance(arr, np.ndarray):
        return [reg.make("MDL_SCHEMA", src, stamp, f"{spec.name}: expected ndarray, got {type(arr).__name__}")]
    if arr.ndim != len(spec.shape) or any(e is not None and e != a for e, a in zip(spec.shape, arr.shape)):
        return [reg.make("MDL_SCHEMA", src, stamp, f"{spec.name}: shape {arr.shape} != {spec.shape}",
                         shape=list(arr.shape))]
    if arr.dtype.kind != spec.dtype_kind:
        return [reg.make("MDL_SCHEMA", src, stamp, f"{spec.name}: dtype {arr.dtype} kind != {spec.dtype_kind}")]
    faults = []
    if arr.dtype.kind == "f" and not spec.allow_nonfinite and not np.all(np.isfinite(arr)):
        faults.append(reg.make("MDL_RANGE", src, stamp, f"{spec.name}: non-finite values",
                               n_bad=int(np.count_nonzero(~np.isfinite(arr)))))
    finite = arr[np.isfinite(arr)] if arr.dtype.kind == "f" else arr
    if finite.size and (finite.min() < spec.lo or finite.max() > spec.hi):
        faults.append(reg.make("MDL_RANGE", src, stamp,
                               f"{spec.name}: values [{finite.min():.3g}, {finite.max():.3g}] outside [{spec.lo}, {spec.hi}]"))
    return faults


# --------------------------------------------------------------------------- detections

# Max plausible (length, width, height) per class, metres. Generous on purpose.
DEFAULT_CLASS_MAX_DIMS: dict[ObjectClass, tuple[float, float, float]] = {
    ObjectClass.PEDESTRIAN: (1.5, 1.5, 2.5),
    ObjectClass.BICYCLE: (2.5, 1.2, 2.5),
    ObjectClass.MOTORCYCLE: (3.0, 1.5, 2.5),
    ObjectClass.CAR: (6.5, 2.6, 2.5),
    ObjectClass.TRUCK: (25.0, 3.0, 4.5),
    ObjectClass.BUS: (25.0, 3.0, 4.5),
    ObjectClass.ANIMAL: (3.0, 1.5, 2.5),
    ObjectClass.TRAFFIC_SIGN: (3.0, 3.0, 8.0),
    ObjectClass.TRAFFIC_LIGHT: (2.0, 2.0, 8.0),
    ObjectClass.STATIC_OBSTACLE: (30.0, 30.0, 10.0),
    ObjectClass.UNKNOWN: (30.0, 30.0, 10.0),
}

# Max plausible speed per class (m/s), for track jump checks.
DEFAULT_CLASS_MAX_SPEED: dict[ObjectClass, float] = {
    ObjectClass.PEDESTRIAN: 12.0,
    ObjectClass.ANIMAL: 20.0,
    ObjectClass.BICYCLE: 25.0,
    ObjectClass.MOTORCYCLE: 90.0,
    ObjectClass.CAR: 90.0,
    ObjectClass.TRUCK: 50.0,
    ObjectClass.BUS: 50.0,
    ObjectClass.TRAFFIC_SIGN: 2.0,  # static; allows localization noise
    ObjectClass.TRAFFIC_LIGHT: 2.0,
    ObjectClass.STATIC_OBSTACLE: 2.0,
    ObjectClass.UNKNOWN: 90.0,
}


@dataclass
class DetectionSanityConfig:
    image_width: int = 1920
    image_height: int = 1080
    box_margin_px: float = 8.0
    min_box_px: float = 2.0
    max_detections: int = 256
    max_range_m: float = 250.0
    min_dim_m: float = 0.05
    class_max_dims: dict = field(default_factory=lambda: dict(DEFAULT_CLASS_MAX_DIMS))
    # Count jump: |n - mean(recent)| > max(abs, frac * mean) -> MDL_COUNT_JUMP
    count_window: int = 10
    count_jump_abs: int = 10
    count_jump_frac: float = 0.8


@dataclass
class DetectionSanityResult:
    valid: list[Detection]
    rejected: list[Detection]
    faults: list[Fault]


class DetectionSanityChecker:
    """Filters implausible detections and reports why.

    Rejected detections are not silently dropped: each produces a fault so the
    health score reflects perception quality.
    """

    def __init__(self, config: Optional[DetectionSanityConfig] = None, registry: Optional[FaultRegistry] = None,
                 source: str = "safety.model.detections") -> None:
        self.config = config or DetectionSanityConfig()
        self.registry = registry or DEFAULT_REGISTRY
        self.source = source
        self._counts: deque[int] = deque(maxlen=self.config.count_window)

    def _issue(self, d: Detection) -> Optional[tuple[str, str]]:
        c = self.config
        if not (isinstance(d.score, (int, float)) and math.isfinite(d.score) and 0.0 <= d.score <= 1.0):
            return "MDL_RANGE", f"score {d.score} not in [0,1]"
        if d.box2d is not None:
            b = d.box2d
            vals = (b.u, b.v, b.w, b.h)
            if not all(math.isfinite(x) for x in vals):
                return "MDL_RANGE", "non-finite 2D box"
            if b.w < c.min_box_px or b.h < c.min_box_px:
                return "MDL_IMPLAUSIBLE", f"2D box too small ({b.w}x{b.h})"
            m = c.box_margin_px
            if b.u < -m or b.v < -m or b.u + b.w > c.image_width + m or b.v + b.h > c.image_height + m:
                return "MDL_IMPLAUSIBLE", "2D box outside image"
        if d.box3d is not None:
            b3 = d.box3d
            vals = (b3.pose.x, b3.pose.y, b3.pose.yaw, b3.length, b3.width, b3.height)
            if not all(math.isfinite(x) for x in vals):
                return "MDL_RANGE", "non-finite 3D box"
            dims = (b3.length, b3.width, b3.height)
            if min(dims) < c.min_dim_m:
                return "MDL_IMPLAUSIBLE", f"3D box degenerate {dims}"
            mx = c.class_max_dims.get(d.cls, DEFAULT_CLASS_MAX_DIMS[ObjectClass.UNKNOWN])
            # length/width may be swapped by the model; compare sorted footprint.
            fp, mfp = sorted(dims[:2]), sorted(mx[:2])
            if fp[0] > mfp[0] or fp[1] > mfp[1] or dims[2] > mx[2]:
                return "MDL_IMPLAUSIBLE", f"{d.cls.value} size {dims} exceeds {mx}"
            if math.hypot(b3.pose.x, b3.pose.y) > c.max_range_m:
                return "MDL_IMPLAUSIBLE", "3D box beyond sensor range"
        return None

    def check(self, detections: Sequence[Detection], stamp: float) -> DetectionSanityResult:
        valid, rejected, faults = [], [], []
        if len(detections) > self.config.max_detections:
            faults.append(self.registry.make("MDL_RANGE", self.source, stamp,
                                             f"{len(detections)} detections > max {self.config.max_detections}"))
        for i, d in enumerate(detections):
            issue = self._issue(d)
            if issue is None:
                valid.append(d)
            else:
                rejected.append(d)
                faults.append(self.registry.make(issue[0], self.source, stamp, issue[1], index=i))
        n = len(detections)
        if len(self._counts) >= max(3, self.config.count_window // 2):
            mean = float(np.mean(self._counts))
            thr = max(self.config.count_jump_abs, self.config.count_jump_frac * mean)
            if abs(n - mean) > thr:
                faults.append(self.registry.make("MDL_COUNT_JUMP", self.source, stamp,
                                                 f"detection count {n} vs recent mean {mean:.1f}"))
        self._counts.append(n)
        return DetectionSanityResult(valid, rejected, faults)


# --------------------------------------------------------------------------- tracks

@dataclass
class TrackJumpConfig:
    class_max_speed: dict = field(default_factory=lambda: dict(DEFAULT_CLASS_MAX_SPEED))
    speed_margin: float = 1.5  # multiply class max speed
    position_noise_m: float = 0.5  # allowed jump independent of dt
    flag_class_flip: bool = True
    forget_after_s: float = 2.0


class TrackJumpChecker:
    """Detects tracks that teleport (implied speed > class limit) or flip class."""

    def __init__(self, config: Optional[TrackJumpConfig] = None, registry: Optional[FaultRegistry] = None,
                 source: str = "safety.model.tracks") -> None:
        self.config = config or TrackJumpConfig()
        self.registry = registry or DEFAULT_REGISTRY
        self.source = source
        self._last: dict[int, TrackedObject] = {}

    def check(self, tracks: Sequence[TrackedObject], stamp: float) -> list[Fault]:
        c = self.config
        faults = []
        for tr in tracks:
            prev = self._last.get(tr.track_id)
            if prev is not None:
                dt = tr.header.stamp - prev.header.stamp
                dist = math.hypot(tr.pose.x - prev.pose.x, tr.pose.y - prev.pose.y)
                vmax = c.class_max_speed.get(tr.cls, DEFAULT_CLASS_MAX_SPEED[ObjectClass.UNKNOWN]) * c.speed_margin
                if dt <= 0:
                    if dist > c.position_noise_m:
                        faults.append(self.registry.make("MDL_JUMP", self.source, stamp,
                                                         f"track {tr.track_id} moved {dist:.2f}m with dt={dt}",
                                                         track_id=tr.track_id))
                elif dist > vmax * dt + c.position_noise_m:
                    faults.append(self.registry.make("MDL_JUMP", self.source, stamp,
                                                     f"track {tr.track_id} implied {dist / dt:.1f} m/s > {vmax:.1f}",
                                                     track_id=tr.track_id, implied_speed=dist / dt))
                if c.flag_class_flip and tr.cls != prev.cls:
                    faults.append(self.registry.make("MDL_IMPLAUSIBLE", self.source, stamp,
                                                     f"track {tr.track_id} class {prev.cls.value}->{tr.cls.value}",
                                                     track_id=tr.track_id))
            self._last[tr.track_id] = tr
        for tid, tr in list(self._last.items()):
            if stamp - tr.header.stamp > c.forget_after_s:
                del self._last[tid]
        return faults


# --------------------------------------------------------------------------- OOD

class OODScorer(abc.ABC):
    """Out-of-distribution scorer: returns a score >= 0, larger = more unusual."""

    threshold: float = 1.0

    @abc.abstractmethod
    def score(self, x: np.ndarray) -> float: ...

    def is_ood(self, x: np.ndarray) -> bool:
        s = self.score(x)
        return (not math.isfinite(s)) or s > self.threshold


def image_features(img: np.ndarray) -> np.ndarray:
    """mean, std, dark fraction, saturated fraction of a grayscale-ized [0,1] image."""
    a = np.asarray(img)
    a = a.astype(np.float64) / 255.0 if a.dtype == np.uint8 else a.astype(np.float64)
    if a.ndim == 3:
        a = a.mean(axis=2)
    return np.array([a.mean(), a.std(), np.mean(a < 0.03), np.mean(a > 0.97)])


class ImageStatsOOD(OODScorer):
    """Baseline: max |z| of simple image statistics against a reference set.

    Catches blinded / covered / black / overexposed cameras; it will NOT catch
    semantic distribution shift. # TODO(safety): learned OOD (energy / Mahalanobis).
    """

    def __init__(self, reference_images: Sequence[np.ndarray], threshold: float = 4.0, min_std: float = 0.02) -> None:
        feats = np.stack([image_features(im) for im in reference_images])
        self.mu = feats.mean(axis=0)
        self.sigma = np.maximum(feats.std(axis=0), min_std)
        self.threshold = threshold

    def score(self, x: np.ndarray) -> float:
        f = image_features(x)
        if not np.all(np.isfinite(f)):
            return float("inf")
        return float(np.max(np.abs((f - self.mu) / self.sigma)))
