"""Object classification: classifier interface, feature baseline, label mapping.

:class:`FeatureClassifier` is a Gaussian naive-Bayes classifier over
hand-made features of an image crop (aspect ratio, log height in pixels or
metric height, fill ratio, mean saturation/value). It ships with weak
hand-set priors (pedestrians are tall and thin, cars wide, ...) and can be
re-fit from labelled feature vectors with :meth:`FeatureClassifier.fit`.

`map_label` maps detector/dataset label strings (COCO, KITTI, nuScenes
style) onto :class:`core.ObjectClass`.

Limits: the default priors only separate gross shape categories; real
use needs a learned crop classifier behind :class:`Classifier`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, replace
from typing import Optional, Sequence

import numpy as np

from cheesestack.core import BoundingBox2D, Detection, ObjectClass

from .detection import rgb_to_hsv, to_float_image

LABEL_MAP: dict[str, ObjectClass] = {
    # COCO
    "person": ObjectClass.PEDESTRIAN, "bicycle": ObjectClass.BICYCLE, "car": ObjectClass.CAR,
    "motorcycle": ObjectClass.MOTORCYCLE, "motorbike": ObjectClass.MOTORCYCLE, "bus": ObjectClass.BUS,
    "truck": ObjectClass.TRUCK, "traffic light": ObjectClass.TRAFFIC_LIGHT,
    "stop sign": ObjectClass.TRAFFIC_SIGN, "dog": ObjectClass.ANIMAL, "cat": ObjectClass.ANIMAL,
    "horse": ObjectClass.ANIMAL, "sheep": ObjectClass.ANIMAL, "cow": ObjectClass.ANIMAL,
    "bird": ObjectClass.ANIMAL, "bench": ObjectClass.STATIC_OBSTACLE, "fire hydrant": ObjectClass.STATIC_OBSTACLE,
    # KITTI
    "pedestrian": ObjectClass.PEDESTRIAN, "cyclist": ObjectClass.BICYCLE, "van": ObjectClass.CAR,
    "tram": ObjectClass.BUS, "person_sitting": ObjectClass.PEDESTRIAN,
    # nuScenes-ish
    "vehicle.car": ObjectClass.CAR, "vehicle.truck": ObjectClass.TRUCK, "vehicle.bus": ObjectClass.BUS,
    "human.pedestrian": ObjectClass.PEDESTRIAN, "movable_object.barrier": ObjectClass.STATIC_OBSTACLE,
    "movable_object.trafficcone": ObjectClass.STATIC_OBSTACLE, "traffic_cone": ObjectClass.STATIC_OBSTACLE,
    "barrier": ObjectClass.STATIC_OBSTACLE, "traffic_sign": ObjectClass.TRAFFIC_SIGN,
    "traffic_light": ObjectClass.TRAFFIC_LIGHT,
}


def map_label(label: str) -> ObjectClass:
    """Map a free-form label to ObjectClass (case/space-insensitive; prefix match for dotted names)."""
    key = label.strip().lower()
    if key in LABEL_MAP:
        return LABEL_MAP[key]
    key2 = key.replace("_", " ")
    if key2 in LABEL_MAP:
        return LABEL_MAP[key2]
    for k, v in LABEL_MAP.items():
        if "." in k and key.startswith(k):
            return v
    try:
        return ObjectClass(key)
    except ValueError:
        return ObjectClass.UNKNOWN


FEATURE_NAMES = ("aspect_hw", "log_height", "fill", "sat", "val")


def crop_features(image: np.ndarray, box: BoundingBox2D, mask: Optional[np.ndarray] = None,
                  metric_height_m: Optional[float] = None) -> np.ndarray:
    """Feature vector for a crop. `log_height` uses metric height if given, else pixels/100."""
    img = to_float_image(image)
    h, w = img.shape[:2]
    u0, v0 = int(max(0, np.floor(box.u))), int(max(0, np.floor(box.v)))
    u1, v1 = int(min(w, np.ceil(box.u + box.w))), int(min(h, np.ceil(box.v + box.h)))
    crop = img[v0:v1, u0:u1]
    if crop.size == 0:
        return np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    if mask is not None:
        m = np.asarray(mask, bool)[v0:v1, u0:u1]
    else:
        m = np.ones(crop.shape[:2], bool)
    fill = float(m.mean())
    if crop.ndim == 3 and m.any():
        hsv = rgb_to_hsv(crop)[m]
        sat, val = float(hsv[:, 1].mean()), float(hsv[:, 2].mean())
    else:
        sat, val = 0.0, float(crop[m].mean()) if m.any() else 0.0
    height = metric_height_m if metric_height_m is not None else box.h / 100.0
    return np.array([box.h / max(box.w, 1e-6), np.log(max(height, 1e-3)), fill, sat, val])


@dataclass
class ClassPrior:
    mean: np.ndarray
    std: np.ndarray
    prior: float = 1.0


def _default_priors() -> dict[ObjectClass, ClassPrior]:
    # Features: aspect h/w, log(metric height m), fill, sat, val. Shape-dominated; color weak.
    big = 10.0
    return {
        ObjectClass.PEDESTRIAN: ClassPrior(np.array([2.6, np.log(1.7), 0.55, 0.3, 0.5]), np.array([0.7, 0.2, 0.2, big, big])),
        ObjectClass.CAR: ClassPrior(np.array([0.55, np.log(1.5), 0.8, 0.3, 0.5]), np.array([0.2, 0.2, 0.2, big, big])),
        ObjectClass.TRUCK: ClassPrior(np.array([0.75, np.log(3.2), 0.85, 0.3, 0.5]), np.array([0.25, 0.2, 0.2, big, big])),
        ObjectClass.BUS: ClassPrior(np.array([0.6, np.log(3.0), 0.9, 0.3, 0.5]), np.array([0.25, 0.2, 0.15, big, big]), 0.5),
        ObjectClass.BICYCLE: ClassPrior(np.array([1.4, np.log(1.6), 0.4, 0.3, 0.5]), np.array([0.4, 0.2, 0.15, big, big]), 0.5),
        ObjectClass.STATIC_OBSTACLE: ClassPrior(np.array([1.0, np.log(0.8), 0.7, 0.3, 0.5]), np.array([0.6, 0.5, 0.3, big, big]), 0.5),
    }


class Classifier(abc.ABC):
    """Crop -> (class, score, per-class probabilities)."""

    @abc.abstractmethod
    def classify(self, image: np.ndarray, box: BoundingBox2D, **kw) -> tuple[ObjectClass, float, dict]:
        ...

    def __call__(self, image: np.ndarray, box: BoundingBox2D) -> tuple[ObjectClass, float]:
        """Adapter for detection.BlobDetector(classifier=...)."""
        cls, score, _ = self.classify(image, box)
        return cls, score

    def classify_detections(self, image: np.ndarray, dets: Sequence[Detection]) -> list[Detection]:
        out = []
        for d in dets:
            if d.box2d is None:
                out.append(d)
                continue
            cls, score, probs = self.classify(image, d.box2d)
            attrs = dict(d.attributes, class_probs={k.value: v for k, v in probs.items()})
            out.append(replace(d, cls=cls, score=float(d.score * score), attributes=attrs))
        return out


@dataclass
class FeatureClassifierConfig:
    min_confidence: float = 0.35  # below this the class is UNKNOWN
    var_floor: float = 1e-3


class FeatureClassifier(Classifier):
    """Gaussian naive Bayes over :data:`FEATURE_NAMES`."""

    def __init__(self, config: Optional[FeatureClassifierConfig] = None,
                 priors: Optional[dict[ObjectClass, ClassPrior]] = None) -> None:
        self.config = config or FeatureClassifierConfig()
        self.priors = priors if priors is not None else _default_priors()

    def fit(self, X: np.ndarray, y: Sequence[ObjectClass]) -> "FeatureClassifier":
        X = np.asarray(X, float)
        y = list(y)
        self.priors = {}
        for c in set(y):
            sel = np.array([yy == c for yy in y])
            xs = X[sel]
            std = np.sqrt(xs.var(axis=0) + self.config.var_floor)
            self.priors[c] = ClassPrior(xs.mean(axis=0), std, float(sel.mean()))
        return self

    def predict_proba(self, f: np.ndarray) -> dict[ObjectClass, float]:
        classes = list(self.priors)
        ll = np.array([
            -0.5 * np.sum(((f - p.mean) / p.std) ** 2) - np.sum(np.log(p.std)) + np.log(max(p.prior, 1e-9))
            for p in (self.priors[c] for c in classes)
        ])
        ll -= ll.max()
        pr = np.exp(ll)
        pr /= pr.sum()
        return dict(zip(classes, pr.tolist()))

    def classify(self, image: np.ndarray, box: BoundingBox2D, mask: Optional[np.ndarray] = None,
                 metric_height_m: Optional[float] = None) -> tuple[ObjectClass, float, dict]:
        f = crop_features(image, box, mask, metric_height_m)
        probs = self.predict_proba(f)
        best = max(probs, key=probs.get)
        if probs[best] < self.config.min_confidence:
            return ObjectClass.UNKNOWN, probs[best], probs
        return best, probs[best], probs
        # TODO(perception): learned crop classifier (CNN) behind Classifier; calibrate probabilities.
