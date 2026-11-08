"""Traffic-sign recognition: detect+classify interface, color/shape baseline, temporal voting.

:class:`ColorShapeSignRecognizer`:

1. HSV masks for sign colors (red, blue, yellow).
2. Connected components per color -> candidate blobs (size/aspect gated).
3. Shape from the hole-filled blob's fill ratio within its bounding box
   (triangle ~0.5, circle ~0.785, octagon ~0.83, rectangle ~1.0; a
   diamond, i.e. a rotated square, is also ~0.5 and is told apart from a
   triangle by vertical symmetry) and triangle orientation from where the
   mass sits.
4. (color, shape, ring-ness) -> :class:`SignType` by a rule table.

:class:`SignVoter` associates per-frame sign detections across frames by
box IoU and emits a sign only once its type has a stable weighted majority.

Limits: no text/number reading (speed value), pictogram content ignored,
sensitive to lighting; rule table is region-specific (Vienna convention).
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from cheesestack.core import BoundingBox2D, Detection, Header, ObjectClass

from .detection import (boxes_to_array, component_boxes, connected_components, fill_holes, hue_mask,
                        iou_matrix, nms_detections, rgb_to_hsv)


class SignType(enum.Enum):
    UNKNOWN = "unknown"
    STOP = "stop"
    YIELD = "yield"
    NO_ENTRY = "no_entry"
    SPEED_LIMIT = "speed_limit"  # red ring; value not read
    PROHIBITORY = "prohibitory"
    WARNING = "warning"
    MANDATORY = "mandatory"
    INFORMATION = "information"


class SignShape(enum.Enum):
    UNKNOWN = "unknown"
    TRIANGLE_UP = "triangle_up"
    TRIANGLE_DOWN = "triangle_down"
    DIAMOND = "diamond"
    CIRCLE = "circle"
    OCTAGON = "octagon"
    RECTANGLE = "rectangle"


class TrafficSignRecognizer(abc.ABC):
    @abc.abstractmethod
    def recognize(self, image: np.ndarray, header: Header) -> list[Detection]:
        """Detections with cls=TRAFFIC_SIGN and attributes['sign_type'] (SignType)."""


@dataclass
class SignRecognizerConfig:
    s_min: float = 0.45
    v_min: float = 0.3
    red_hue: tuple[float, float] = (0.94, 0.05)  # wraps
    blue_hue: tuple[float, float] = (0.55, 0.70)
    yellow_hue: tuple[float, float] = (0.10, 0.18)
    min_area_px: int = 60
    max_aspect_dev: float = 0.5  # |log(h/w)| limit
    ring_interior_min: float = 0.25  # interior (non-colored) share of filled blob for a ring sign
    nms_iou: float = 0.3


def classify_shape(filled: np.ndarray) -> tuple[SignShape, float]:
    """Shape from a hole-filled blob crop (bool, tight bbox). Returns (shape, fill ratio)."""
    h, w = filled.shape
    fill = float(filled.mean())
    if fill > 0.93:
        return SignShape.RECTANGLE, fill
    if fill > 0.81:
        return SignShape.OCTAGON, fill
    if fill > 0.68:
        return SignShape.CIRCLE, fill
    if 0.35 < fill <= 0.68:
        rows = filled.sum(axis=1).astype(float)
        top, bot = rows[: h // 3].mean(), rows[-(h // 3):].mean()
        mid = rows[h // 3: h - h // 3].mean() if h >= 3 else rows.mean()
        if mid > 1.2 * max(top, bot):
            return SignShape.DIAMOND, fill
        return (SignShape.TRIANGLE_UP if bot > top else SignShape.TRIANGLE_DOWN), fill
    return SignShape.UNKNOWN, fill


def sign_type_from(color: str, shape: SignShape, ring: bool, center_bar: bool) -> SignType:
    if color == "red":
        if shape == SignShape.OCTAGON:
            return SignType.STOP
        if shape == SignShape.TRIANGLE_DOWN:
            return SignType.YIELD
        if shape == SignShape.TRIANGLE_UP:
            return SignType.WARNING
        if shape == SignShape.CIRCLE:
            if center_bar:
                return SignType.NO_ENTRY
            return SignType.SPEED_LIMIT if ring else SignType.PROHIBITORY
    if color == "blue":
        if shape == SignShape.CIRCLE:
            return SignType.MANDATORY
        if shape in (SignShape.RECTANGLE, SignShape.OCTAGON):
            return SignType.INFORMATION
    if color == "yellow" and shape in (SignShape.DIAMOND, SignShape.TRIANGLE_UP, SignShape.TRIANGLE_DOWN):
        return SignType.WARNING
    return SignType.UNKNOWN


class ColorShapeSignRecognizer(TrafficSignRecognizer):
    def __init__(self, config: Optional[SignRecognizerConfig] = None) -> None:
        self.config = config or SignRecognizerConfig()

    def color_masks(self, hsv: np.ndarray) -> dict[str, np.ndarray]:
        c = self.config
        return {
            "red": hue_mask(hsv, *c.red_hue, c.s_min, c.v_min),
            "blue": hue_mask(hsv, *c.blue_hue, c.s_min, c.v_min),
            "yellow": hue_mask(hsv, *c.yellow_hue, c.s_min, c.v_min),
        }

    def recognize(self, image: np.ndarray, header: Header) -> list[Detection]:
        cfg = self.config
        if image.ndim != 3:
            return []
        hsv = rgb_to_hsv(image)
        dets = []
        for color, m in self.color_masks(hsv).items():
            labels, n = connected_components(m, 8)
            for k, box, area in component_boxes(labels, n):
                if area < cfg.min_area_px or abs(np.log(box.h / box.w)) > cfg.max_aspect_dev:
                    continue
                u0, v0 = int(box.u), int(box.v)
                blob = labels[v0:v0 + int(box.h), u0:u0 + int(box.w)] == k
                filled = fill_holes(blob)
                shape, fill = classify_shape(filled)
                interior = 1.0 - blob.sum() / max(1, filled.sum())
                ring = interior > cfg.ring_interior_min
                # white horizontal bar through the center (no-entry)
                hh = filled.shape[0]
                band = slice(int(hh * 0.42), max(int(hh * 0.42) + 1, int(hh * 0.58)))
                center_bar = bool(interior > 0.1 and (~blob[band][filled[band]]).mean() > 0.6
                                  and not ring) if filled[band].any() else False
                stype = sign_type_from(color, shape, ring, center_bar)
                shape_conf = 1.0 - min(1.0, abs(np.log(box.h / box.w)) / cfg.max_aspect_dev) * 0.5
                score = float(np.clip(shape_conf * (0.5 if stype == SignType.UNKNOWN else 1.0)
                                      * (1 - np.exp(-area / 150.0)), 0, 1))
                dets.append(Detection(header, ObjectClass.TRAFFIC_SIGN, score, box2d=box,
                                      attributes={"sign_type": stype, "sign_shape": shape, "color": color,
                                                  "fill": fill, "ring": ring}))
        return nms_detections(dets, cfg.nms_iou, class_aware=False)
        # TODO(perception): learned detector+classifier (GTSRB-style) behind TrafficSignRecognizer,
        # TODO(perception): OCR of speed values, 3D localization via size prior, map association.


@dataclass
class SignVoterConfig:
    iou_match: float = 0.3
    decay: float = 0.8  # per-frame vote decay
    min_votes: float = 1.5
    min_ratio: float = 0.6
    max_missed: int = 5


@dataclass
class _SignHypothesis:
    box: BoundingBox2D
    votes: dict = field(default_factory=dict)
    missed: int = 0
    last: Optional[Detection] = None


class SignVoter:
    """Temporal voting over per-frame sign detections (associated by IoU)."""

    def __init__(self, config: Optional[SignVoterConfig] = None) -> None:
        self.config = config or SignVoterConfig()
        self.hyps: list[_SignHypothesis] = []

    def update(self, dets: Sequence[Detection]) -> list[Detection]:
        cfg = self.config
        dets = [d for d in dets if d.box2d is not None]
        for h in self.hyps:
            for k in h.votes:
                h.votes[k] *= cfg.decay
        used_d: set[int] = set()
        used_h: set[int] = set()
        if self.hyps and dets:
            ious = iou_matrix(boxes_to_array([h.box for h in self.hyps]), boxes_to_array([d.box2d for d in dets]))
            for flat in np.argsort(-ious, axis=None):  # greedy best-IoU-first matching
                i, j = np.unravel_index(flat, ious.shape)
                if ious[i, j] < cfg.iou_match:
                    break
                if i in used_h or j in used_d:
                    continue
                used_h.add(int(i))
                used_d.add(int(j))
                self._add(self.hyps[i], dets[j])
        for i, h in enumerate(self.hyps):
            if i not in used_h:
                h.missed += 1
        for j, d in enumerate(dets):
            if j not in used_d:
                h = _SignHypothesis(d.box2d)
                self._add(h, d)
                self.hyps.append(h)
        self.hyps = [h for h in self.hyps if h.missed <= cfg.max_missed]
        out = []
        for h in self.hyps:
            total = sum(h.votes.values())
            if total <= 0:
                continue
            best = max(h.votes, key=h.votes.get)
            if h.votes[best] >= cfg.min_votes and h.votes[best] / total >= cfg.min_ratio and h.missed == 0:
                attrs = dict(h.last.attributes, sign_type=best, vote_ratio=h.votes[best] / total)
                out.append(Detection(h.last.header, ObjectClass.TRAFFIC_SIGN, float(h.votes[best] / total),
                                     box2d=h.box, attributes=attrs))
        return out

    @staticmethod
    def _add(h: _SignHypothesis, d: Detection) -> None:
        st = d.attributes.get("sign_type", SignType.UNKNOWN)
        h.votes[st] = h.votes.get(st, 0.0) + d.score
        h.box = d.box2d
        h.missed = 0
        h.last = d
