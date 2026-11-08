"""Annotation tools: label schemas, validation, format converters, pre-labelling.

Label primitives (all image coordinates in pixels, u right / v down, continuous
coordinates so an image spans [0, W] x [0, H]):

- ``BoxLabel``      axis-aligned box (u, v, w, h), top-left origin (matches core.BoundingBox2D)
- ``PolylineLabel`` ordered (N, 2) points, e.g. lane boundaries
- ``MaskLabel``     boolean HxW mask, e.g. drivable area
- image-level class tags, e.g. ``"night"``, ``"rain"``, ``"light_state=red"``

Converters: COCO-style JSON (boxes, uncompressed RLE masks, polylines as a custom
``"polyline"`` field) and YOLO txt (normalised cx cy w h per line).

Pre-labelling: ``PreLabeler`` is the hook where a trained detector plugs in. The
baseline ``ColorBlobPreLabeler`` finds saturated colour blobs (good enough to
seed traffic-light boxes for a human to correct).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol, Sequence

import numpy as np


# --------------------------------------------------------------------------- labels


@dataclass
class BoxLabel:
    cls: str
    u: float
    v: float
    w: float
    h: float
    score: float = 1.0
    attributes: dict = field(default_factory=dict)

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def xyxy(self) -> np.ndarray:
        return np.array([self.u, self.v, self.u + self.w, self.v + self.h], dtype=float)

    def to_dict(self) -> dict:
        return {"cls": self.cls, "u": float(self.u), "v": float(self.v), "w": float(self.w),
                "h": float(self.h), "score": float(self.score), "attributes": dict(self.attributes)}

    @classmethod
    def from_dict(cls, d: dict) -> "BoxLabel":
        return cls(d["cls"], d["u"], d["v"], d["w"], d["h"], d.get("score", 1.0), dict(d.get("attributes", {})))


@dataclass
class PolylineLabel:
    cls: str
    points: np.ndarray  # (N, 2) u, v
    attributes: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float).reshape(-1, 2)

    def to_dict(self) -> dict:
        return {"cls": self.cls, "points": self.points.tolist(), "attributes": dict(self.attributes)}

    @classmethod
    def from_dict(cls, d: dict) -> "PolylineLabel":
        return cls(d["cls"], np.asarray(d["points"], dtype=float), dict(d.get("attributes", {})))


@dataclass
class MaskLabel:
    cls: str
    mask: np.ndarray  # HxW bool

    def __post_init__(self) -> None:
        self.mask = np.asarray(self.mask, dtype=bool)

    def to_dict(self) -> dict:
        return {"cls": self.cls, "rle": mask_to_rle(self.mask)}

    @classmethod
    def from_dict(cls, d: dict) -> "MaskLabel":
        return cls(d["cls"], rle_to_mask(d["rle"]))


@dataclass
class Annotation:
    """All labels for one image."""

    image_id: str
    width: int
    height: int
    boxes: list[BoxLabel] = field(default_factory=list)
    polylines: list[PolylineLabel] = field(default_factory=list)
    masks: list[MaskLabel] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    needs_review: bool = False

    def copy(self) -> "Annotation":
        return Annotation.from_dict(self.to_dict())

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id, "width": int(self.width), "height": int(self.height),
            "boxes": [b.to_dict() for b in self.boxes],
            "polylines": [p.to_dict() for p in self.polylines],
            "masks": [m.to_dict() for m in self.masks],
            "tags": list(self.tags), "needs_review": bool(self.needs_review),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Annotation":
        return cls(
            d["image_id"], int(d["width"]), int(d["height"]),
            [BoxLabel.from_dict(b) for b in d.get("boxes", [])],
            [PolylineLabel.from_dict(p) for p in d.get("polylines", [])],
            [MaskLabel.from_dict(m) for m in d.get("masks", [])],
            list(d.get("tags", [])), bool(d.get("needs_review", False)),
        )


# --------------------------------------------------------------------------- RLE


def mask_to_rle(mask: np.ndarray) -> dict:
    """Uncompressed COCO RLE: column-major run lengths, starting with a 0-run."""
    m = np.asarray(mask, dtype=bool)
    flat = m.flatten(order="F").astype(np.int8)
    if flat.size == 0:
        return {"size": list(m.shape), "counts": []}
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    runs = np.diff(bounds).tolist()
    if flat[0] == 1:
        runs = [0] + runs
    return {"size": [int(m.shape[0]), int(m.shape[1])], "counts": [int(r) for r in runs]}


def rle_to_mask(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for r in rle["counts"]:
        if val:
            flat[pos:pos + r] = True
        pos += r
        val = not val
    return flat.reshape((h, w), order="F")


# --------------------------------------------------------------------------- schema / validation


@dataclass
class LabelSchema:
    """Which classes are legal for each label kind. Empty set = kind not allowed."""

    box_classes: Sequence[str] = ("car", "truck", "bus", "pedestrian", "bicycle", "motorcycle",
                                  "traffic_sign", "traffic_light")
    polyline_classes: Sequence[str] = ("lane_solid", "lane_dashed", "road_edge", "stop_line")
    mask_classes: Sequence[str] = ("drivable",)
    tag_classes: Optional[Sequence[str]] = None  # None = any tag allowed
    min_box_size_px: float = 2.0
    min_polyline_points: int = 2

    def validate(self, ann: Annotation) -> list[str]:
        """Return a list of human-readable errors (empty list = valid)."""
        errs: list[str] = []
        W, H = ann.width, ann.height
        if W <= 0 or H <= 0:
            errs.append(f"non-positive image size {W}x{H}")
        for i, b in enumerate(ann.boxes):
            if b.cls not in self.box_classes:
                errs.append(f"box[{i}] unknown class '{b.cls}'")
            if b.w < self.min_box_size_px or b.h < self.min_box_size_px:
                errs.append(f"box[{i}] too small ({b.w:.1f}x{b.h:.1f})")
            if b.u < 0 or b.v < 0 or b.u + b.w > W + 1e-6 or b.v + b.h > H + 1e-6:
                errs.append(f"box[{i}] outside image")
            if not 0.0 <= b.score <= 1.0:
                errs.append(f"box[{i}] score {b.score} not in [0,1]")
        for i, p in enumerate(ann.polylines):
            if p.cls not in self.polyline_classes:
                errs.append(f"polyline[{i}] unknown class '{p.cls}'")
            if len(p.points) < self.min_polyline_points:
                errs.append(f"polyline[{i}] has {len(p.points)} points")
            if len(p.points) and (np.any(p.points < 0) or np.any(p.points[:, 0] > W) or np.any(p.points[:, 1] > H)):
                errs.append(f"polyline[{i}] outside image")
            if not np.all(np.isfinite(p.points)):
                errs.append(f"polyline[{i}] non-finite points")
        for i, m in enumerate(ann.masks):
            if m.cls not in self.mask_classes:
                errs.append(f"mask[{i}] unknown class '{m.cls}'")
            if m.mask.shape != (H, W):
                errs.append(f"mask[{i}] shape {m.mask.shape} != {(H, W)}")
        if self.tag_classes is not None:
            for t in ann.tags:
                if t not in self.tag_classes:
                    errs.append(f"unknown tag '{t}'")
        return errs


# --------------------------------------------------------------------------- COCO


def to_coco(annotations: Sequence[Annotation], schema: LabelSchema,
            file_names: Optional[dict[str, str]] = None) -> dict:
    """Export to a COCO-style dict. Category ids are 1-based over all schema classes."""
    file_names = file_names or {}
    all_classes = list(dict.fromkeys([*schema.box_classes, *schema.polyline_classes, *schema.mask_classes]))
    cat_id = {c: i + 1 for i, c in enumerate(all_classes)}
    images, anns = [], []
    ann_id = 1
    for img_idx, a in enumerate(annotations, start=1):
        images.append({"id": img_idx, "file_name": file_names.get(a.image_id, a.image_id),
                       "width": a.width, "height": a.height, "image_key": a.image_id, "tags": list(a.tags)})
        for b in a.boxes:
            anns.append({"id": ann_id, "image_id": img_idx, "category_id": cat_id[b.cls],
                         "bbox": [b.u, b.v, b.w, b.h], "area": b.area, "iscrowd": 0, "score": b.score,
                         "attributes": dict(b.attributes)})
            ann_id += 1
        for p in a.polylines:
            anns.append({"id": ann_id, "image_id": img_idx, "category_id": cat_id[p.cls],
                         "polyline": p.points.flatten().tolist(), "attributes": dict(p.attributes)})
            ann_id += 1
        for m in a.masks:
            ys, xs = np.nonzero(m.mask)
            bbox = [float(xs.min()), float(ys.min()), float(xs.max() - xs.min() + 1),
                    float(ys.max() - ys.min() + 1)] if xs.size else [0.0, 0.0, 0.0, 0.0]
            anns.append({"id": ann_id, "image_id": img_idx, "category_id": cat_id[m.cls],
                         "segmentation": mask_to_rle(m.mask), "bbox": bbox,
                         "area": float(m.mask.sum()), "iscrowd": 1})
            ann_id += 1
    cats = [{"id": cat_id[c], "name": c} for c in all_classes]
    return {"images": images, "annotations": anns, "categories": cats}


def from_coco(coco: dict) -> list[Annotation]:
    """Import a COCO-style dict produced by ``to_coco`` (or plain COCO detection JSON)."""
    names = {c["id"]: c["name"] for c in coco["categories"]}
    by_img: dict[int, Annotation] = {}
    for im in coco["images"]:
        by_img[im["id"]] = Annotation(im.get("image_key", im["file_name"]), im["width"], im["height"],
                                      tags=list(im.get("tags", [])))
    for a in coco["annotations"]:
        ann = by_img[a["image_id"]]
        cls = names[a["category_id"]]
        if "polyline" in a:
            ann.polylines.append(PolylineLabel(cls, np.asarray(a["polyline"], dtype=float).reshape(-1, 2),
                                               dict(a.get("attributes", {}))))
        elif isinstance(a.get("segmentation"), dict):
            ann.masks.append(MaskLabel(cls, rle_to_mask(a["segmentation"])))
        else:
            # TODO(ml): polygon segmentations and compressed RLE strings.
            u, v, w, h = a["bbox"]
            ann.boxes.append(BoxLabel(cls, u, v, w, h, a.get("score", 1.0), dict(a.get("attributes", {}))))
    return list(by_img.values())


# --------------------------------------------------------------------------- YOLO


def to_yolo(ann: Annotation, class_list: Sequence[str]) -> str:
    """YOLO txt: ``class_idx cx cy w h`` normalised to [0,1]; boxes only."""
    idx = {c: i for i, c in enumerate(class_list)}
    lines = []
    for b in ann.boxes:
        if b.cls not in idx:
            continue
        cx = (b.u + b.w / 2) / ann.width
        cy = (b.v + b.h / 2) / ann.height
        lines.append(f"{idx[b.cls]} {cx:.6f} {cy:.6f} {b.w / ann.width:.6f} {b.h / ann.height:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def from_yolo(text: str, class_list: Sequence[str], width: int, height: int, image_id: str) -> Annotation:
    ann = Annotation(image_id, width, height)
    for line in text.strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        k = int(parts[0])
        cx, cy, w, h = (float(x) for x in parts[1:5])
        score = float(parts[5]) if len(parts) > 5 else 1.0
        ann.boxes.append(BoxLabel(class_list[k], (cx - w / 2) * width, (cy - h / 2) * height,
                                  w * width, h * height, score))
    return ann


# --------------------------------------------------------------------------- pre-labelling


class PreLabeler(Protocol):
    """Hook for model-assisted labelling: image -> draft Annotation."""

    def prelabel(self, image: np.ndarray, image_id: str) -> Annotation: ...


@dataclass
class ColorBlobConfig:
    # (class name, attribute dict, rgb-predicate thresholds) are built into the labeler
    saturation_min: float = 0.5
    value_min: float = 0.6
    min_area_px: int = 4
    review_below_score: float = 0.8


class ColorBlobPreLabeler:
    """Baseline pre-labeler: saturated red/yellow/green blobs -> traffic_light boxes.

    Pixels are classified by dominant channel with HSV-like saturation/value gates,
    then 4-connected components become boxes. Score = mean saturation of the blob.
    # TODO(ml): replace with the team detector (ONNX/CoreML) behind the same interface.
    """

    def __init__(self, config: Optional[ColorBlobConfig] = None) -> None:
        self.config = config or ColorBlobConfig()

    def prelabel(self, image: np.ndarray, image_id: str) -> Annotation:
        img = np.asarray(image)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        if img.ndim != 3 or img.shape[2] < 3:
            raise ValueError("ColorBlobPreLabeler needs HxWx3 RGB")
        H, W = img.shape[:2]
        mx = img[..., :3].max(axis=2)
        mn = img[..., :3].min(axis=2)
        sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9), 0.0)
        bright = (sat >= self.config.saturation_min) & (mx >= self.config.value_min)
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        colors = {
            "red": bright & (r >= g) & (r >= b) & (g < 0.6 * r),
            "yellow": bright & (r > b) & (g > b) & (g >= 0.6 * r) & (r >= 0.6 * g),
            "green": bright & (g > r) & (g >= b) & (r < 0.6 * g),
        }
        ann = Annotation(image_id, W, H)
        for color, m in colors.items():
            for ys, xs in connected_components(m):
                if ys.size < self.config.min_area_px:
                    continue
                score = float(np.clip(sat[ys, xs].mean(), 0, 1))
                ann.boxes.append(BoxLabel("traffic_light", float(xs.min()), float(ys.min()),
                                          float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1),
                                          score, {"state": color, "prelabel": True}))
        ann.needs_review = any(b.score < self.config.review_below_score for b in ann.boxes)
        return ann


def connected_components(mask: np.ndarray) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """4-connected components of a bool mask (BFS). Yields (rows, cols) index arrays."""
    m = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(m)
    H, W = m.shape
    for sy, sx in zip(*np.nonzero(m)):
        if seen[sy, sx]:
            continue
        q = deque([(sy, sx)])
        seen[sy, sx] = True
        ys, xs = [], []
        while q:
            y, x = q.popleft()
            ys.append(y)
            xs.append(x)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < H and 0 <= nx < W and m[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        yield np.array(ys), np.array(xs)


def auto_label(images: dict[str, np.ndarray], labeler: PreLabeler) -> list[Annotation]:
    """Run a pre-labeler over many images; returns drafts (``needs_review`` flagged)."""
    return [labeler.prelabel(img, key) for key, img in images.items()]
