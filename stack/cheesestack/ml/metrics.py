"""Model validation metrics.

Classification: accuracy, confusion matrix, per-class / macro / micro precision,
recall, F1. Detection: IoU matrix, VOC/COCO-style average precision, mAP over
classes and IoU thresholds (greedy score-ordered matching, as in pycocotools).
Lanes: TuSimple-style point accuracy with lane-level precision/recall/F1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .annotation import BoxLabel


# --------------------------------------------------------------------------- classification


def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred)) if y_true.size else 0.0


def confusion_matrix(y_true, y_pred, num_classes: Optional[int] = None) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    n = num_classes or int(max(y_true.max(initial=-1), y_pred.max(initial=-1)) + 1)
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


@dataclass
class PRF:
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    support: np.ndarray
    macro_precision: float
    macro_recall: float
    macro_f1: float
    micro_f1: float


def precision_recall_f1(y_true, y_pred, num_classes: Optional[int] = None) -> PRF:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    support = cm.sum(axis=1)
    present = support > 0
    micro = tp.sum() / max(cm.sum(), 1)  # single-label: micro P = R = F1 = accuracy
    m = (lambda a: float(a[present].mean()) if present.any() else 0.0)
    return PRF(p, r, f1, support, m(p), m(r), m(f1), float(micro))


# --------------------------------------------------------------------------- detection


def iou_matrix(a_uvwh: np.ndarray, b_uvwh: np.ndarray) -> np.ndarray:
    a = np.asarray(a_uvwh, float).reshape(-1, 4)
    b = np.asarray(b_uvwh, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def average_precision(recall: np.ndarray, precision: np.ndarray, method: str = "coco101") -> float:
    """Area under the PR curve with monotone precision envelope.

    ``coco101``: mean interpolated precision at 101 recall points; ``voc``: all-point.
    """
    recall = np.asarray(recall, float)
    precision = np.asarray(precision, float)
    if recall.size == 0:
        return 0.0
    if method == "voc":
        r = np.concatenate([[0.0], recall, [1.0]])
        p = np.concatenate([[0.0], precision, [0.0]])
        p = np.maximum.accumulate(p[::-1])[::-1]
        idx = np.flatnonzero(r[1:] != r[:-1]) + 1
        return float(np.sum((r[idx] - r[idx - 1]) * p[idx]))
    p_env = np.maximum.accumulate(precision[::-1])[::-1]
    pts = np.linspace(0, 1, 101)
    inds = np.searchsorted(recall, pts, side="left")
    vals = np.where(inds < len(recall), p_env[np.minimum(inds, len(recall) - 1)], 0.0)
    return float(vals.mean())


def _match_class(preds: list[tuple[int, BoxLabel]], gts: dict[int, list[BoxLabel]], thr: float):
    """Greedy matching by descending score. Returns (tp flags sorted by score, n_gt)."""
    n_gt = sum(len(v) for v in gts.values())
    order = sorted(range(len(preds)), key=lambda i: -preds[i][1].score)
    used = {k: np.zeros(len(v), bool) for k, v in gts.items()}
    tp = np.zeros(len(order))
    for rank, i in enumerate(order):
        img, p = preds[i]
        g = gts.get(img, [])
        if not g:
            continue
        ious = iou_matrix(np.array([[p.u, p.v, p.w, p.h]]), np.array([[b.u, b.v, b.w, b.h] for b in g]))[0]
        ious[used[img]] = -1.0
        j = int(np.argmax(ious))
        if ious[j] >= thr:
            used[img][j] = True
            tp[rank] = 1
    return tp, n_gt


@dataclass
class DetectionReport:
    map: float  # mean over classes and IoU thresholds
    ap_per_class: dict  # class -> mean AP over thresholds
    ap_per_threshold: dict  # thr -> mAP over classes


def mean_average_precision(predictions: Sequence[Sequence[BoxLabel]], ground_truth: Sequence[Sequence[BoxLabel]],
                           iou_thresholds: Sequence[float] = (0.5,), classes: Optional[Sequence[str]] = None,
                           method: str = "coco101") -> DetectionReport:
    """mAP over images. ``predictions[i]`` / ``ground_truth[i]`` are box lists for image i.

    Classes with no ground truth are skipped (as in COCO).
    # TODO(ml): area ranges (small/medium/large), maxDets, crowd/ignore regions.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground_truth must align per image")
    classes = list(classes) if classes is not None else sorted({b.cls for g in ground_truth for b in g})
    ap = np.full((len(classes), len(iou_thresholds)), np.nan)
    for ci, c in enumerate(classes):
        preds = [(i, p) for i, ps in enumerate(predictions) for p in ps if p.cls == c]
        gts = {i: [b for b in g if b.cls == c] for i, g in enumerate(ground_truth)}
        gts = {k: v for k, v in gts.items() if v}
        if not gts:
            continue
        for ti, thr in enumerate(iou_thresholds):
            tp, n_gt = _match_class(preds, gts, thr)
            if len(tp) == 0:
                ap[ci, ti] = 0.0
                continue
            ctp = np.cumsum(tp)
            cfp = np.cumsum(1 - tp)
            rec = ctp / n_gt
            prec = ctp / np.maximum(ctp + cfp, 1e-12)
            ap[ci, ti] = average_precision(rec, prec, method)
    valid = ~np.isnan(ap[:, 0]) if len(classes) else np.array([], bool)
    per_class = {c: float(np.mean(ap[i])) for i, c in enumerate(classes) if valid[i]}
    per_thr = {float(t): float(np.mean(ap[valid, ti])) if valid.any() else 0.0
               for ti, t in enumerate(iou_thresholds)}
    total = float(np.mean(ap[valid])) if valid.any() else 0.0
    return DetectionReport(total, per_class, per_thr)


# --------------------------------------------------------------------------- lanes


@dataclass
class LaneMetrics:
    accuracy: float  # mean point accuracy over GT lanes (TuSimple "Accuracy")
    precision: float
    recall: float
    f1: float
    fp: int
    fn: int
    matched: int


def _u_at_rows(lane: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Interpolate a lane's u at image rows v; NaN outside its vertical extent."""
    lane = np.asarray(lane, float).reshape(-1, 2)
    order = np.argsort(lane[:, 1])
    v, u = lane[order, 1], lane[order, 0]
    out = np.interp(rows, v, u)
    out[(rows < v[0]) | (rows > v[-1])] = np.nan
    return out


def lane_point_metrics(pred_lanes: Sequence[np.ndarray], gt_lanes: Sequence[np.ndarray],
                       rows: Optional[np.ndarray] = None, pixel_thresh: float = 20.0,
                       match_thresh: float = 0.85) -> LaneMetrics:
    """TuSimple-style lane evaluation for one image.

    Each lane is an (N, 2) array of (u, v). Lanes are sampled at ``rows`` (default:
    the union of GT v's). A GT point is correct if the matched prediction's u at that
    row is within ``pixel_thresh``. A GT lane is matched (TP) if its point accuracy
    >= ``match_thresh``; each prediction can match at most one GT lane (greedy).
    """
    gts = [np.asarray(g, float).reshape(-1, 2) for g in gt_lanes if len(g) >= 2]
    preds = [np.asarray(p, float).reshape(-1, 2) for p in pred_lanes if len(p) >= 2]
    if rows is None:
        rows = np.unique(np.concatenate([g[:, 1] for g in gts])) if gts else np.array([])
    if not gts:
        return LaneMetrics(1.0 if not preds else 0.0, 0.0 if preds else 1.0, 1.0, 0.0 if preds else 1.0,
                           len(preds), 0, 0)
    acc = np.zeros((len(gts), max(len(preds), 1)))
    for i, g in enumerate(gts):
        gu = _u_at_rows(g, rows)
        valid = ~np.isnan(gu)
        for j, p in enumerate(preds):
            pu = _u_at_rows(p, rows)
            ok = valid & ~np.isnan(pu) & (np.abs(pu - gu) <= pixel_thresh)
            acc[i, j] = ok.sum() / max(valid.sum(), 1)
    if not preds:
        acc[:] = 0.0
    used = set()
    best_acc = np.zeros(len(gts))
    matched = 0
    for flat in np.argsort(-acc, axis=None):
        i, j = np.unravel_index(flat, acc.shape)
        if best_acc[i] > 0 or j in used or not preds or acc[i, j] <= 0:
            continue
        best_acc[i] = acc[i, j]
        used.add(j)
        if acc[i, j] >= match_thresh:
            matched += 1
    fp = len(preds) - matched
    fn = len(gts) - matched
    prec = matched / len(preds) if preds else 0.0
    rec = matched / len(gts)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return LaneMetrics(float(best_acc.mean()), prec, rec, f1, fp, fn, matched)


def evaluate_classifier(model, X: np.ndarray, y: np.ndarray, num_classes: Optional[int] = None) -> dict:
    """Convenience report for any object with ``predict``."""
    pred = model.predict(X)
    prf = precision_recall_f1(y, pred, num_classes)
    return {"accuracy": accuracy(y, pred), "macro_f1": prf.macro_f1,
            "confusion": confusion_matrix(y, pred, num_classes), "per_class_f1": prf.f1}
