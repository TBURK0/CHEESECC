import numpy as np

from cheesestack.ml.annotation import BoxLabel
from cheesestack.ml.metrics import (
    accuracy, average_precision, confusion_matrix, evaluate_classifier, iou_matrix, lane_point_metrics,
    mean_average_precision, precision_recall_f1,
)


def test_classification_metrics():
    yt = [0, 0, 1, 1, 2, 2]
    yp = [0, 1, 1, 1, 2, 0]
    cm = confusion_matrix(yt, yp, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    prf = precision_recall_f1(yt, yp, 3)
    assert np.allclose(prf.precision, [0.5, 2 / 3, 1.0])
    assert np.allclose(prf.recall, [0.5, 1.0, 0.5])
    assert np.isclose(prf.f1[1], 0.8)
    assert np.isclose(prf.micro_f1, accuracy(yt, yp)) and np.isclose(accuracy(yt, yp), 4 / 6)


def test_iou_matrix():
    m = iou_matrix([[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 0, 10, 10], [20, 20, 1, 1]])
    assert np.allclose(m, [[1.0, 50 / 150, 0.0]])


def test_average_precision_perfect_and_voc():
    assert np.isclose(average_precision(np.array([0.5, 1.0]), np.array([1.0, 1.0])), 1.0)
    # one TP then one FP then TP: recall .5,.5,1  precision 1,.5,.667
    r, p = np.array([0.5, 0.5, 1.0]), np.array([1.0, 0.5, 2 / 3])
    assert np.isclose(average_precision(r, p, "voc"), 0.5 * 1 + 0.5 * 2 / 3)


def test_map_detection():
    gt = [[BoxLabel("car", 0, 0, 10, 10), BoxLabel("sign", 50, 50, 5, 5)], [BoxLabel("car", 20, 20, 10, 10)]]
    perfect = [[BoxLabel(b.cls, b.u, b.v, b.w, b.h, 0.9) for b in g] for g in gt]
    rep = mean_average_precision(perfect, gt, iou_thresholds=(0.5, 0.75))
    assert np.isclose(rep.map, 1.0) and set(rep.ap_per_class) == {"car", "sign"}
    shifted = [[BoxLabel("car", 3, 0, 10, 10, 0.9)], [BoxLabel("car", 20, 20, 10, 10, 0.8),
                                                      BoxLabel("car", 60, 60, 10, 10, 0.95)]]
    rep2 = mean_average_precision(shifted, gt, iou_thresholds=(0.5, 0.75))
    assert rep2.ap_per_class["sign"] == 0.0
    assert rep2.ap_per_threshold[0.5] > rep2.ap_per_threshold[0.75]
    assert 0 < rep2.map < 1


def test_lane_metrics():
    rows = np.arange(100, 200, 10, dtype=float)
    g1 = np.stack([np.full_like(rows, 50.0), rows], 1)
    g2 = np.stack([150 + 0.5 * (rows - 100), rows], 1)
    perfect = lane_point_metrics([g1, g2], [g1, g2])
    assert perfect.accuracy == 1.0 and perfect.f1 == 1.0
    off = g1 + [40, 0]
    junk = np.stack([np.full_like(rows, 300.0), rows], 1)
    m = lane_point_metrics([g1 + [5, 0], off, junk], [g1, g2], pixel_thresh=20)
    assert m.matched == 1 and m.fn == 1 and m.fp == 2
    assert np.isclose(m.recall, 0.5) and np.isclose(m.precision, 1 / 3)
    assert lane_point_metrics([], [g1]).recall == 0.0


def test_evaluate_classifier():
    class Const:
        def predict(self, X):
            return np.zeros(len(X), int)
    rep = evaluate_classifier(Const(), np.zeros((4, 1)), np.array([0, 0, 1, 1]), 2)
    assert rep["accuracy"] == 0.5 and rep["confusion"].tolist() == [[2, 0], [2, 0]]
