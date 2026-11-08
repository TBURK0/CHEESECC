import math

import numpy as np
import pytest

from cheesestack.core import Header, ObjectClass, Pose2D, TrackedObject
from cheesestack.prediction import (Lane, LaneChangePredictor, LogisticLaneChangeModel,
                                    extract_lane_change_features, project_to_lane)


def _lanes():
    xs = np.linspace(-50, 200, 26)
    return [Lane(np.c_[xs, np.full_like(xs, y)], 3.5, name) for y, name in ((0.0, "ego"), (3.5, "L"), (-3.5, "R"))]


def test_projection_signs():
    lane = _lanes()[0]
    pr = project_to_lane(lane, (55.0, 1.0))
    assert pr.s == pytest.approx(105.0) and pr.d == pytest.approx(1.0) and pr.heading == pytest.approx(0.0)
    assert project_to_lane(lane, (10.0, -2.0)).d == pytest.approx(-2.0)
    assert project_to_lane(lane, (-60.0, 0.5)).s == pytest.approx(-10.0)  # extrapolated


def test_features_and_predictions():
    lanes = _lanes()
    pred = LaneChangePredictor()
    v, ang = 20.0, math.radians(4)
    drift_left = pred.predict_state(lanes, 0.0, 0.8, ang, v)
    drift_right = pred.predict_state(lanes, 0.0, -0.8, -ang, v)
    straight = pred.predict_state(lanes, 0.0, 0.0, 0.0, v)
    assert straight.most_likely == "keep" and straight.p_keep > 0.85
    assert drift_left.most_likely == "left" and drift_left.left_lane == "L"
    assert drift_right.most_likely == "right" and drift_right.right_lane == "R"
    assert drift_left.p_keep + drift_left.p_left + drift_left.p_right == pytest.approx(1.0)
    f = extract_lane_change_features(lanes[0], 0.0, 0.8, ang, v)
    assert f.d_dot == pytest.approx(v * math.sin(ang)) and f.inv_tlc > 0


def test_no_neighbor_moves_mass_to_keep_and_tracked_object():
    lanes = _lanes()[:1]
    p = LaneChangePredictor().predict_state(lanes, 0.0, 0.8, math.radians(4), 20.0)
    assert p.p_left == 0.0 and p.p_keep == pytest.approx(1 - p.p_right)
    obj = TrackedObject(3, Header(0.0, "map"), ObjectClass.CAR, Pose2D(0, 0.8, 0),
                        np.array([20 * math.cos(0.07), 20 * math.sin(0.07)]))
    assert LaneChangePredictor().predict(obj, _lanes()).most_likely == "left"


def test_fit_learns_separable_data():
    rng = np.random.default_rng(0)
    X, y = [], []
    for cls, sign in ((0, 0.0), (1, 1.0), (2, -1.0)):
        base = np.array([0.6, 0.8, 0.05, 1.0]) * sign
        X.append(base + 0.1 * rng.standard_normal((50, 4)))
        y += [cls] * 50
    X = np.vstack(X)
    m = LogisticLaneChangeModel().fit(X, np.array(y), iters=300)
    acc = np.mean(np.argmax(m.predict_proba(X), axis=1) == np.array(y))
    assert acc > 0.95
