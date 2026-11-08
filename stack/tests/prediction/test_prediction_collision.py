import math

import numpy as np
import pytest

from cheesestack.prediction import (CollisionChecker, CollisionCheckerConfig, MotionModel,
                                    MotionModelType, OrientedBox, boxes_overlap)


def test_sat_basic_cases():
    a = OrientedBox(0, 0, 0, 4, 2)
    assert boxes_overlap(a, OrientedBox(3.9, 0, 0, 4, 2))[0]
    assert not boxes_overlap(a, OrientedBox(4.1, 0, 0, 4, 2))[0]
    # Rotated 45deg box: corner reaches 2 + sqrt(2) along x from its centre... check separation on its axis
    b = OrientedBox(3.3, 0, math.pi / 4, 2, 2)
    assert boxes_overlap(a, b)[0]  # its corner at x = 3.3 - 1.414 < 2
    assert not boxes_overlap(a, OrientedBox(3.5, 0, math.pi / 4, 2, 2))[0]
    hit, depth = boxes_overlap(a, OrientedBox(3.0, 0, 0, 4, 2))
    assert hit and depth == pytest.approx(1.0)
    assert boxes_overlap(a, OrientedBox(4.2, 0, 0, 4, 2), margin=0.15)[0]


def _cv(x, y, yaw, v, horizon=6.0, std=None):
    cov = None if std is None else np.eye(6) * std ** 2
    return MotionModel(MotionModelType.CV).rollout(np.array([x, y, yaw, v, 0, 0]), cov, horizon=horizon)


def test_head_on_earliest_time():
    ego = _cv(0, 0, 0, 10)
    obj = _cv(50, 0, math.pi, 10)
    r = CollisionChecker().check(ego, (4.5, 1.8), obj, (4.5, 1.8))
    # gap closes when centre distance = 4.5 -> (50 - 4.5) / 20 = 2.275 -> first grid step 2.3
    assert r.collides and r.time == pytest.approx(2.3, abs=1e-9)


def test_parallel_no_collision_and_probability_ordering():
    cc = CollisionChecker(CollisionCheckerConfig(n_samples=100, seed=1))
    ego = _cv(0, 0, 0, 10)
    side = _cv(0, 3.5, 0, 10)
    r = cc.check(ego, (4.5, 1.8), side, (4.5, 1.8), with_probability=True)
    assert not r.collides and r.min_distance == pytest.approx(3.5)
    near = cc.check(ego, (4.5, 1.8), _cv(0, 2.2, 0, 10), (4.5, 1.8), with_probability=True)
    assert near.probability > r.probability
    assert 0.0 <= r.probability <= 1.0


def test_multi_hypothesis_aggregation():
    ego = _cv(0, 0, 0, 10, std=0.01)
    hit = _cv(40, 0, math.pi, 10, std=0.01)
    miss = _cv(40, 10, math.pi, 10, std=0.01)
    hit.probability, miss.probability = 0.3, 0.7
    r = CollisionChecker(CollisionCheckerConfig(n_samples=50)).check_prediction(ego, (4.5, 1.8), [hit, miss], (4.5, 1.8))
    assert r.collides and 0.25 <= r.probability <= 0.35
