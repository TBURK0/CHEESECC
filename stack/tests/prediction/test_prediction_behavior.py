import math

import numpy as np

from cheesestack.core import Header, ObjectClass, Pose2D, TrackedObject
from cheesestack.prediction import BehaviorFeatures, BehaviorForecaster, Intent, Lane


def _run(gen, n=15, ego_lane=None):
    bf = BehaviorForecaster()
    out = None
    for k in range(n):
        t = 0.1 * k
        x, y, vx, vy = gen(t)
        obj = TrackedObject(7, Header(t, "map"), ObjectClass.CAR, Pose2D(x, y, math.atan2(vy, vx)), np.array([vx, vy]))
        out = bf.update(obj, ego_lane)
    return out


def test_keep_lane():
    out = _run(lambda t: (10 * t, 0.0, 10.0, 0.0))
    assert out.most_likely == Intent.KEEP_LANE
    assert abs(sum(out.probabilities.values()) - 1.0) < 1e-9


def test_turn_left_and_right():
    w, v = 0.3, 8.0
    left = _run(lambda t: (0, 0, v * math.cos(w * t), v * math.sin(w * t)))
    right = _run(lambda t: (0, 0, v * math.cos(-w * t), v * math.sin(-w * t)))
    assert left.most_likely == Intent.TURN_LEFT
    assert right.most_likely == Intent.TURN_RIGHT


def test_stop_vs_yield():
    stop = _run(lambda t: (0, 0, max(6 - 2.5 * t, 0.0), 0.0), n=12)
    yld = _run(lambda t: (0, 0, 10 - 1.0 * t, 0.0))
    assert stop.most_likely == Intent.STOP
    assert yld.most_likely == Intent.YIELD


def test_cut_in_requires_ego_lane():
    xs = np.linspace(-10, 100, 12)
    lane = Lane(np.c_[xs, np.zeros_like(xs)])
    gen = lambda t: (10 * t, 3.5 - 0.8 * t, 10.0, -0.8)
    assert _run(gen, ego_lane=lane).most_likely == Intent.CUT_IN
    assert _run(gen).most_likely != Intent.CUT_IN


def test_feature_api_direct():
    bf = BehaviorForecaster()
    for _ in range(5):
        f = bf.update_features(1, BehaviorFeatures(speed=0.2, accel=-0.5, yaw_rate=0.0))
    assert f.most_likely == Intent.STOP
