import numpy as np
import pytest

from cheesestack.core import Header, ObjectClass, Pose2D, TrackedObject
from cheesestack.prediction import ObjectPredictor, tracked_object_state


def _obj(cls=ObjectClass.CAR, v=(10.0, 0.0), t=0.0, x=0.0, y=0.0, yaw=0.0, tid=1, cov=None):
    return TrackedObject(tid, Header(t, "map"), cls, Pose2D(x, y, yaw), np.array(v, float), covariance=cov)


def test_probabilities_normalised_and_class_specific():
    p = ObjectPredictor()
    car = p.predict(_obj())
    assert len(car.hypotheses) >= 2
    assert sum(h.probability for h in car.hypotheses) == pytest.approx(1.0)
    assert car.most_likely.label == "keep"
    ped = p.predict(_obj(ObjectClass.PEDESTRIAN, v=(1.2, 0.0)))
    assert {h.label for h in ped.hypotheses} <= {"keep", "stop"}


def test_stationary_and_static():
    p = ObjectPredictor()
    for o in (_obj(v=(0.0, 0.0)), _obj(ObjectClass.STATIC_OBSTACLE, v=(0.0, 0.0))):
        pred = p.predict(o)
        assert len(pred.hypotheses) == 1 and pred.hypotheses[0].label == "stationary"
        assert np.allclose(pred.hypotheses[0].positions[-1], [0, 0])


def test_history_estimates_yaw_rate_and_shifts_probability():
    p = ObjectPredictor()
    w, v = 0.3, 8.0
    for k in range(6):
        t = 0.1 * k
        yaw = w * t
        p.observe(_obj(v=(v * np.cos(yaw), v * np.sin(yaw)), t=t, yaw=yaw))
    w_est, a_est = p.estimate_derivatives(1)
    assert w_est == pytest.approx(w, abs=1e-6) and a_est == pytest.approx(0.0, abs=1e-6)
    turning = p.predict(_obj(v=(v * np.cos(0.5 * w), v * np.sin(0.5 * w)), t=0.5, yaw=0.15))
    straight = ObjectPredictor().predict(_obj(v=(v, 0.0)))
    pl = lambda pr: sum(h.probability for h in pr.hypotheses if h.label == "turn_left")
    pr_ = lambda pr: sum(h.probability for h in pr.hypotheses if h.label == "turn_right")
    assert pl(turning) > pl(straight)
    assert pl(turning) > pr_(turning)


def test_covariance_conversion():
    cov = np.diag([0.1, 0.2, 1.0, 0.04])
    s, P = tracked_object_state(_obj(v=(10.0, 0.0), cov=cov))
    assert s[3] == pytest.approx(10.0)
    assert P[3, 3] == pytest.approx(1.0)  # var(v) = var(vx) when heading +x
    assert P[2, 2] == pytest.approx(0.04 / 100)  # var(yaw) = var(vy)/v^2
