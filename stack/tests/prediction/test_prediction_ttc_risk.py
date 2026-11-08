import math

import numpy as np
import pytest

from cheesestack.core import Header, ObjectClass, Pose2D, TrackedObject, VehicleState
from cheesestack.prediction import (RiskScorer, TTCEstimator, closest_point_of_approach,
                                    ttc_2d_circles, ttc_constant_accel, ttc_longitudinal,
                                    uncertainty_from_cov)


def test_1d_ttc():
    assert ttc_longitudinal(20, 15, 5) == pytest.approx(2.0)
    assert ttc_longitudinal(20, 5, 15) == math.inf
    assert ttc_longitudinal(-1, 5, 5) == 0.0


def test_enhanced_ttc():
    # Lead stopped, follower constant 10 m/s: same as 1-D
    assert ttc_constant_accel(20, 10, 0) == pytest.approx(2.0)
    # Same speeds, lead brakes at 5 m/s^2: gap(t) = 2 - 2.5 t^2 -> t = sqrt(0.8) before lead stops
    assert ttc_constant_accel(2, 10, 10, 0, -5) == pytest.approx(math.sqrt(0.8))
    # Gap 20: lead stops at t=2 (gap 10 left), then closing at 10 m/s -> t = 3.0
    assert ttc_constant_accel(20, 10, 10, 0, -5) == pytest.approx(3.0)
    # Without stop-awareness the lead "reverses": 20 - 2.5 t^2 = 0 -> t = sqrt(8)
    assert ttc_constant_accel(20, 10, 10, 0, -5, stop_aware=False) == pytest.approx(math.sqrt(8))
    # Follower braking hard enough never reaches
    assert ttc_constant_accel(20, 10, 0, -5, 0) == math.inf  # stops after 10 m


def test_cpa_and_circles():
    t, d = closest_point_of_approach([0, 0], [10, 0], [50, 5], [0, 0])
    assert t == pytest.approx(5.0) and d == pytest.approx(5.0)
    assert ttc_2d_circles([0, 0], [10, 0], 1, [50, 0], [0, 0], 1) == pytest.approx(4.8)
    assert ttc_2d_circles([0, 0], [10, 0], 1, [50, 5], [0, 0], 1) == math.inf


def test_ttc_estimator_ego_frame():
    ego = VehicleState(Header(0.0, "map"), Pose2D(10, 10, math.pi / 2), v=10.0)
    # object 31.5 m ahead of base_link along +y, stationary
    obj = TrackedObject(1, Header(0.0, "map"), ObjectClass.CAR, Pose2D(10, 41.5, math.pi / 2), np.zeros(2))
    r = TTCEstimator().estimate(ego, obj)
    # ego centre = base_link + (2.25-1.0) = 1.25; gap = 31.5 - 1.25 - 2.25 - 2.25 = 25.75
    assert r.in_path and r.gap == pytest.approx(25.75)
    assert r.ttc == pytest.approx(2.575) and r.ttc_accel == pytest.approx(2.575)
    side = TrackedObject(2, Header(0.0, "map"), ObjectClass.CAR, Pose2D(6.0, 41.5, math.pi / 2), np.zeros(2))
    assert not TTCEstimator().estimate(ego, side).in_path


def test_risk_scoring_monotone():
    rs = RiskScorer()
    assert rs.score().risk == 0.0 and rs.score().level == "low"
    hi = rs.score(ttc=1.0, collision_probability=0.8, cls=ObjectClass.PEDESTRIAN)
    lo = rs.score(ttc=5.0, collision_probability=0.0, cls=ObjectClass.PEDESTRIAN)
    car = rs.score(ttc=1.0, collision_probability=0.8, cls=ObjectClass.CAR)
    assert 0 <= lo.risk < car.risk < hi.risk <= 1.0
    assert hi.level == "critical"
    unc = rs.score(ttc=3.0, cls=ObjectClass.CAR, uncertainty=0.8)
    assert unc.risk > rs.score(ttc=3.0, cls=ObjectClass.CAR).risk
    assert rs.ttc_term(1.5) == 1.0 and rs.ttc_term(6.0) == 0.0 and 0 < rs.ttc_term(3.0) < 1
    assert uncertainty_from_cov(np.eye(2) * 0.0) == 0.0 and 0 < uncertainty_from_cov(np.eye(4)) < 1
    assert rs.aggregate([hi, lo]) >= hi.risk
