import math

import pytest

from cheesestack.planning import OtherVehicle, OvertakeContext, OvertakeDecision, OvertakeEvaluator
from cheesestack.planning.overtake import pass_time


def test_pass_time_kinematics():
    # pure cruise with 5 m/s advantage over 50 m -> 10 s
    assert pass_time(50, 20, 15, 1.0, 20) == pytest.approx(10.0)
    # accelerating only: 0.5 * 2 * t^2 = 16 -> 4 s
    assert pass_time(16, 15, 15, 2.0, 100) == pytest.approx(4.0)
    assert math.isinf(pass_time(10, 15, 15, 1.0, 15))


def _ctx(**kw):
    base = dict(ego_speed=18.0, desired_speed=27.0, lead=OtherVehicle(15.0, 15.0))
    base.update(kw)
    return OvertakeContext(**base)


def test_clear_road_overtakes():
    r = OvertakeEvaluator().evaluate(_ctx())
    assert r.decision == OvertakeDecision.OVERTAKE, r.reasons
    assert r.benefit > r.cost and 0 < r.pass_time_s < 20


def test_oncoming_blocks():
    r = OvertakeEvaluator().evaluate(_ctx(oncoming=[OtherVehicle(300.0, 25.0)]))
    assert r.decision == OvertakeDecision.FOLLOW and "oncoming traffic" in r.reasons
    far = OvertakeEvaluator().evaluate(_ctx(oncoming=[OtherVehicle(2000.0, 25.0)]))
    assert far.decision == OvertakeDecision.OVERTAKE, far.reasons


def test_other_blockers():
    ev = OvertakeEvaluator()
    assert "insufficient speed gain" in ev.evaluate(_ctx(desired_speed=16.0)).reasons
    assert "passing not allowed" in ev.evaluate(_ctx(passing_allowed=False)).reasons
    assert "insufficient sight distance" in ev.evaluate(_ctx(sight_distance_m=150.0)).reasons
    r = ev.evaluate(_ctx(passing_lane=[OtherVehicle(-30.0, 35.0)]))
    assert "fast vehicle approaching in passing lane" in r.reasons
    r = ev.evaluate(_ctx(passing_lane=[OtherVehicle(30.0, 16.0)]))
    assert "passing lane blocked ahead" in r.reasons
