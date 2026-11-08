import numpy as np
import pytest

from cheesestack.core import Header, ObjectClass, Pose2D, TrackedObject
from cheesestack.planning import GapAssessor, LaneObject, MergePlanner


def _obj(i, x, y, vx):
    return TrackedObject(i, Header(0.0), ObjectClass.CAR, Pose2D(x, y, 0), np.array([vx, 0.0]))


def test_lane_object_filtering():
    ga = GapAssessor()
    objs = [_obj(1, 30, 3.5, 20), _obj(2, -20, 3.7, 20), _obj(3, 10, 0.0, 20), _obj(4, 5, -3.5, 20)]
    lo = ga.lane_objects(objs, target_offset=3.6)
    assert [o.track_id for o in lo] == [2, 1]


def test_open_and_bounded_gaps():
    ga = GapAssessor()
    gaps = ga.assess(0.0, 20.0, [])
    assert len(gaps) == 1 and gaps[0].acceptable and gaps[0].contains_ego
    lo = [LaneObject(1, -40.0, 3.6, 20.0), LaneObject(2, 40.0, 3.6, 20.0)]
    gaps = ga.assess(0.0, 20.0, lo)
    assert len(gaps) == 3
    cur = [g for g in gaps if g.contains_ego][0]
    assert cur.acceptable, cur.reasons
    assert cur.front_headway == pytest.approx((40 - 2.25 - 2.35) / 20)
    best = ga.best_current_gap(0.0, 20.0, lo)
    assert best.rear.track_id == 1 and best.front.track_id == 2


def test_unacceptable_gap_reasons():
    ga = GapAssessor()
    # fast rear vehicle closing in, slow front vehicle close ahead
    lo = [LaneObject(1, -15.0, 3.6, 30.0), LaneObject(2, 12.0, 3.6, 10.0)]
    cur = [g for g in ga.assess(0.0, 20.0, lo) if g.contains_ego][0]
    assert not cur.acceptable
    assert {"rear ttc", "front ttc", "front headway", "rear headway"} <= set(cur.reasons)
    assert ga.best_current_gap(0.0, 20.0, lo) is None


def test_merge_into_gap_ahead():
    ga = GapAssessor()
    mp = MergePlanner()
    # ego alongside a vehicle; a big gap opens ahead of it
    lo = [LaneObject(1, 2.0, 3.6, 20.0), LaneObject(2, 60.0, 3.6, 20.0)]
    gaps = ga.assess(0.0, 18.0, lo)
    assert not any(g.acceptable for g in gaps if g.contains_ego)
    plan = mp.plan(0.0, 18.0, 0.0, gaps)
    assert plan is not None
    assert not plan.gap.contains_ego
    if plan.gap.front and plan.gap.rear:
        assert plan.v[-1] == pytest.approx(plan.gap.speed)
    assert plan.a.max() <= mp.config.a_max + 1e-9 and plan.a.min() >= mp.config.a_min - 1e-9
    assert np.all(np.diff(plan.t) > 0)
    # ego ends inside the chosen gap (propagated)
    g, T = plan.gap, plan.T
    assert g.s_start + g.v_start * T < plan.s[-1] < g.s_end + g.v_end * T


def test_merge_infeasible_when_gaps_too_small():
    lo = [LaneObject(i, -50.0 + 6.0 * i, 3.6, 20.0) for i in range(20)]
    ga = GapAssessor()
    gaps = [g for g in ga.assess(0.0, 20.0, lo) if g.rear and g.front]
    assert MergePlanner().plan(0.0, 20.0, 0.0, gaps) is None
