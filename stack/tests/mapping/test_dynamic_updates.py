import math

import pytest

from cheesestack.core import SimClock
from cheesestack.mapping import MapOverlay, MapUpdate, OverlayConfig, RoadGraph, RoadGraphConfig, UpdateKind


def test_closure_expiry_and_versioning(sample_map):
    clock = SimClock()
    ov = MapOverlay(sample_map)
    assert ov.version == 0
    uid = ov.close_lane("r2_1", clock.now(), duration_s=60.0, source="v2x")
    assert ov.version == 1 and ov.is_closed("r2_1", 0.0)
    g = RoadGraph(sample_map)
    p = g.shortest_path("r1_0", "r3_1", **ov.graph_constraints(clock.now()))
    assert "r2_1" not in p.lanes
    clock.advance(61.0)
    assert not ov.is_closed("r2_1", clock.now())
    assert ov.expire(clock.now()) == [uid]
    assert ov.version == 2 and len(ov) == 0
    assert [h.action for h in ov.history] == ["add", "expire"]
    assert "r2_1" in g.shortest_path("r1_0", "r3_1", **ov.graph_constraints(clock.now())).lanes


def test_partial_closure_and_future_start(sample_map):
    ov = MapOverlay(sample_map)
    ov.add(MapUpdate(UpdateKind.CLOSURE, "r1_0", created=10.0, expires_at=20.0, s_start=40.0, s_end=60.0))
    assert not ov.is_closed("r1_0", 5.0)  # not yet active
    assert ov.is_closed("r1_0", 15.0, s=50.0) and not ov.is_closed("r1_0", 15.0, s=70.0)
    lm = ov.materialize(15.0)
    assert "r1_0" in lm and lm["r1_0"].attributes["closures"] == [(40.0, 60.0)]
    assert "r1_0" in ov.blocked_lanes(15.0)
    assert "closures" not in sample_map["r1_0"].attributes  # base untouched
    assert ov.add(MapUpdate(UpdateKind.CLOSURE, "r1_0", created=5.0, expires_at=5.0)) == -1
    with pytest.raises(KeyError):
        ov.add(MapUpdate(UpdateKind.CLOSURE, "nope"))


def test_speed_limits_and_time_routing(sample_map):
    ov = MapOverlay(sample_map)
    ov.set_speed_limit("r1_0", 5.0, now=0.0, duration_s=100)
    ov.set_speed_limit("r1_0", 8.0, now=0.0, duration_s=100)
    assert ov.speed_limit("r1_0", 1.0) == 5.0  # most restrictive wins
    ov.set_speed_limit("r2_0", 50.0, now=0.0)
    assert ov.speed_limit("r2_0", 1.0) == pytest.approx(8.33)  # never raises the base limit
    gt = RoadGraph(sample_map, RoadGraphConfig(cost_mode="time"))
    base = gt.shortest_path("r1_0", "r3_0").cost
    slowed = gt.shortest_path("r1_0", "r3_0", **ov.graph_constraints(1.0)).cost
    assert slowed > base
    lm = ov.materialize(1.0)
    assert lm["r1_0"].speed_limit == 5.0


def test_observed_lane_shift_fusion(sample_map):
    ov = MapOverlay(sample_map, OverlayConfig(shift_gain=0.5, observation_ttl_s=10.0))
    uid = ov.observe_lane_shift("r3_0", 1.0, now=0.0)
    for t in range(1, 6):
        assert ov.observe_lane_shift("r3_0", 1.0, now=float(t)) == uid
    assert ov.lane_shift("r3_0", 5.0) == pytest.approx(1.0)
    ov.observe_lane_shift("r3_0", 0.0, now=6.0)
    assert ov.lane_shift("r3_0", 6.0) == pytest.approx(0.5)
    lm = ov.materialize(6.0)
    # northbound lane shifted left (+) => x decreases
    assert lm["r3_0"].centerline.points[0, 0] == pytest.approx(130 - 0.5)
    assert ov.lane_shift("r3_0", 17.0) == 0.0  # expired (ttl refreshed at t=6)
    assert ov.observe_lane_shift("r3_0", 1.0, now=7.0, confidence=0.01) == -1


def test_lane_change_block_materialize(sample_map):
    ov = MapOverlay(sample_map)
    ov.block_lane_change("r1_0", +1, now=0.0, duration_s=30.0)
    assert not ov.lane_change_allowed("r1_0", +1, 1.0) and ov.lane_change_allowed("r1_0", -1, 1.0)
    lm = ov.materialize(1.0)
    g = RoadGraph(lm)
    assert g.shortest_path("r1_0", "r3_1").lanes[1] != "r1_1"  # cannot change left on r1_0
    assert RoadGraph(ov.materialize(31.0)).shortest_path("r1_0", "r3_1").lanes[1] == "r1_1"


def test_materialize_drops_closed_lane_and_remove(sample_map):
    ov = MapOverlay(sample_map)
    uid = ov.close_lane("r4_0", now=0.0)
    lm = ov.materialize(1.0)
    assert "r4_0" not in lm and "r4_0" not in lm["r1_0"].successors
    assert lm.validate() == []
    assert ov.remove(uid, now=2.0) and not ov.remove(uid)
    assert "r4_0" in ov.materialize(3.0)
