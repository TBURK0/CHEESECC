import math

import numpy as np
import pytest

from cheesestack.core import Pose2D
from cheesestack.mapping import RoadGraph, Route


def test_route_successor_chain(sample_map):
    r = Route.from_lane_ids(sample_map, ["r1_0", "r2_0", "r3_0"])
    expected = sum(sample_map[l].length for l in ["r1_0", "r2_0", "r3_0"])
    assert r.length == pytest.approx(expected)
    s, d = r.to_frenet(40.0, 1.0)
    assert s == pytest.approx(40.0) and d == pytest.approx(1.0)
    np.testing.assert_allclose(r.from_frenet(40.0, 1.0), [40, 1], atol=1e-9)
    assert r.lane_at(50.0) == ("r1_0", pytest.approx(50.0))
    assert r.lane_at(110.0)[0] == "r2_0"
    assert r.speed_limit_at(110.0) == pytest.approx(8.33)
    assert r.remaining_distance(Pose2D(40, 0, 0)) == pytest.approx(expected - 40)
    assert r.remaining_distance(1e6) == 0.0
    assert r.distance_to_next_lane_change(0.0) is None


def test_frenet_roundtrip_on_curve(sample_map):
    r = Route.from_lane_ids(sample_map, ["r1_0", "r2_0", "r3_0"])
    rng = np.random.default_rng(0)
    for s in rng.uniform(5, r.length - 5, 20):
        d = rng.uniform(-1.5, 1.5)
        xy = r.from_frenet(s, d)
        s2, d2 = r.to_frenet(*xy, s_hint=s)
        assert s2 == pytest.approx(s, abs=0.3)  # polyline kinks on the arc
        assert d2 == pytest.approx(d, abs=0.05)


def test_frenet_extrapolation(sample_map):
    r = Route.from_lane_ids(sample_map, ["r1_0"])
    s, d = r.to_frenet(-5.0, 0.5)
    assert s == pytest.approx(-5.0) and d == pytest.approx(0.5)
    np.testing.assert_allclose(r.from_frenet(105.0, -1.0), [105.0, -1.0])
    s, d, he = r.pose_to_frenet(Pose2D(20, 0, 0.1))
    assert he == pytest.approx(0.1)


def test_route_with_lane_change_from_graph(sample_map):
    path = RoadGraph(sample_map).shortest_path("r1_0", "r3_1")
    r = Route.from_lane_ids(sample_map, path.lanes)
    assert r.lane_ids == ["r1_0", "r1_1", "r2_1", "r3_1"]
    assert r.segments[0].s_end == pytest.approx(50.0)
    assert r.segments[1].is_lane_change_from_prev and r.segments[1].s_start == pytest.approx(50.0)
    assert r.distance_to_next_lane_change(10.0) == pytest.approx(40.0 + 3.5)
    assert r.lane_at(70.0)[0] == "r1_1"
    with pytest.raises(ValueError):
        Route.from_lane_ids(sample_map, ["r1_0", "r3_0"])


def test_route_partial_ends(sample_map):
    r = Route.from_lane_ids(sample_map, ["r1_0", "r2_0"], start_s=20.0, end_s=10.0)
    assert r.length == pytest.approx(80.0 + 10.0, abs=1e-6)
    assert r.sample(5.0).shape[1] == 4
