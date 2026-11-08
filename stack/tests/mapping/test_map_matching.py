import math

import numpy as np
import pytest

from cheesestack.core import Pose2D
from cheesestack.mapping import HMMMapMatcher, MapMatcher, MapMatcherConfig, Route


def test_single_pose_match(sample_map):
    mm = MapMatcher(sample_map)
    m = mm.match(Pose2D(30.0, 0.4, 0.05))
    assert m.lane_id == "r1_0" and m.s == pytest.approx(30.0) and m.d == pytest.approx(0.4)
    assert 0.5 < m.confidence <= 1.0
    # between the two lanes, slightly closer to the left one
    c = mm.candidates(Pose2D(30.0, 2.0, 0.0))
    assert [x.lane_id for x in c[:2]] == ["r1_1", "r1_0"]
    assert sum(x.confidence for x in c) <= 1.0 + 1e-9


def test_heading_gating_rejects_opposing_lane(sample_map):
    mm = MapMatcher(sample_map)
    # exactly on the westbound lane but driving east -> must not match r1_w
    m = mm.match(Pose2D(50.0, 7.0, 0.0))
    assert m is None or m.lane_id != "r1_w"
    assert mm.match(Pose2D(50.0, 7.0, math.pi)).lane_id == "r1_w"
    assert mm.match(Pose2D(500.0, 500.0, 0.0)) is None


def test_far_candidate_has_low_confidence(sample_map):
    mm = MapMatcher(sample_map, MapMatcherConfig(max_distance_m=5.0))
    near = mm.match(Pose2D(50.0, 0.1, 0.0))
    far = mm.match(Pose2D(50.0, -4.5, 0.0))
    assert far.confidence < near.confidence


def test_hmm_fork_disambiguation(sample_map):
    """Near the fork end of r1_0 the left-turn and right-turn lanes both start at the
    same point; the sequence must resolve onto the branch actually driven (right turn)."""
    route = Route.from_lane_ids(sample_map, ["r1_0", "r4_0", "r5_0"])
    rng = np.random.default_rng(3)
    poses = []
    for s in np.arange(80.0, 140.0, 2.0):
        xy = route.from_frenet(s, rng.normal(0, 0.3))
        poses.append(Pose2D(xy[0], xy[1], route.heading_at(s) + rng.normal(0, 0.05)))
    hmm = HMMMapMatcher(sample_map)
    res = hmm.match_sequence(poses)
    ids = [r.lane_id for r in res]
    assert ids[0] == "r1_0" and ids[-1] == "r5_0"
    assert "r2_0" not in ids and "r2_1" not in ids and "r1_1" not in ids
    # lane ids along the sequence never go backwards in the route order
    order = {"r1_0": 0, "r4_0": 1, "r5_0": 2}
    assert all(order[a] <= order[b] for a, b in zip(ids[:-1], ids[1:]))


def test_hmm_smooths_ambiguous_pose_and_handles_gaps(sample_map):
    hmm = HMMMapMatcher(sample_map)
    poses = [Pose2D(x, 0.2, 0.0) for x in (10, 12, 14)]
    poses.append(Pose2D(16, 1.9, 0.0))  # closer to r1_1 alone, but a jump is unlikely
    poses += [Pose2D(x, 0.2, 0.0) for x in (18, 20)]
    poses.append(Pose2D(900, 900, 0.0))  # off-map -> None, chain restarts
    poses.append(Pose2D(40, 3.5, 0.0))
    res = hmm.match_sequence(poses)
    assert [r.lane_id for r in res[:6]] == ["r1_0"] * 6
    assert res[6] is None
    assert res[7].lane_id == "r1_1"


def test_route_distance_rules(sample_map):
    hmm = HMMMapMatcher(sample_map)
    mm = hmm.single
    a = mm.match(Pose2D(90, 0, 0))
    b = mm.match(Pose2D(120, -20 + 0.01, -math.pi / 2))
    dist, lc = hmm.route_distance(a, b)
    assert lc == 0 and dist == pytest.approx(10 + sample_map["r4_0"].length, abs=0.5)
    back = mm.match(Pose2D(80, 0, 0))
    assert hmm.route_distance(a, back) is None
    lc_t = mm.match(Pose2D(95, 3.5, 0))
    d2, lc2 = hmm.route_distance(a, lc_t)
    assert lc2 == 1 and d2 == pytest.approx(math.hypot(5, 3.5))
