import pytest

from cheesestack.mapping import EdgeType, RoadGraph, RoadGraphConfig


def test_graph_edges(sample_map):
    g = RoadGraph(sample_map)
    types = {(e.dst, e.type) for e in g.out_edges("r1_0")}
    assert ("r2_0", EdgeType.SUCCESSOR) in types
    assert ("r1_1", EdgeType.LANE_CHANGE_LEFT) in types
    # r1_1 left boundary is double solid -> no left change; right is dashed
    assert all(e.type != EdgeType.LANE_CHANGE_LEFT for e in g.out_edges("r1_1"))
    # r2 boundaries between lanes are solid -> no lane change in the curve
    assert all(e.type == EdgeType.SUCCESSOR for e in g.out_edges("r2_0"))
    assert g.reachable("r1_0") >= {"r3_0", "r3_1", "r5_0"}
    assert "r1_0" not in g.reachable("r3_0")


@pytest.mark.parametrize("algo", ["astar", "dijkstra"])
def test_shortest_path(sample_map, algo):
    g = RoadGraph(sample_map)
    p = g.shortest_path("r1_0", "r3_1", algorithm=algo)
    assert p.lanes == ["r1_0", "r1_1", "r2_1", "r3_1"]
    assert p.num_lane_changes == 1
    assert p.lanes[0] == "r1_0" and p.lanes[-1] == "r3_1"
    assert g.shortest_path("r1_0", "r5_0", algorithm=algo).lanes == ["r1_0", "r4_0", "r5_0"]
    assert g.shortest_path("r3_0", "r1_0", algorithm=algo) is None
    assert g.shortest_path("r1_0", "r1_0", algorithm=algo).lanes == ["r1_0"]


def test_astar_matches_dijkstra_and_expands_less(sample_map):
    g = RoadGraph(sample_map)
    a = g.shortest_path("r1_0", "r5_0", algorithm="astar")
    d = g.shortest_path("r1_0", "r5_0", algorithm="dijkstra")
    assert a.cost == pytest.approx(d.cost)
    assert a.expanded <= d.expanded


def test_blocked_and_time_mode(sample_map):
    g = RoadGraph(sample_map)
    assert g.shortest_path("r1_0", "r3_1", blocked={"r2_1"}) is None
    assert g.shortest_path("r1_0", "r3_0", blocked={"r3_0"}) is None
    gt = RoadGraph(sample_map, RoadGraphConfig(cost_mode="time"))
    p = gt.shortest_path("r1_0", "r3_0")
    expected = sample_map["r1_0"].length / 13.89 + sample_map["r2_0"].length / 8.33
    assert p.cost == pytest.approx(expected)
    slow = gt.shortest_path("r1_0", "r3_0", speed_limits={"r2_0": 2.0})
    assert slow.cost > p.cost
    no_lc = RoadGraph(sample_map, RoadGraphConfig(allow_lane_changes=False))
    assert no_lc.shortest_path("r1_0", "r3_1") is None
    with pytest.raises(KeyError):
        g.shortest_path("nope", "r1_0")
