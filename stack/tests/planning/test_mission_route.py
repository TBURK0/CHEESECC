import pytest

from cheesestack.core import Pose2D
from cheesestack.planning import (Goal, Mission, MissionState, RoutePlanner, WaypointGraph,
                                  lookahead_distance)


def test_mission_lifecycle():
    m = Mission([Goal("a", Pose2D(10, 0)), Goal("b", Pose2D(20, 0))])
    assert m.state == MissionState.IDLE
    m.start()
    st = m.update(Pose2D(0, 0), 5.0)
    assert st.state == MissionState.ACTIVE and st.goal_index == 0
    assert st.distance_to_goal == pytest.approx(10)
    st = m.update(Pose2D(9, 1), 5.0)  # intermediate goal, no speed requirement
    assert st.goal_index == 1
    st = m.update(Pose2D(20, 0), 3.0)  # final goal requires arrive speed
    assert st.state == MissionState.ACTIVE
    st = m.update(Pose2D(20, 0), 0.1)
    assert st.state == MissionState.ARRIVED and m.reached == ["a", "b"]
    with pytest.raises(RuntimeError):
        m.start()


def test_mission_abort_and_heading():
    m = Mission([Goal("a", Pose2D(0, 0, 0), yaw_tolerance=0.2)])
    m.start()
    assert m.update(Pose2D(0, 0, 1.0), 0).state == MissionState.ACTIVE
    m.abort("driver takeover")
    assert m.state == MissionState.ABORTED and m.reason == "driver takeover"
    empty = Mission([])
    empty.start()
    assert empty.state == MissionState.ABORTED


def _grid():
    g = WaypointGraph()
    for i in range(4):
        for j in range(4):
            g.add_node((i, j), i * 10.0, j * 10.0)
    for i in range(4):
        for j in range(4):
            if i < 3:
                g.add_edge((i, j), (i + 1, j))
            if j < 3:
                g.add_edge((i, j), (i, j + 1))
    return g


def test_astar_shortest_path_and_blocking():
    g = _grid()
    rp = RoutePlanner(g)
    r = rp.plan_nodes((0, 0), (3, 3))
    assert r.cost == pytest.approx(60.0)
    assert r.nodes[0] == (0, 0) and r.nodes[-1] == (3, 3)
    assert r.length == pytest.approx(60.0)
    # isolate the goal
    g.remove_edge((3, 3), (2, 3))
    g.remove_edge((3, 3), (3, 2))
    assert rp.plan_nodes((0, 0), (3, 3)) is None


def test_directed_edge_and_snap_plan():
    g = WaypointGraph()
    g.add_node("a", 0, 0)
    g.add_node("b", 10, 0)
    g.add_edge("a", "b", bidirectional=False)
    rp = RoutePlanner(g)
    assert rp.plan_nodes("b", "a") is None
    r = rp.plan((-1, 0), (11, 0))
    assert r.length == pytest.approx(12.0)
    p, yaw = r.point_at(6.0)
    assert p[0] == pytest.approx(5.0) and yaw == pytest.approx(0.0)
    along, off = r.closest((4.0, 2.0))
    assert along == pytest.approx(5.0) and off == pytest.approx(2.0)


def test_lookahead_matches_app_scheduler():
    assert lookahead_distance(0.0) == 8.0
    assert lookahead_distance(20.0) == 8.0  # 20 * 0.36 = 7.2 -> floor of 8 m
    assert lookahead_distance(40.0) == pytest.approx(40 * 0.25)
