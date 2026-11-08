import math

import numpy as np
import pytest

from cheesestack.mapping import BoundaryType, Lane, LaneMap, Polyline


def test_polyline_arclength_and_interpolate():
    pl = Polyline([[0, 0], [3, 4], [3, 4], [3, 10]])  # duplicate dropped
    assert len(pl) == 3
    assert pl.length == pytest.approx(11.0)
    np.testing.assert_allclose(pl.interpolate(2.5), [1.5, 2.0])
    np.testing.assert_allclose(pl.interpolate(100), [3, 10])
    assert pl.heading_at(8.0) == pytest.approx(math.pi / 2)
    assert pl.interpolate(np.array([0.0, 5.0])).shape == (2, 2)


def test_polyline_project_signed_and_extrapolate():
    pl = Polyline([[0, 0], [10, 0]])
    pr = pl.project([4, 2])
    assert pr.s == pytest.approx(4) and pr.d == pytest.approx(2) and pr.distance == pytest.approx(2)
    assert pl.project([4, -1]).d == pytest.approx(-1)
    pr = pl.project([-3, 1], extrapolate=True)
    assert pr.s == pytest.approx(-3) and pr.d == pytest.approx(1)
    pr = pl.project([-3, 1])
    assert pr.s == 0.0 and pr.distance == pytest.approx(math.hypot(3, 1))


def test_polyline_window_resample_sub():
    # U-shape: point near both legs; window picks the later leg
    pl = Polyline([[0, 0], [10, 0], [10, 2], [0, 2]])
    assert pl.project([5, 0.9]).s == pytest.approx(5)
    assert pl.project([5, 0.9], s_min=12).s == pytest.approx(17)
    rs = Polyline([[0, 0], [10, 0]]).resample(1.0)
    assert len(rs) == 11
    sub = pl.sub(5, 13)
    assert sub.length == pytest.approx(8)
    np.testing.assert_allclose(sub.start, [5, 0])
    assert pl.reversed().length == pytest.approx(pl.length)
    with pytest.raises(ValueError):
        Polyline([[1, 1], [1, 1]])


def test_lane_map_topology(sample_map):
    m = sample_map
    assert m.validate() == []
    assert {l.id for l in m.successors("r1_0")} == {"r2_0", "r4_0"}
    assert [l.id for l in m.predecessors("r2_0")] == ["r1_0"]
    assert m.left_of("r1_0").id == "r1_1" and m.right_of("r1_1").id == "r1_0"
    assert m.leftmost("r1_0").id == "r1_1" and m.rightmost("r1_1").id == "r1_0"
    assert m["r1_0"].can_change_left() and not m["r1_1"].can_change_left()
    near = m.nearest_lanes([50, 0.3], radius=3.0)
    assert near[0][0].id == "r1_0"
    xmin, ymin, xmax, ymax = m.bounds()
    assert xmin == pytest.approx(0) and ymax == pytest.approx(90)
    assert BoundaryType.DASHED.allows_crossing and not BoundaryType.SOLID.allows_crossing


def test_validate_detects_problems():
    a = Lane("a", Polyline([[0, 0], [10, 0]]), successors=["b", "zz"], left_neighbor="b")
    b = Lane("b", Polyline([[20, 0], [30, 0]]))
    m = LaneMap([a, b])
    errs = " ".join(m.validate())
    assert "unknown successor zz" in errs
    assert "endpoint gap" in errs
    assert "not reciprocal" in errs
    m.remove_lane("b")
    assert m["a"].left_neighbor is None and "b" not in m["a"].successors
    with pytest.raises(KeyError):
        m.add_lane(Lane("a", Polyline([[0, 0], [1, 0]])))
