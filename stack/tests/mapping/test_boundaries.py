import math

import numpy as np
import pytest

from cheesestack.mapping import (
    BoundaryType, Lane, LaneMap, Polyline, Route, build_corridor, corridor_from_route,
    generate_lane_boundaries, lane_boundary_polylines, offset_polyline, point_in_polygon, polygon_area,
)


def test_offset_straight_and_corner():
    off = offset_polyline([[0, 0], [10, 0]], 2.0)
    np.testing.assert_allclose(off, [[0, 2], [10, 2]])
    corner = offset_polyline([[0, 0], [10, 0], [10, 10]], 1.0)  # left turn, inner side
    np.testing.assert_allclose(corner[1], [9, 1], atol=1e-9)
    outer = offset_polyline([[0, 0], [10, 0], [10, 10]], -1.0)
    np.testing.assert_allclose(outer[1], [11, -1], atol=1e-9)
    hairpin = offset_polyline([[0, 0], [10, 0], [0, 0.01]], 1.0, miter_limit=3.0)
    assert np.linalg.norm(hairpin[1] - [10, 0]) <= 3.0 + 1e-9


def test_offset_arc_radius(sample_map):
    c = sample_map["r2_0"].centerline.points
    left = offset_polyline(c, 1.75)
    r = np.hypot(left[:, 0] - 100, left[:, 1] - 30)
    np.testing.assert_allclose(r, 30 - 1.75, atol=0.02)


def test_generate_boundaries_types():
    a = Lane("a", Polyline([[0, 0], [50, 0]]), left_neighbor="b")
    b = Lane("b", Polyline([[0, 3.5], [50, 3.5]]), right_neighbor="a")
    m = LaneMap([a, b])
    assert generate_lane_boundaries(m) == 4
    assert m["a"].left_boundary.type == BoundaryType.DASHED
    assert m["a"].right_boundary.type == BoundaryType.CURB
    assert m["b"].left_boundary.type == BoundaryType.SOLID
    np.testing.assert_allclose(m["a"].left_boundary.polyline.points, m["b"].right_boundary.polyline.points)
    assert generate_lane_boundaries(m) == 0
    left, right = lane_boundary_polylines(m["a"])
    assert left.points[0, 1] == pytest.approx(1.75) and right.points[0, 1] == pytest.approx(-1.75)


def test_polygon_helpers():
    sq = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    assert polygon_area(sq) == pytest.approx(4.0)
    assert polygon_area(sq[::-1]) == pytest.approx(-4.0)
    assert point_in_polygon([1, 1], sq) and not point_in_polygon([3, 1], sq)


def test_corridor(sample_map):
    c = build_corridor(sample_map, ["r1_0"])
    assert c.area == pytest.approx(100 * 3.5, rel=1e-3)
    assert polygon_area(c.polygon) > 0  # CCW
    assert c.contains([50, 1.0]) and not c.contains([50, 3.0])
    wide = build_corridor(sample_map, ["r1_0"], include_neighbors=True)
    assert wide.area == pytest.approx(100 * 7.0, rel=1e-3)
    assert wide.contains([50, 3.0])
    r = Route.from_lane_ids(sample_map, ["r1_0", "r2_0", "r3_0"])
    rc = corridor_from_route(r)
    assert rc.contains(r.from_frenet(120.0, 0.0)) and rc.contains([130, 80])
    assert not rc.contains(r.from_frenet(120.0, 2.5))
    lane_area = 3.5 * sum(sample_map[l].length for l in ["r1_0", "r2_0", "r3_0"])
    assert rc.area == pytest.approx(lane_area, rel=0.02)
