import math

import numpy as np
import pytest

from cheesestack.core import Header, ObjectClass, Pose2D, TrackedObject
from cheesestack.planning.common import (ObstacleBox, QuarticPolynomial, QuinticPolynomial,
                                         ReferencePath, boxes_overlap, object_frenet)


def test_quintic_boundary_conditions():
    q = QuinticPolynomial(1.0, 2.0, 0.5, 10.0, 0.0, 0.0, 4.0)
    assert q.x(0) == pytest.approx(1.0)
    assert q.v(0) == pytest.approx(2.0)
    assert q.acc(0) == pytest.approx(0.5)
    assert q.x(4.0) == pytest.approx(10.0)
    assert q.v(4.0) == pytest.approx(0.0, abs=1e-9)
    assert q.acc(4.0) == pytest.approx(0.0, abs=1e-9)


def test_quartic_boundary_conditions():
    q = QuarticPolynomial(0.0, 10.0, 0.0, 15.0, 0.0, 5.0)
    assert q.v(5.0) == pytest.approx(15.0)
    assert q.acc(5.0) == pytest.approx(0.0, abs=1e-9)
    assert q.x(5.0) > 50.0


def test_reference_path_frenet_roundtrip_on_arc():
    th = np.linspace(0, math.pi / 2, 200)
    ref = ReferencePath(np.stack([50 * np.sin(th), 50 * (1 - np.cos(th))], 1))
    assert ref.length == pytest.approx(50 * math.pi / 2, rel=1e-3)
    assert np.median(ref.kappa) == pytest.approx(1 / 50, rel=0.02)
    x, y = ref.to_cartesian(30.0, 1.5)
    s, d = ref.to_frenet(float(x[0]), float(y[0]))
    assert s == pytest.approx(30.0, abs=0.05)
    assert d == pytest.approx(1.5, abs=0.02)
    # left of a left-turning arc is toward the centre (0, 50)
    s, d = ref.to_frenet(0.0, 49.0)
    assert d > 0


def test_straight_path_and_extrapolation():
    ref = ReferencePath.straight(10.0)
    s, d = ref.to_frenet(12.0, -1.0)
    assert s == pytest.approx(12.0)
    assert d == pytest.approx(-1.0)
    with pytest.raises(ValueError):
        ReferencePath([[0, 0], [0, 0]])


def test_boxes_overlap_and_motion():
    a = ObstacleBox(0, 0, 0, 4, 2)
    b = ObstacleBox(3.5, 0, 0.5, 4, 2)
    c = ObstacleBox(10, 0, 0, 4, 2, vx=-2.0)
    assert boxes_overlap(a.corners(), b.corners())
    assert not boxes_overlap(a.corners(), c.corners())
    assert boxes_overlap(a.corners(), c.at(3.5).corners())


def test_object_frenet_and_from_tracked():
    o = TrackedObject(7, Header(0.0), ObjectClass.CAR, Pose2D(20, 3.5, 0), np.array([12.0, 0.0]))
    assert object_frenet(o) == (20, 3.5, 12.0)
    box = ObstacleBox.from_tracked(o)
    assert box.track_id == 7 and box.vx == 12.0
