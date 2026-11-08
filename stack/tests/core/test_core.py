import math

import numpy as np

from cheesestack.core import MessageBus, Pose2D, SimClock, wrap_angle, BoundingBox2D


def test_pose_compose_inverse_roundtrip():
    a = Pose2D(1.0, 2.0, 0.7)
    b = Pose2D(-0.5, 3.0, -2.0)
    c = a.compose(b)
    back = a.inverse().compose(c)
    assert np.allclose(back.as_array(), b.as_array())


def test_wrap_angle():
    assert math.isclose(wrap_angle(3 * math.pi), -math.pi)
    assert math.isclose(wrap_angle(0.5), 0.5)


def test_bus_pubsub_and_wildcard():
    bus = MessageBus()
    got, everything = [], []
    bus.subscribe("/a", lambda t, m: got.append(m))
    bus.subscribe("*", lambda t, m: everything.append((t, m)))
    bus.publish("/a", 1)
    bus.publish("/b", 2)
    assert got == [1]
    assert everything == [("/a", 1), ("/b", 2)]
    assert bus.latest("/b") == 2


def test_sim_clock_monotonic():
    c = SimClock()
    c.advance(0.1)
    assert c.now() == 0.1


def test_iou():
    a = BoundingBox2D(0, 0, 10, 10)
    assert a.iou(a) == 1.0
    assert a.iou(BoundingBox2D(20, 20, 5, 5)) == 0.0
