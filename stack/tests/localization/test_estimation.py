import math

import numpy as np
import pytest

from cheesestack.core import Header, MessageBus, Pose2D, SimClock, VehicleState
from cheesestack.localization import (
    ClockOffsetEstimator, CurvatureEstimator, FrameLookupError, FrameTree, GpsMeasurement, HeadingFilter,
    ImuSample, InterpolationBuffer, PoseTracker, StateEstimator, VelocityEstimator, WheelOdomEstimate,
    course_from_positions, curvature_from_path,
)


def test_state_estimator_fuses_circle():
    rng = np.random.default_rng(0)
    bus, clock = MessageBus(), SimClock()
    est = StateEstimator(bus=bus, clock=clock)
    v, w, R = 8.0, 0.1, 80.0
    dt = 0.02
    for k in range(1, 1001):
        t = k * dt
        clock.set(t)
        yaw = w * t
        pos = np.array([R * math.sin(yaw), R * (1 - math.cos(yaw))])
        bus.publish("/sensors/imu", ImuSample(t, np.array([0, v * w, 9.81]), np.array([0, 0, w + rng.normal(0, 0.005)])))
        if k % 5 == 0:
            bus.publish("/localization/wheel_odom", WheelOdomEstimate(t, v + rng.normal(0, 0.05), w, v_var=0.01, yaw_rate_source="bicycle"))
        if k % 50 == 0:
            course = yaw + rng.normal(0, 0.03)
            bus.publish("/localization/gps", GpsMeasurement(Header(t, "map"), pos + rng.normal(0, 1.0, 2), np.eye(2), True,
                                                           speed_mps=v, course_yaw=course))
        est.tick()
    st = bus.latest("/localization/state")
    assert isinstance(st, VehicleState)
    yaw = w * 20.0
    truth = np.array([R * math.sin(yaw), R * (1 - math.cos(yaw))])
    assert np.hypot(st.pose.x - truth[0], st.pose.y - truth[1]) < 1.5
    assert abs(math.remainder(st.pose.yaw - yaw, 2 * math.pi)) < 0.05
    assert st.v == pytest.approx(v, abs=0.1) and st.yaw_rate == pytest.approx(w, abs=0.01)
    assert st.curvature == pytest.approx(1 / R, abs=2e-3)
    assert st.covariance.shape == (5, 5)


def test_state_estimator_rejects_gps_outlier():
    est = StateEstimator()
    m = lambda t, xy: GpsMeasurement(Header(t, "map"), np.array(xy, float), np.eye(2) * 0.25, True)
    assert est.update_gps(m(0.0, [0, 0]))
    assert est.update_gps(m(0.1, [0.1, 0]))
    assert not est.update_gps(m(0.2, [200, 0]))
    assert est.rejected["gps"] == 1


def test_heading_filter_converges_and_learns_bias():
    hf = HeadingFilter()
    true_yaw, bias = 1.0, 0.01
    hf.correct(0.0, true_yaw + 0.3, 10.0)  # initializes (with error)
    for k in range(1, 3000):
        t = k * 0.01
        hf.predict(t, 0.0 + bias)
        if k % 10 == 0:
            hf.correct(t, true_yaw, 10.0)
    assert hf.yaw == pytest.approx(true_yaw, abs=0.01)
    assert hf.bias == pytest.approx(bias, abs=0.004)
    assert not hf.correct(31.0, 0.0, 0.5)  # too slow
    assert course_from_positions([0, 0], [0, 1]) == pytest.approx(math.pi / 2)
    assert course_from_positions([0, 0], [0, 0.1]) is None


def test_velocity_estimator():
    ve = VelocityEstimator()
    for k in range(100):
        ve.update_position(k * 0.1, [3.0 * k * 0.1, 0.0])
    assert ve.speed(0.0) == pytest.approx(3.0, abs=0.05)
    ve.update_speed(10.0, 4.0, 0.0)
    assert 3.0 < ve.speed(0.0) <= 4.0


def test_curvature_low_speed_and_path():
    ce = CurvatureEstimator()
    assert ce.instantaneous(10.0, 0.5) == pytest.approx(0.05)
    assert abs(ce.instantaneous(0.01, 0.05)) < 0.001  # no blow-up at standstill
    assert ce.instantaneous(0.0, 0.0, steer=0.1) == pytest.approx(math.tan(0.1) / 2.7)
    th = np.linspace(0, math.pi, 50)
    left = np.column_stack([10 * np.sin(th), 10 * (1 - np.cos(th))])
    assert np.allclose(curvature_from_path(left), 0.1, atol=1e-6)
    assert np.allclose(curvature_from_path(left[::-1]), -0.1, atol=1e-6)
    assert np.allclose(curvature_from_path([[0, 0], [1, 0], [2, 0]]), 0)


def test_clock_offset_and_drift():
    rng = np.random.default_rng(4)
    est = ClockOffsetEstimator()
    for k in range(200):
        s = k * 0.5
        est.add(s, s * (1 + 1e-3) + 2.5 + 0.01 + rng.exponential(0.005))  # 10 ms fixed + jitter latency
    assert est.drift == pytest.approx(1e-3, abs=1e-4)
    assert est.to_host(50.0) == pytest.approx(50.0 * (1 + 1e-3) + 2.51, abs=0.003)


def test_interpolation_buffer():
    buf = InterpolationBuffer(max_age_s=1.0, angle_indices=(1,))
    buf.add(0.0, np.array([0.0, 3.0]))
    buf.add(1.0, np.array([10.0, -3.0]))
    v = buf.interpolate(0.5)
    assert v[0] == pytest.approx(5.0) and abs(abs(v[1]) - math.pi) < 0.2  # through pi, not 0
    assert buf.interpolate(2.0) is None
    buf.add(1.5, np.array([15.0, 0.0]))
    assert len(buf) == 2  # t=0 evicted
    pb = InterpolationBuffer()
    pb.add(0.0, Pose2D(0, 0, 3.0))
    pb.add(1.0, Pose2D(2, 0, -3.0))
    p = pb.interpolate(0.5)
    assert p.x == pytest.approx(1.0) and abs(abs(p.yaw) - math.pi) < 1e-9


def test_frame_tree_chain():
    tf = FrameTree()
    tf.set_static("base_link", "camera_front", Pose2D(1.5, 0.0, 0.0))
    tf.set_static("base_link", "gps", Pose2D(0.5, 0.2, 0.0))
    tf.set_static("map", "odom", Pose2D(100, 50, math.pi / 2))
    tf.set_transform("odom", "base_link", Pose2D(0, 0, 0), 0.0)
    tf.set_transform("odom", "base_link", Pose2D(10, 0, 0), 1.0)
    T = tf.lookup("map", "camera_front", 0.5)
    assert np.allclose(T.as_array(), [100, 50 + 5 + 1.5, math.pi / 2])
    T = tf.lookup("camera_front", "gps")
    assert np.allclose(T.as_array(), [-1.0, 0.2, 0.0])
    p = tf.transform_points("map", "camera_front", [[1, 0]], 1.0)
    assert np.allclose(p, [[100, 50 + 10 + 2.5]])
    assert np.allclose(tf.lookup("gps", "map", 1.0).compose(tf.lookup("map", "gps", 1.0)).as_array(), 0, atol=1e-9)
    with pytest.raises(FrameLookupError):
        tf.lookup("map", "base_link", 5.0)
    with pytest.raises(ValueError):
        tf.set_static("camera_front", "map", Pose2D())


def test_pose_tracker_delayed_correction():
    pt = PoseTracker()
    pt.reset(0.0, Pose2D(0, 0, 0))
    for k in range(1, 11):
        pt.add_velocity(k * 0.1, 10.0, 0.0)
    assert pt.map_pose.x == pytest.approx(10.0)
    # Fix says at t=0.5 we were actually at x=6 (1 m ahead of dead reckoning) -> now at 11.
    assert pt.correct(0.5, Pose2D(6.0, 0.0, 0.0))
    assert pt.map_pose.x == pytest.approx(11.0)
    assert pt.odom_pose.x == pytest.approx(10.0)  # odom continuous
    assert not pt.correct(1.0, Pose2D(1000, 0, 0))
    pt.add_velocity(1.1, 0.0, 0.0)
    pt.add_velocity(1.2, math.pi / 2, math.pi / 2 / 0.1 * 0.1 / 0.1 / 10)  # quarter-arc check below
