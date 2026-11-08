import math

import numpy as np
import pytest

from cheesestack.localization import imu as I
from cheesestack.localization import wheel_odometry as W


def test_tilt_and_gravity_removal():
    roll, pitch = 0.05, -0.03
    f = I.level_rotation(roll, pitch).T @ np.array([0, 0, I.GRAVITY])
    r, p = I.tilt_from_accel(f)
    assert r == pytest.approx(roll) and p == pytest.approx(pitch)
    assert np.allclose(I.remove_gravity(f, r, p), 0, atol=1e-9)


def test_bias_estimation_at_standstill():
    rng = np.random.default_rng(1)
    est = I.ImuBiasEstimator()
    bias = np.array([0.002, -0.001, 0.004])
    for k in range(300):
        s = I.ImuSample(k * 0.01, np.array([0, 0, I.GRAVITY]) + rng.normal(0, 0.01, 3), bias + rng.normal(0, 0.001, 3))
        est.update(s)
    assert est.ready
    assert np.allclose(est.gyro_bias, bias, atol=5e-4)
    # Moving: large accel variance -> not standstill.
    det = I.StandstillDetector()
    still = [det.update(I.ImuSample(k * 0.01, np.array([math.sin(k), 0, I.GRAVITY]), np.zeros(3))) for k in range(100)]
    assert not any(still)


def test_strapdown_circle():
    v, w, dt = 5.0, 0.2, 0.01
    sd = I.PlanarStrapdown(velocity=(v, 0.0))
    n = int(round(2 * math.pi / w / dt))
    for k in range(n + 1):
        sd.update(I.ImuSample(k * dt, np.array([0.0, v * w, I.GRAVITY]), np.array([0, 0, w])))
    # Full circle -> back at origin.
    assert math.hypot(sd.x, sd.y) < 0.5
    assert sd.speed == pytest.approx(v, rel=0.02)


def test_ticks_and_rpm():
    assert W.ticks_to_speed(100, 0.1, 100, 0.3) == pytest.approx(2 * math.pi * 0.3 / 0.1)
    assert W.tick_delta(5, 65530) == 11
    assert W.tick_delta(65530, 5) == -11
    assert W.rpm_to_speed(60, 0.5) == pytest.approx(math.pi)


def test_wheel_odom_turn_and_slip():
    cfg = W.WheelOdomConfig(lowpass_tau_s=0.0)
    wo = W.WheelOdometry(cfg)
    v, r = 10.0, 0.2
    h = cfg.track_rear_m / 2
    ws = W.WheelSpeeds(0.0, v - r * h, v + r * h, v - r * h, v + r * h)
    est = wo.update(ws)
    assert est.v == pytest.approx(v) and est.yaw_rate == pytest.approx(r)
    assert not any(est.slipping.values())
    # Rear-left spins up (slip) -> excluded, falls back to front pair for yaw rate.
    ws2 = W.WheelSpeeds(0.01, v - r * h, v + r * h, v + 3.0, v + r * h)
    est2 = wo.update(ws2)
    assert est2.slipping["rl"] and est2.quality == pytest.approx(0.75)
    assert est2.yaw_rate_source == "front"
    assert est2.v == pytest.approx(v, abs=0.2)
    assert est2.v_var > est.v_var


def test_wheel_odom_integrates_straight():
    wo = W.WheelOdometry(W.WheelOdomConfig(lowpass_tau_s=0.0))
    for k in range(101):
        wo.update(W.WheelSpeeds(k * 0.1, 2, 2, 2, 2))
    assert wo.pose.x == pytest.approx(20.0) and abs(wo.pose.y) < 1e-9
