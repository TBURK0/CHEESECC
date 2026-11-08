import math

import numpy as np
import pytest

from cheesestack.vehicle import DynamicBicycle, SlipEstimator, VehicleParams, YawRateEstimator, slip_ratio, wheel_ground_speeds

P = VehicleParams()


def test_yaw_rate_estimates_gyro_bias():
    rng = np.random.default_rng(0)
    est = YawRateEstimator(P)
    bias = 0.03
    out = None
    for i in range(2000):
        t = i * 0.01
        delta = 0.01 * math.sin(0.3 * t)
        v = 15.0
        r_true = P.steady_state_yaw_rate(v, delta)
        out = est.update(t, v, delta, gyro_z=r_true + bias + rng.normal(0, 0.005))
    assert out.gyro_bias == pytest.approx(bias, abs=0.005)
    assert out.yaw_rate == pytest.approx(r_true, abs=0.01)
    assert out.model_valid


def test_yaw_rate_rejects_model_when_sliding():
    est = YawRateEstimator(P)
    for i in range(100):
        est.update(i * 0.01, 10.0, 0.0, gyro_z=0.0)
    # car starts yawing (2 rad/s^2 ramp) with straight wheels: model must be gated out
    out = None
    for i in range(1, 41):
        out = est.update(1.0 + i * 0.01, 10.0, 0.0, gyro_z=min(0.5, 0.02 * i))
    assert not out.model_valid
    assert out.yaw_rate == pytest.approx(0.5, abs=0.03)
    assert abs(out.gyro_bias) < 0.01  # sliding must not be absorbed into the bias


def test_slip_ratio_basic():
    assert slip_ratio(11.0, 10.0) == pytest.approx(1 / 11)
    assert slip_ratio(0.0, 10.0) == pytest.approx(-1.0)  # locked
    assert slip_ratio(0.0, 0.0) == 0.0


def test_wheel_ground_speeds_turning():
    g = wheel_ground_speeds(P, 10.0, 0.0, 0.5, 0.0)
    assert g["rr"] > g["rl"]  # left turn: right (outer) wheel faster
    assert g["rr"] - g["rl"] == pytest.approx(0.5 * P.track_rear)


def test_sideslip_estimator_tracks_dynamic_model():
    rng = np.random.default_rng(1)
    model = DynamicBicycle(P)
    est = SlipEstimator(P)
    v, delta, dt = 20.0, 0.03, 0.01
    x = np.array([0, 0, 0, v, 0, 0], dtype=float)
    _, _, C, D = model.lateral_state_space(v)
    out = None
    for i in range(400):
        x = model.step(x, delta, 0.0, dt)
        x[3] = v
        ay = float(C @ x[4:] + D * delta)
        omega = [wheel_ground_speeds(P, v, x[4], x[5], delta)[w] / P.tire_radius for w in ("fl", "fr", "rl", "rr")]
        omega[0] *= 1.3  # front-left spinning
        out = est.update((i + 1) * dt, v, delta, omega, gyro_z=x[5] + rng.normal(0, 0.002),
                         lat_accel=ay + rng.normal(0, 0.05))
    beta_true = math.atan2(x[4], v)
    assert out.sideslip == pytest.approx(beta_true, abs=0.003)
    assert out.slipping_wheels == ("fl",)
    assert out.slip_ratio["rr"] == pytest.approx(0.0, abs=0.02)


def test_sideslip_low_speed_kinematic():
    est = SlipEstimator(P)
    out = est.update(0.0, 1.0, 0.3)
    assert out.sideslip == pytest.approx(math.atan(P.lr * math.tan(0.3) / P.wheelbase))
