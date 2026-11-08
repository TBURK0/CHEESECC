import math

import numpy as np
import pytest

from cheesestack.vehicle import DynamicBicycle, KinematicBicycle, LongitudinalModel, VehicleParams, rk4_step

P = VehicleParams()


def test_rk4_exponential():
    x = np.array([1.0])
    for _ in range(100):
        x = rk4_step(lambda x, u: -x, x, np.zeros(0), 0.01)
    assert x[0] == pytest.approx(math.exp(-1.0), rel=1e-8)


def test_kinematic_circle():
    m = KinematicBicycle(P)
    delta = 0.1
    R = P.wheelbase / math.tan(delta)
    x = np.array([0.0, 0.0, 0.0, 5.0])
    T = 2 * math.pi * R / 5.0
    n = 2000
    for _ in range(n):
        x = m.step(x, delta, 0.0, T / n)
    assert np.hypot(x[0], x[1]) < 1e-3  # back to origin after one lap
    assert m.yaw_rate(5.0, delta) == pytest.approx(5.0 / R)


def test_dynamic_bicycle_converges_to_steady_state():
    m = DynamicBicycle(P)
    v, delta = 20.0, 0.02
    x = np.array([0, 0, 0, v, 0, 0], dtype=float)
    for _ in range(300):
        x = m.step(x, delta, 0.0, 0.01)
        x[3] = v  # hold speed
    vy_ss, r_ss = m.steady_state(v, delta)
    assert x[5] == pytest.approx(r_ss, rel=1e-3)
    assert x[4] == pytest.approx(vy_ss, abs=1e-3)
    assert r_ss == pytest.approx(P.steady_state_yaw_rate(v, delta), rel=1e-9)
    assert r_ss < v * delta / P.wheelbase  # understeer reduces yaw gain


def test_dynamic_low_speed_matches_kinematic():
    m = DynamicBicycle(P)
    x = np.array([0, 0, 0, 1.0, 0, 0], dtype=float)
    for _ in range(100):
        x = m.step(x, 0.2, 0.0, 0.01)
    assert x[5] == pytest.approx(1.0 * math.cos(math.atan(P.lr * math.tan(0.2) / P.wheelbase)) * math.tan(0.2) / P.wheelbase, rel=1e-2)


def test_lateral_state_space_stable():
    A, _, _, _ = DynamicBicycle(P).lateral_state_space(25.0)
    assert np.all(np.linalg.eigvals(A).real < 0)


def test_longitudinal_coastdown_and_accel():
    lm = LongitudinalModel(P)
    assert lm.acceleration(20.0, 0.0, 0.0) < 0
    assert lm.coastdown_decel(20.0) == pytest.approx((0.5 * 1.225 * 0.3 * 2.2 * 400 + 0.013 * 1500 * 9.81) / 1500)
    assert lm.acceleration(5.0, 1.0, 0.0) > 1.0
    assert lm.acceleration(5.0, 0.0, 1.0) < -6.0
    # uphill reduces accel
    assert lm.acceleration(10.0, 0.5, 0, grade_rad=0.05) < lm.acceleration(10.0, 0.5, 0)
    v = 10.0
    for _ in range(1000):
        v = lm.step(v, 0.0, 1.0, 0.01)
    assert v == 0.0
