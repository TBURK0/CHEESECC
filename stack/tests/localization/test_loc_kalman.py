import math

import numpy as np
import pytest

from cheesestack.localization import kalman as K


def test_linear_kf_constant_velocity():
    rng = np.random.default_rng(0)
    dt = 0.1
    F = np.array([[1, dt], [0, 1]])
    kf = K.KalmanFilter([0, 0], np.eye(2) * 10, F=F, H=np.array([[1.0, 0]]), Q=np.eye(2) * 1e-4, R=np.array([[0.25]]))
    nis = []
    for k in range(200):
        kf.predict()
        nis.append(kf.update([2.0 * k * dt + rng.normal(0, 0.5)]))
    assert kf.x[1] == pytest.approx(2.0, abs=0.1)
    assert 0.5 < np.mean(nis[20:]) < 1.6  # consistent (expected 1)


def test_ctrv_jacobian_matches_numeric():
    m = K.CTRVModel()
    for s in (np.array([1, 2, 0.3, 5, 0.2]), np.array([1, 2, -2.0, 3, 0.0])):
        Fa = m.jacobian(s, 0.1)
        Fn = K.numerical_jacobian(lambda q: m.f(q, 0.1), s, angle_out=(2,))
        assert np.allclose(Fa, Fn, atol=1e-5)


def test_ctrv_full_circle():
    m = K.CTRVModel()
    s = np.array([0, 0, 0, 2 * math.pi, 2 * math.pi])  # radius 1, one rev per second
    out = m.f(s, 1.0)
    assert np.allclose(out[:2], 0, atol=1e-9)


def _simulate(filter_kind):
    rng = np.random.default_rng(3)
    m = K.CTRVModel(sigma_accel=0.2, sigma_yaw_accel=0.05)
    truth = np.array([0, 0, 0.5, 8.0, 0.1])
    x0 = truth + np.array([2, -2, 0.2, 1, 0.05])
    P0 = np.diag([4, 4, 0.1, 1, 0.01])
    f = K.ExtendedKalmanFilter(x0, P0, (2,)) if filter_kind == "ekf" else K.UnscentedKalmanFilter(x0, P0, angle_indices=(2,))
    R = np.eye(2) * 0.5**2
    dt = 0.1
    for _ in range(150):
        truth = m.f(truth, dt)
        if filter_kind == "ekf":
            f.predict(lambda s: m.f(s, dt), m.Q(f.x, dt), m.jacobian(f.x, dt))
        else:
            f.predict(lambda s: m.f(s, dt), m.Q(f.x, dt))
        z = truth[:2] + rng.normal(0, 0.5, 2)
        f.update(z, lambda s: s[:2], R)
        f.update([truth[3] + rng.normal(0, 0.1)], lambda s: s[3:4], [[0.01]])
    return truth, f


@pytest.mark.parametrize("kind", ["ekf", "ukf"])
def test_ekf_ukf_track_ctrv(kind):
    truth, f = _simulate(kind)
    assert np.hypot(*(f.x[:2] - truth[:2])) < 0.6
    assert abs(math.remainder(f.x[2] - truth[2], 2 * math.pi)) < 0.05
    assert f.x[4] == pytest.approx(0.1, abs=0.02)


def test_ekf_gate_and_angle_residual():
    ekf = K.ExtendedKalmanFilter([0, 0, 3.1, 0, 0], np.eye(5) * 0.01, (2,))
    nis = ekf.update([-3.1], lambda s: s[2:3], [[0.01]], residual_angles=(0,))
    assert nis < 1.0  # wrapped residual is ~0.08 rad, not 6.2
    nis_bad = ekf.update([100.0], lambda s: s[0:1], [[0.01]], gate=10.0)
    assert nis_bad > 10 and abs(ekf.x[0]) < 1e-9


def test_bicycle_model():
    m = K.KinematicBicycleModel(wheelbase_m=2.5)
    s = np.array([0, 0, 0, 5.0])
    out = m.f(s, 0.1, (1.0, 0.0))
    assert out[0] == pytest.approx(0.5) and out[3] == pytest.approx(5.1)
    out = m.f(s, 0.1, (0.0, 0.2))
    assert out[2] == pytest.approx(5 * math.tan(0.2) / 2.5 * 0.1)
    Q = m.Q(s, 0.1, (0.0, 0.1))
    assert Q.shape == (4, 4) and np.all(np.linalg.eigvalsh(Q) >= 0)
