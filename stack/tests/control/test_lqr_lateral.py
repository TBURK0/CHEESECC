import math

import numpy as np

from cheesestack.core import Pose2D
from cheesestack.control import LQRLateralController, ReferencePath, VehicleParams, c2d_zoh, dlqr, error_dynamics, expm, solve_dare
from _sim import straight_path


def test_expm_matches_rotation():
    th = 0.7
    E = expm(np.array([[0.0, -th], [th, 0.0]]))
    assert np.allclose(E, [[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], atol=1e-10)


def test_dare_scalar_closed_form():
    a, b, q, r = 1.2, 1.0, 1.0, 1.0
    P = solve_dare(np.array([[a]]), np.array([[b]]), np.array([[q]]), np.array([[r]]))[0, 0]
    # scalar DARE: P = q + a^2 P - a^2 b^2 P^2 / (r + b^2 P)
    assert abs(P - (q + a * a * P - (a * b * P) ** 2 / (r + b * b * P))) < 1e-8


def test_lqr_closed_loop_stable_across_speeds():
    p = VehicleParams()
    for v in (3.0, 10.0, 25.0):
        A, B = error_dynamics(p, v)
        Ad, Bd = c2d_zoh(A, B, 0.05)
        K, _ = dlqr(Ad, Bd, np.diag([1, 0, 1, 0]), np.array([[10.0]]))
        assert np.max(np.abs(np.linalg.eigvals(Ad - Bd @ K))) < 1.0


def test_lqr_reduces_lateral_offset_on_dynamic_model():
    p = VehicleParams()
    ctl = LQRLateralController(p)
    path = ReferencePath.from_trajectory(straight_path(400.0, 801))
    v, dt = 15.0, 0.05
    A, B = error_dynamics(p, v)
    Ad, Bd = c2d_zoh(A, B, dt)
    x = np.array([1.0, 0.0, 0.0, 0.0])  # e_y, de_y, e_psi, de_psi
    pose_x = 0.0
    for _ in range(200):
        pose = Pose2D(pose_x, x[0], x[2])
        out = ctl.update(pose, v, x[3], path, dt)
        x = Ad @ x + Bd.ravel() * out.delta_rad
        pose_x += v * dt
    assert abs(x[0]) < 0.05 and abs(x[2]) < 0.01
