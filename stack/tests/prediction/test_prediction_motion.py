import math

import numpy as np
import pytest

from cheesestack.core import Header, Pose2D, VehicleState
from cheesestack.prediction import EgoPredictor, MotionModel, MotionModelConfig, MotionModelType


def _state(v=10.0, w=0.0, a=0.0, yaw=0.0):
    return np.array([0.0, 0.0, yaw, v, w, a])


def test_cv_straight_line():
    tr = MotionModel(MotionModelType.CV).rollout(_state(10.0, w=0.5, a=2.0), horizon=2.0)
    assert len(tr) == 21
    assert tr.states[-1, 0] == pytest.approx(20.0, abs=1e-6)
    assert tr.states[-1, 1] == pytest.approx(0.0, abs=1e-9)  # yaw rate ignored by CV


def test_ca_accel_and_stop_no_reverse():
    tr = MotionModel(MotionModelType.CA).rollout(_state(10.0, a=2.0), horizon=2.0)
    assert tr.states[-1, 0] == pytest.approx(10 * 2 + 0.5 * 2 * 4, abs=1e-6)
    assert tr.states[-1, 3] == pytest.approx(14.0)
    stop = MotionModel(MotionModelType.CA).rollout(_state(10.0, a=-5.0), horizon=5.0)
    assert stop.states[-1, 0] == pytest.approx(10.0, abs=1e-6)  # v^2 / 2|a|
    assert stop.states[-1, 3] == 0.0
    assert np.all(np.diff(stop.states[:, 0]) >= -1e-12)


def test_ctrv_circle():
    v, w = 5.0, 0.5
    T = 2 * math.pi / w
    tr = MotionModel(MotionModelType.CTRV, MotionModelConfig(max_yaw_rate=2.0)).rollout(_state(v, w=w), horizon=T / 2, dt=T / 200)
    r = v / w
    assert tr.states[-1, 0] == pytest.approx(0.0, abs=1e-6)
    assert tr.states[-1, 1] == pytest.approx(2 * r, abs=1e-6)
    radii = np.linalg.norm(tr.positions - np.array([0.0, r]), axis=1)
    assert np.allclose(radii, r, atol=1e-6)


def test_uncertainty_grows_and_traj_conversion():
    tr = MotionModel(MotionModelType.CTRA).rollout(_state(10.0, w=0.1, a=0.5), horizon=3.0)
    tr_cov = np.trace(tr.pos_cov, axis1=1, axis2=2)
    assert np.all(np.diff(tr_cov) > 0)
    core = tr.to_trajectory()
    assert len(core) == len(tr) and core.as_array().shape == (len(tr), 7)
    mid = tr.state_at(0.05)
    assert tr.states[0, 0] < mid[0] < tr.states[1, 0]


def test_ego_predictor_uses_curvature_fallback():
    vs = VehicleState(Header(1.0, "map"), Pose2D(0, 0, 0), v=10.0, curvature=0.02)
    tr = EgoPredictor().predict(vs, horizon=3.0)
    assert tr.states[0, 4] == pytest.approx(0.2)
    assert tr.states[-1, 1] > 0.5  # turned left
    assert tr.stamp == 1.0 and tr.label == "ego"
