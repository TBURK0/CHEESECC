import numpy as np
import pytest

from cheesestack.core import Trajectory
from cheesestack.planning import FrenetState, FrenetTrajectoryGenerator, ObstacleBox, ReferencePath
from cheesestack.planning.trajectory import frenet_to_cartesian


def test_velocity_keeping_on_straight():
    gen = FrenetTrajectoryGenerator(ReferencePath.straight(200))
    res = gen.generate(FrenetState(0.0, 10.0), v_target=12.0, stamp=3.0)
    assert res.feasible and not res.fallback
    assert isinstance(res.trajectory, Trajectory) and res.trajectory.header.stamp == 3.0
    arr = res.trajectory.as_array()
    assert arr[0, 4] == pytest.approx(10.0, abs=0.05)
    assert arr[-1, 4] == pytest.approx(12.0, abs=0.05)
    assert np.allclose(arr[:, 2], 0.0, atol=1e-6)
    assert np.all(np.diff(arr[:, 0]) > 0)


def test_lane_change_target():
    gen = FrenetTrajectoryGenerator(ReferencePath.straight(200))
    res = gen.generate(FrenetState(0.0, 15.0), v_target=15.0, d_target=3.6)
    arr = res.trajectory.as_array()
    assert res.feasible
    assert arr[-1, 2] == pytest.approx(3.6, abs=0.05)
    assert np.abs(arr[:, 4] ** 2 * arr[:, 6]).max() < gen.config.constraints.max_lateral_accel


def test_avoids_static_obstacle():
    gen = FrenetTrajectoryGenerator(ReferencePath.straight(200))
    obs = [ObstacleBox(30.0, 0.0, 0.0)]
    res = gen.generate(FrenetState(0.0, 10.0), v_target=12.0, obstacles=obs)
    assert res.feasible and abs(res.chosen.d_end) >= 1.8
    assert gen.checker.check(res.trajectory, obs).feasible


def test_stopping_mode():
    gen = FrenetTrajectoryGenerator(ReferencePath.straight(200))
    res = gen.generate(FrenetState(0.0, 8.0), v_target=0.0, stop_s=25.0)
    arr = res.trajectory.as_array()
    assert res.feasible
    assert arr[-1, 4] == pytest.approx(0.0, abs=0.05)
    assert arr[-1, 1] == pytest.approx(25.0, abs=0.1)


def test_fallback_when_blocked():
    gen = FrenetTrajectoryGenerator(ReferencePath.straight(200))
    wall = [ObstacleBox(12.0, y, 0.0, 2.0, 2.0) for y in np.arange(-6, 6.1, 1.5)]
    res = gen.generate(FrenetState(0.0, 15.0), v_target=15.0, obstacles=wall)
    assert not res.feasible and res.fallback
    arr = res.trajectory.as_array()
    assert np.all(np.diff(arr[:, 4]) <= 1e-9) and arr[-1, 4] == 0.0


def test_frenet_to_cartesian_on_curve():
    th = np.linspace(0, np.pi / 2, 300)
    ref = ReferencePath(np.stack([40 * np.sin(th), 40 * (1 - np.cos(th))], 1))
    t = np.arange(0, 3, 0.1)
    s = 5.0 + 10.0 * t
    arr = frenet_to_cartesian(ref, t, s, np.full_like(t, 10.0), np.zeros_like(t), np.zeros_like(t))
    assert np.median(arr[:, 6]) == pytest.approx(1 / 40, rel=0.05)
    assert np.allclose(arr[:, 4], 10.0, atol=1e-6)
