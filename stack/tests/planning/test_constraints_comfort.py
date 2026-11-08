import numpy as np
import pytest

from cheesestack.core import Header, Trajectory, TrajectoryPoint
from cheesestack.planning import ComfortModel, ConstraintChecker, ObstacleBox
from cheesestack.planning.comfort import iso_band


def straight(v=10.0, a=0.0, k=0.0, n=51, dt=0.1):
    t = np.arange(n) * dt
    x = v * t
    return np.stack([t, x, np.zeros(n), np.zeros(n), np.full(n, v), np.full(n, a), np.full(n, k)], 1)


def test_feasible_straight():
    rep = ConstraintChecker().check_array(straight())
    assert rep.feasible and rep.violations == []


def test_curvature_and_lateral_accel_violations():
    rep = ConstraintChecker().check_array(straight(v=20.0, k=0.05))
    assert "lateral_accel" in rep.names and "curvature" not in rep.names
    assert rep.max_lateral_accel == pytest.approx(20.0)
    assert "curvature" in ConstraintChecker().check_array(straight(v=1.0, k=0.3)).names


def test_accel_and_jerk_violations():
    arr = straight(a=0.0)
    arr[20:, 5] = 3.0  # step -> jerk 30 m/s^3 and accel over limit
    rep = ConstraintChecker().check_array(arr)
    assert {"accel_max", "jerk"} <= rep.names
    assert "accel_min" in ConstraintChecker().check_array(straight(a=-8.0)).names


def test_collision_static_and_moving():
    ch = ConstraintChecker()
    rep = ch.check_array(straight(), [ObstacleBox(30.0, 0.0, 0.0, track_id=5)])
    assert not rep.feasible and rep.collision_track == 5
    # obstacle in the next lane does not collide
    assert ch.check_array(straight(), [ObstacleBox(30.0, 3.6, 0.0)]).feasible
    # same-speed lead stays ahead
    assert ch.check_array(straight(), [ObstacleBox(15.0, 0.0, 0.0, vx=10.0)]).feasible
    # crossing vehicle arrives at x=25 when ego does (t=2.5 s)
    rep = ch.check_array(straight(), [ObstacleBox(25.0, -25.0, np.pi / 2, vy=10.0)])
    assert rep.collision_index is not None and 1.8 < rep.violations[-1][2] < 3.2


def test_check_trajectory_wrapper():
    pts = [TrajectoryPoint(*r) for r in straight()]
    assert ConstraintChecker().check(Trajectory(Header(0.0), pts)).feasible


def test_iso_bands_and_score():
    assert iso_band(0.1) == "not uncomfortable"
    assert iso_band(1.0) == "uncomfortable"
    assert iso_band(5.0) == "extremely uncomfortable"
    cm = ComfortModel()
    smooth = cm.evaluate_array(straight())
    assert smooth.score == 1.0 and smooth.cost == 0.0
    t = np.arange(0, 5, 0.1)
    harsh = cm.evaluate_signals(t, 1.5 * np.sin(2 * t), 1.5 * np.cos(2 * t))
    assert harsh.a_v == pytest.approx(1.4 * 1.5, rel=0.05)
    assert harsh.score < 0.1 and harsh.cost > 0 and harsh.peak_jerk_lon > 2.5
    mild = cm.evaluate_signals(t, 0.2 * np.sin(2 * t), np.zeros_like(t))
    assert mild.band == "not uncomfortable" and mild.score > harsh.score
