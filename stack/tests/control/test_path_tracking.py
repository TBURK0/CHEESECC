import math

import numpy as np
import pytest

from cheesestack.core import Pose2D
from cheesestack.control import (
    PurePursuit, ReferencePath, RearWheelFeedback, Stanley, pure_pursuit_curvature,
)
from _sim import circle_xy, step_kinematic, straight_path


def test_pure_pursuit_formula_matches_app():
    assert pure_pursuit_curvature(10.0, 0.0) == 0.0
    assert pure_pursuit_curvature(10.0, math.pi / 6) == pytest.approx(2 * 0.5 / 10)
    assert pure_pursuit_curvature(0.2, 0.1) == pytest.approx(2 * math.sin(0.1) / 1.0)  # lookahead floor 1 m
    assert pure_pursuit_curvature(5.0, 3.0) == pytest.approx(2 / 5.0)  # alpha clamp pi/2


def test_reference_path_projection_signs():
    path = ReferencePath.from_trajectory(straight_path())
    p = path.project(Pose2D(10.0, 1.5, 0.2))
    assert p.cross_track == pytest.approx(1.5)
    assert p.heading_error == pytest.approx(0.2)
    assert p.s == pytest.approx(10.0)
    xy, th = circle_xy(30.0)
    cp = ReferencePath(xy)
    assert np.median(cp.curvature) == pytest.approx(1 / 30.0, rel=0.02)


def test_lookahead_is_speed_adaptive():
    pp = PurePursuit()
    assert pp.lookahead(0.0) == pp.config.lookahead_min_m
    assert pp.lookahead(10.0) > pp.lookahead(5.0)
    assert pp.lookahead(100.0) == pp.config.lookahead_max_m


@pytest.mark.parametrize("ctrl", [PurePursuit(), Stanley(), RearWheelFeedback()])
def test_converges_to_straight_line(ctrl):
    path = ReferencePath.from_trajectory(straight_path())
    pose, v, dt = Pose2D(0.0, 2.0, 0.0), 8.0, 0.05
    for _ in range(300):
        k = max(-0.2, min(0.2, ctrl.update(pose, v, path).curvature))
        pose = step_kinematic(pose, v, k, dt)
    res = path.project(pose)
    assert abs(res.cross_track) < 0.1
    assert abs(res.heading_error) < 0.05


@pytest.mark.parametrize("ctrl", [PurePursuit(), Stanley(), RearWheelFeedback()])
def test_tracks_circle(ctrl):
    xy, _ = circle_xy(30.0)
    path = ReferencePath(xy)
    pose, v, dt = Pose2D(0.0, -0.5, 0.0), 6.0, 0.05
    errs = []
    for i in range(250):
        r = ctrl.update(pose, v, path)
        pose = step_kinematic(pose, v, r.curvature, dt)
        if i > 120:
            errs.append(abs(r.cross_track))
    assert max(errs) < 0.6  # pure pursuit cuts corners slightly; others ~0
