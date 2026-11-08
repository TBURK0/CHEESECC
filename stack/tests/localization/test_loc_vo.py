import math

import numpy as np
import pytest

from cheesestack.core import Pose2D
from cheesestack.localization import visual_odometry as V


def test_umeyama_exact():
    rng = np.random.default_rng(0)
    src = rng.uniform(-5, 5, (20, 2))
    T = Pose2D(1.0, -2.0, 0.3)
    R, t, s = V.umeyama_2d(src, T.transform_points(src))
    assert math.atan2(R[1, 0], R[0, 0]) == pytest.approx(0.3)
    assert np.allclose(t, [1, -2]) and s == 1.0
    R, t, s = V.umeyama_2d(src, 2.0 * src, with_scale=True)
    assert s == pytest.approx(2.0)


def test_ransac_rejects_outliers():
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 20, (60, 2))
    T = Pose2D(0.8, 0.05, 0.02)
    dst = T.transform_points(src) + rng.normal(0, 0.01, src.shape)
    out = rng.choice(60, 18, replace=False)
    dst[out] += rng.uniform(-5, 5, (18, 2))
    est, inl = V.ransac_rigid_2d(src, dst)
    assert est.x == pytest.approx(0.8, abs=0.02) and est.yaw == pytest.approx(0.02, abs=0.002)
    assert not inl[out].any() or inl[out].sum() <= 2


def test_ground_projection_roundtrip_and_vo():
    cam = V.CameraModel(pitch_rad=0.1)
    proj = V.GroundPlaneProjector(cam)
    rng = np.random.default_rng(2)
    world = np.column_stack([rng.uniform(6, 25, 80), rng.uniform(-4, 4, 80)])  # ground pts in frame k-1
    uv0 = proj.ground_to_pixels(world)
    back, valid = proj.pixels_to_ground(uv0)
    assert valid.all() and np.allclose(back, world, atol=1e-6)
    motion = Pose2D(1.0, 0.02, 0.01)  # base_link_k in base_link_{k-1}
    uv1 = proj.ground_to_pixels(motion.inverse().transform_points(world))
    vo = V.PlanarVisualOdometry(projector=proj)
    res = vo.estimate(0.1, uv0, uv1)
    assert res.ok and res.inlier_ratio > 0.95
    assert res.delta.x == pytest.approx(1.0, abs=1e-3) and res.delta.yaw == pytest.approx(0.01, abs=1e-4)
    assert res.covariance.shape == (3, 3)
    # Pixels above horizon are invalid.
    _, v = proj.pixels_to_ground([[960, 0]])
    assert not v[0]


def test_vo_too_few_points():
    res = V.PlanarVisualOdometry().estimate(0.0, [[0, 0]], [[1, 1]])
    assert not res.ok
