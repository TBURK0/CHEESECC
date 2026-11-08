import math

import numpy as np
import pytest

from cheesestack.perception.calibration import (
    CalibrationConfig, CameraExtrinsics, CameraIntrinsics, InversePerspectiveMapper, IPMConfig, PinholeCamera,
    apply_homography, calibrate_zhang, checkerboard_points, decompose_projection, estimate_homography,
    estimate_projection_dlt, project_points, rodrigues, rodrigues_inv, rq,
)
from cheesestack.perception.distortion import BrownConrady


@pytest.fixture
def cam():
    return PinholeCamera(CameraIntrinsics(500, 520, 320, 240, 640, 480), CameraExtrinsics(1.5, 0.0, 1.3, 0.0, 0.1, 0.0))


def test_optical_axis_and_conventions(cam):
    c0 = PinholeCamera(CameraIntrinsics(500, 500, 320, 240, 640, 480), CameraExtrinsics(0, 0, 1, 0, 0, 0))
    uv, ok = c0.project([[10, 0, 1], [10, 1, 1], [10, 0, 2], [-5, 0, 1]])
    np.testing.assert_allclose(uv[0], [320, 240])
    assert uv[1, 0] < 320  # +y (left) -> smaller u
    assert uv[2, 1] < 240  # +z (up) -> smaller v
    assert list(ok) == [True, True, True, False]
    # pitch down: horizon moves up (smaller v) -> point at horizon height appears above center
    uvp, _ = cam.project([[1000, 0, 1.3]])
    assert uvp[0, 1] < 240


def test_project_unproject_ground_roundtrip(cam):
    pts = np.array([[10, 0, 0], [20, 2, 0], [8, -3, 0.0]])
    uv, ok = cam.project(pts)
    assert ok.all()
    g, gok = cam.pixel_to_ground(uv)
    assert gok.all()
    np.testing.assert_allclose(g, pts[:, :2], atol=1e-9)
    np.testing.assert_allclose(apply_homography(cam.ground_homography(), pts[:, :2]), uv, atol=1e-9)
    _, sky = cam.pixel_to_ground([[320, 0]])  # top of image looks above horizon
    assert not sky[0]


def test_projection_with_distortion_roundtrip(cam):
    cam.distortion = BrownConrady(-0.2, 0.05, 0.001, -0.001)
    pts = np.array([[12, 3, 0], [6, -2, 0.5]])
    uv, _ = cam.project(pts)
    rays = cam.unproject(uv)
    d = pts - cam.extrinsics.t_base_cam
    np.testing.assert_allclose(rays, d / np.linalg.norm(d, axis=1, keepdims=True), atol=1e-9)


def test_extrinsics_roundtrip_and_rodrigues():
    e = CameraExtrinsics(1.0, 0.2, 1.4, 0.05, 0.12, -0.3)
    e2 = CameraExtrinsics.from_Rt(e.R_base_cam, e.t_base_cam)
    for k in ("x", "y", "z", "roll", "pitch", "yaw"):
        assert getattr(e2, k) == pytest.approx(getattr(e, k))
    r = np.array([0.3, -0.2, 0.9])
    np.testing.assert_allclose(rodrigues_inv(rodrigues(r)), r, atol=1e-12)
    np.testing.assert_allclose(rodrigues_inv(rodrigues([0, math.pi - 1e-9, 0])), [0, math.pi, 0], atol=1e-4)


def test_intrinsics_helpers():
    i = CameraIntrinsics.from_fov(640, 480, math.radians(90))
    assert i.fx == pytest.approx(320)
    s = i.scaled(0.5)
    assert s.width == 320 and s.fx == pytest.approx(160) and s.cx == pytest.approx((i.cx + 0.5) * 0.5 - 0.5)
    np.testing.assert_allclose(CameraIntrinsics.from_K(i.K).K, i.K)


def test_homography_dlt():
    H = np.array([[1.2, 0.1, 5], [-0.05, 0.9, 3], [1e-3, 2e-3, 1]])
    src = np.random.default_rng(0).uniform(0, 100, (10, 2))
    dst = apply_homography(H, src)
    np.testing.assert_allclose(estimate_homography(src, dst), H, atol=1e-8)
    with pytest.raises(ValueError):
        estimate_homography(src[:3], dst[:3])


def test_projection_dlt_and_decompose(cam):
    X = np.random.default_rng(0).uniform([5, -5, -1], [30, 5, 2], (20, 3))
    uv, _ = cam.project(X)
    K, R, t = decompose_projection(estimate_projection_dlt(X, uv))
    np.testing.assert_allclose(K, cam.K, atol=1e-6)
    np.testing.assert_allclose(R.T, cam.extrinsics.R_base_cam, atol=1e-9)
    np.testing.assert_allclose(-R.T @ t, cam.extrinsics.t_base_cam, atol=1e-6)
    A = np.random.default_rng(1).normal(size=(3, 3))
    Rr, Q = rq(A)
    np.testing.assert_allclose(Rr @ Q, A, atol=1e-12)
    assert np.allclose(np.tril(Rr, -1), 0) and np.all(np.diag(Rr) > 0)


def _views(K, dist, n=5, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    obj = checkerboard_points(7, 9, 0.1)  # 0.8 x 0.6 m board filling most of the frame
    objs, imgs = [], []
    for i in range(n):
        R = rodrigues(rng.normal(0, 0.25, 3))
        t = np.array([-0.4, -0.3, 1.2 + 0.1 * i])
        imgs.append(project_points(obj, K, R, t, dist) + rng.normal(0, noise, (len(obj), 2)))
        objs.append(obj)
    return objs, imgs


def test_zhang_linear_exact():
    K = np.array([[500, 0, 320], [0, 520, 240], [0, 0, 1.0]])
    res = calibrate_zhang(*_views(K, None, n=4), CalibrationConfig(refine=False))
    np.testing.assert_allclose(res.K, K, atol=1e-6)
    assert res.rms_px < 1e-6


def test_zhang_refined_with_distortion_and_noise():
    K = np.array([[500, 0, 320], [0, 520, 240], [0, 0, 1.0]])
    dist = BrownConrady(-0.1, 0.02)
    res = calibrate_zhang(*_views(K, dist, n=5, noise=0.2, seed=3))
    assert abs(res.K[0, 0] - 500) < 5 and abs(res.K[1, 1] - 520) < 5
    assert abs(res.K[0, 2] - 320) < 5 and abs(res.K[1, 2] - 240) < 5
    assert abs(res.dist.k1 + 0.1) < 0.03
    assert res.rms_px < 0.5


def test_ipm_maps_ground_stripe(cam):
    # Paint a bright stripe at y in [-0.5, 0.5] on the ground in the image, then IPM it back.
    H, W = 480, 640
    vv, uu = np.mgrid[0:H, 0:W]
    g, ok = cam.pixel_to_ground(np.column_stack([uu.ravel(), vv.ravel()]))
    img = np.where(ok & (np.abs(np.nan_to_num(g[:, 1], nan=99)) < 0.5), 1.0, 0.2).reshape(H, W).astype(np.float32)
    ipm = InversePerspectiveMapper(cam, IPMConfig(x_range=(5, 25), y_range=(-4, 4), resolution=0.2))
    bev = ipm.warp(img)
    assert bev.shape == ipm.shape == (100, 40)
    xy = ipm.bev_to_base(np.array([[0, 0], [99, 39]]))
    np.testing.assert_allclose(xy, [[24.9, 3.9], [5.1, -3.9]])
    center = np.abs(ipm.xy[..., 1]) < 0.3
    side = np.abs(ipm.xy[..., 1]) > 1.5
    assert bev[center].mean() > 0.9 and bev[side].mean() < 0.3
