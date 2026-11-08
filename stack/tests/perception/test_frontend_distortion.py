import numpy as np
import pytest

from cheesestack.perception.distortion import (
    BrownConrady, UndistortMap, distort_image, distort_points, remap_bilinear, undistort_points,
)

K = np.array([[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]])


def test_identity_and_array_roundtrip():
    d = BrownConrady()
    assert d.is_identity
    xy = np.array([[0.1, -0.2]])
    np.testing.assert_allclose(d.distort_normalized(xy), xy)
    assert BrownConrady.from_array([0.1, 0.2]).k2 == 0.2


@pytest.mark.parametrize("coeffs", [(-0.3, 0.1, 0, 0, 0), (0.1, -0.02, 0.002, -0.001, 0.001), (-0.25, 0.08, 0.001, 0.001, -0.01)])
def test_point_roundtrip(coeffs):
    d = BrownConrady(*coeffs)
    g = np.stack(np.meshgrid(np.linspace(0, 319, 9), np.linspace(0, 239, 7)), -1).reshape(-1, 2)
    np.testing.assert_allclose(undistort_points(distort_points(g, K, d), K, d), g, atol=1e-6)


def test_barrel_moves_corners_inward():
    d = BrownConrady(-0.3)
    uvd = distort_points(np.array([[0.0, 0.0], [160, 120]]), K, d)
    assert uvd[0, 0] > 0 and uvd[0, 1] > 0
    np.testing.assert_allclose(uvd[1], [160, 120])


def test_remap_bilinear():
    img = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = remap_bilinear(img, np.array([[0.5, 3.0, -1.0]]), np.array([[0.5, 2.0, 0.0]]), fill=-1)
    np.testing.assert_allclose(out, [[2.5, 11.0, -1.0]])
    rgb = np.stack([img] * 3, -1)
    assert remap_bilinear(rgb, np.zeros((2, 2)), np.zeros((2, 2))).shape == (2, 2, 3)


def test_undistort_image_recovers_straight_lines():
    H, W = 240, 320
    img = np.zeros((H, W), np.float32)
    img[:, 60:64] = 1.0  # vertical line far from center -> bent by distortion
    d = BrownConrady(-0.25, 0.05)
    distorted = distort_image(img, K, d)
    # line is bent in distorted image: column of max differs between top and middle rows
    assert abs(np.argmax(distorted[5]) - np.argmax(distorted[120])) >= 3
    rec = UndistortMap(K, d, (H, W)).apply(distorted)
    inner = slice(30, 210)
    cols = np.argmax(rec[inner], axis=1)
    assert np.all(np.abs(cols - 61) <= 2)
