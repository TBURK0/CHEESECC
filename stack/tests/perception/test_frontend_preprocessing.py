import numpy as np
import pytest

from cheesestack.perception.preprocessing import (
    PreprocessConfig, Preprocessor, clahe_lite, crop_roi, equalize_hist, gaussian_blur, gaussian_kernel1d, grayscale,
    normalize, resize, to_float, to_uint8,
)


def test_dtype_roundtrip_and_gray():
    img = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
    f = to_float(img)
    assert f.dtype == np.float32 and f.max() == 1.0
    np.testing.assert_array_equal(to_uint8(f), img)
    np.testing.assert_allclose(grayscale(img)[0], [0.299, 0.587], atol=1e-6)


def test_resize_constant_and_linear():
    img = np.full((10, 20, 3), 0.3, np.float32)
    out = resize(img, (5, 7))
    assert out.shape == (5, 7, 3) and np.allclose(out, 0.3)
    ramp = np.tile(np.arange(8, dtype=np.float32), (4, 1))
    up = resize(ramp, (4, 16))
    assert np.all(np.diff(up[0]) >= -1e-6)  # monotone
    assert resize(ramp, (2, 4), method="nearest").shape == (2, 4)
    with pytest.raises(ValueError):
        resize(ramp, (2, 2), method="cubic")


def test_normalize_crop():
    img = np.full((4, 4, 3), 0.75, np.float32)
    np.testing.assert_allclose(normalize(img, [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]), 1.0)
    c = crop_roi(np.zeros((10, 10)), (-2, 3, 5, 100))
    assert c.shape == (7, 3)
    assert crop_roi(np.zeros((10, 10)), (20, 20, 3, 3)).size == 0


def test_gaussian_blur_preserves_mean_and_smooths():
    k = gaussian_kernel1d(1.5)
    assert k.sum() == pytest.approx(1.0) and len(k) == 11
    rng = np.random.default_rng(0)
    img = rng.random((32, 32)).astype(np.float32)
    b = gaussian_blur(img, 2.0)
    assert abs(b.mean() - img.mean()) < 0.01
    assert b.std() < img.std() * 0.5
    impulse = np.zeros((9, 9), np.float32)
    impulse[4, 4] = 1
    bi = gaussian_blur(impulse, 1.0)
    assert bi.sum() == pytest.approx(1.0, abs=1e-5) and bi[4, 4] == bi.max()


def test_equalize_hist_spreads_range():
    rng = np.random.default_rng(1)
    img = (0.4 + 0.1 * rng.random((32, 32))).astype(np.float32)
    eq = equalize_hist(img)
    assert eq.min() < 0.05 and eq.max() > 0.95
    assert np.all(np.diff(eq.ravel()[np.argsort(img.ravel())]) >= -1e-6)  # monotone mapping
    col = equalize_hist(np.stack([img] * 3, -1))
    assert col.shape == (32, 32, 3)
    cl = clahe_lite(img, (2, 2))
    assert cl.shape == img.shape and cl.std() > img.std()


def test_preprocessor_chain_and_back_mapping():
    img = np.random.default_rng(2).random((40, 60, 3)).astype(np.float32)
    pp = Preprocessor(PreprocessConfig(roi=(10, 20, 40, 20), out_hw=(10, 20), gray=True, blur_sigma=0.5, normalize=True))
    out = pp.process(img)
    assert out.shape == (10, 20)
    raw = pp.to_raw(np.array([[0, 0], [19, 9]]))
    np.testing.assert_allclose(raw[0], [10.5, 20.5])
    np.testing.assert_allclose(raw[1], [48.5, 38.5])
