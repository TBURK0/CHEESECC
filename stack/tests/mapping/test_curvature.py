import math

import numpy as np
import pytest

from cheesestack.mapping import (
    ClothoidSegment, CubicSpline1D, CubicSpline2D, Polyline, curvature_profile, discrete_curvature,
    fit_clothoid_g1, fit_linear_curvature, heading_rate_curvature, smooth_polyline, smooth_signal,
)


def circle(r, n=60, ccw=True):
    a = np.linspace(0, math.pi, n)
    pts = np.column_stack([r * np.cos(a), r * np.sin(a)])
    return pts if ccw else pts[::-1]


def test_discrete_curvature_circle_sign():
    k = discrete_curvature(circle(20.0))
    np.testing.assert_allclose(k, 1 / 20.0, rtol=1e-3)
    assert np.all(discrete_curvature(circle(20.0, ccw=False)) < 0)
    np.testing.assert_allclose(heading_rate_curvature(circle(20.0)), 1 / 20.0, rtol=1e-2)
    assert np.all(discrete_curvature([[0, 0], [1, 0], [2, 0]]) == 0)
    assert len(discrete_curvature([[0, 0], [1, 0]])) == 2


def test_curvature_profile_on_sample_arc(sample_map):
    s, k = curvature_profile(sample_map["r2_0"].centerline, ds=1.0)
    assert np.median(k) == pytest.approx(1 / 30.0, rel=0.05)
    s, k = curvature_profile(sample_map["r4_0"].centerline, ds=1.0)
    assert np.median(k) == pytest.approx(-1 / 20.0, rel=0.05)


def test_smoothing_reduces_noise():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 50, 51)
    clean = np.column_stack([x, np.zeros_like(x)])
    noisy = clean + np.column_stack([np.zeros_like(x), rng.normal(0, 0.2, len(x))])
    sm = smooth_polyline(noisy, iterations=20)
    assert np.abs(discrete_curvature(sm)).mean() < 0.3 * np.abs(discrete_curvature(noisy)).mean()
    np.testing.assert_allclose(sm[[0, -1]], noisy[[0, -1]])
    sig = smooth_signal([0, 0, 10, 0, 0], window=3)
    assert sig.shape == (5,) and sig[2] == pytest.approx(10 / 3)


def test_clothoid_basics_and_g1_fit():
    arc = ClothoidSegment(0, 0, 0, 1 / 10.0, 0.0, math.pi * 10 / 2)  # quarter circle r=10
    x, y, th = arc.end_pose()
    assert (x, y, th) == pytest.approx((10, 10, math.pi / 2), abs=1e-9)
    pts = arc.sample(40)
    np.testing.assert_allclose(pts[-1], [10, 10], atol=1e-6)
    np.testing.assert_allclose(np.hypot(pts[:, 0], pts[:, 1] - 10), 10, atol=1e-6)

    truth = ClothoidSegment(2.0, -1.0, 0.4, 0.01, 0.003, 35.0)
    ex, ey, eth = truth.end_pose()
    fit = fit_clothoid_g1([2.0, -1.0], 0.4, [ex, ey], eth)
    assert fit is not None
    assert (fit.kappa0, fit.dkappa, fit.length) == pytest.approx((0.01, 0.003, 35.0), rel=1e-5)
    assert fit_clothoid_g1([0, 0], 0, [0, 0], 1.0) is None


def test_fit_linear_curvature():
    s = np.linspace(0, 20, 30)
    k0, dk = fit_linear_curvature(s, 0.02 + 0.001 * s)
    assert (k0, dk) == pytest.approx((0.02, 0.001))


def test_cubic_splines():
    x = np.linspace(0, 2 * math.pi, 30)
    sp = CubicSpline1D(x, np.sin(x))
    t = np.linspace(0.5, 5.5, 11)
    np.testing.assert_allclose(sp(t), np.sin(t), atol=2e-3)
    np.testing.assert_allclose(sp(t, 1), np.cos(t), atol=2e-2)
    with pytest.raises(ValueError):
        CubicSpline1D([0, 0], [1, 2])
    cs = CubicSpline2D(circle(15.0, n=25))
    s, pts, h, k = cs.sample(1.0)
    inner = slice(5, -5)
    np.testing.assert_allclose(k[inner], 1 / 15.0, rtol=0.02)
    assert cs.length == pytest.approx(math.pi * 15, rel=1e-3)
