import numpy as np
import pytest

from cheesestack.planning import (LateralPathSampler, ReferencePath, SmoothingConfig, path_curvature,
                                  resample_polyline, smooth_path)


def test_smoothing_reduces_roughness_and_keeps_endpoints():
    rng = np.random.default_rng(0)
    x = np.arange(0, 50, 1.0)
    noisy = np.stack([x, rng.normal(0, 0.3, len(x))], 1)
    sm = smooth_path(noisy, SmoothingConfig(w_smooth=20.0))
    rough = lambda p: float(np.sum(np.diff(p[:, 1], 2) ** 2))
    assert rough(sm) < 0.05 * rough(noisy)
    assert np.allclose(sm[0], noisy[0], atol=1e-3) and np.allclose(sm[-1], noisy[-1], atol=1e-3)
    # zero smoothing returns the input
    assert np.allclose(smooth_path(noisy, SmoothingConfig(w_smooth=0.0)), noisy)


def test_resample_and_curvature_on_circle():
    th = np.linspace(0, np.pi, 1000)
    circ = np.stack([20 * np.cos(th), 20 * np.sin(th)], 1)
    rs = resample_polyline(circ, 0.5)
    seg = np.linalg.norm(np.diff(rs, axis=0), axis=1)
    assert np.allclose(seg[:-1], 0.5, atol=1e-3)  # chord across vertices
    k = path_curvature(rs)
    assert np.median(k[5:-5]) == pytest.approx(1 / 20, rel=0.02)


def test_lateral_sampler_prefers_centre_and_avoids_obstacle():
    ref = ReferencePath.straight(100)
    ls = LateralPathSampler(ref)
    best = ls.best(0.0, 0.0)
    assert best.d_end == 0.0 and np.allclose(best.d, 0.0)
    obs = np.array([[25.0, 0.0, 1.0]])
    best = ls.best(0.0, 0.0, obstacles_sd=obs)
    assert abs(best.d_end) >= 1.8
    assert best.min_clearance >= ls.config.safety_margin
    assert best.xy.shape == (len(best.s), 2)
    # starting offset: converges back to preferred lane centre
    best = ls.best(0.0, 1.5, preferred_d=0.0)
    assert best.d[0] == pytest.approx(1.5) and best.d_end == 0.0
