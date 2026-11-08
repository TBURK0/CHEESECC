import numpy as np
import pytest

from cheesestack.perception.confidence import (
    ConfidenceConfig, ConfidenceScorer, HistogramBinningCalibrator, PlattScaler, SensorHealth, TemperatureScaler,
    aggregate, brier_score, expected_calibration_error, logit, sigmoid,
)
from cheesestack.perception.environment import Condition, EnvironmentConfig, EnvironmentEstimator, image_stats
from cheesestack.perception.preprocessing import gaussian_blur


def _scene(seed=0, scale=1.0, offset=0.0):
    rng = np.random.default_rng(seed)
    H, W = 60, 80
    base = np.tile(np.linspace(0.25, 0.75, H)[:, None], (1, W))
    blocks = (rng.random((6, 8)) * 0.5 - 0.25).repeat(10, 0).repeat(10, 1)
    img = base + blocks + rng.normal(0, 0.03, (H, W))
    return np.clip(img * scale + offset, 0, 1).astype(np.float32)


def _est(img, n=1):
    e = EnvironmentEstimator(EnvironmentConfig(ema_alpha=1.0))
    for _ in range(n):
        r = e.update(img, 0.0)
    return r


def test_nominal_scene():
    r = _est(_scene())
    assert r.condition == Condition.NOMINAL
    assert r.visibility > 0.6


def test_night_scene():
    img = _scene(scale=0.12)
    img[10:12, 30:32] = 1.0  # a headlight
    r = _est(img)
    assert r.condition == Condition.NIGHT
    assert r.visibility < _est(_scene()).visibility


def test_glare_scene():
    img = _scene()
    img[0:15, 20:50] = 1.0
    r = _est(img)
    assert r.condition == Condition.GLARE


def test_blur_and_fog_and_blocked():
    sharp = _scene()
    assert _est(gaussian_blur(sharp, 3.0)).score(Condition.BLUR) > _est(sharp).score(Condition.BLUR)
    fog = 0.5 + (sharp - 0.5) * 0.15
    assert _est(fog).condition == Condition.FOG
    assert _est(np.full((40, 40), 0.3, np.float32)).condition == Condition.BLOCKED
    assert _est(np.full((40, 40), 0.3, np.float32)).visibility == pytest.approx(0.0, abs=1e-6)


def test_rain_streaks():
    rng = np.random.default_rng(4)
    img = _scene()
    for c in rng.choice(80, 25, replace=False):
        img[:, c] = np.clip(img[:, c] + 0.35, 0, 1)
    assert _est(img).score(Condition.RAIN) > _est(_scene()).score(Condition.RAIN)


def test_ema_smoothing_and_stats():
    e = EnvironmentEstimator(EnvironmentConfig(ema_alpha=0.2))
    e.update(_scene(), 0.0)
    r = e.update(_scene(scale=0.1), 0.1)
    assert r.score(Condition.NIGHT) < 0.5  # smoothed, not yet declared
    s = image_stats(_scene())
    assert 0.3 < s.brightness < 0.7 and s.sharpness > 0


def test_aggregate_methods():
    v = [0.9, 0.5]
    assert aggregate(v, method="mean") == pytest.approx(0.7)
    assert aggregate(v, method="min") == 0.5
    assert aggregate(v, method="geometric") == pytest.approx(np.sqrt(0.45), rel=1e-6)
    assert aggregate(v, method="noisy_or") == pytest.approx(0.95)
    assert aggregate(v, method="harmonic") < aggregate(v, method="geometric")
    assert aggregate([]) == 0.0
    with pytest.raises(ValueError):
        aggregate(v, method="median")


def _overconfident(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    true_logit = rng.normal(0, 1.5, n)
    y = (rng.random(n) < sigmoid(true_logit)).astype(float)
    return true_logit * 3.0, y  # model logits 3x too sharp


def test_temperature_scaling_binary_and_multiclass():
    z, y = _overconfident()
    ts = TemperatureScaler().fit(z, y)
    assert ts.temperature == pytest.approx(3.0, rel=0.2)
    assert expected_calibration_error(ts(z), y) < expected_calibration_error(sigmoid(z), y)
    rng = np.random.default_rng(1)
    zt = rng.normal(0, 1.0, (1500, 4))
    p = np.exp(zt) / np.exp(zt).sum(1, keepdims=True)
    labels = np.array([rng.choice(4, p=pi) for pi in p])
    tm = TemperatureScaler().fit(zt * 2.5, labels)
    assert tm.temperature == pytest.approx(2.5, rel=0.25)
    assert tm(zt).shape == (1500, 4)


def test_platt_and_binning():
    z, y = _overconfident(seed=2)
    p = sigmoid(z)
    pl = PlattScaler().fit(p, y)
    assert pl.a == pytest.approx(1 / 3, rel=0.25)
    assert brier_score(pl(p), y) < brier_score(p, y)
    hb = HistogramBinningCalibrator(10).fit(p, y)
    assert np.all(np.diff(hb.values) >= 0)
    assert expected_calibration_error(hb(p), y) < expected_calibration_error(p, y)
    np.testing.assert_allclose(logit(sigmoid(np.array([-2.0, 0.0, 3.0]))), [-2, 0, 3], atol=1e-6)


def test_confidence_scorer():
    sc = ConfidenceScorer()
    assert sc.score(0.8).value == pytest.approx(0.8)
    healthy = SensorHealth(drop_rate=0.0, latency=0.02)
    degraded = SensorHealth(drop_rate=0.15, latency=0.2, age=0.3)
    a, b = sc.score(0.8, healthy, 1.0), sc.score(0.8, degraded, 0.5)
    assert b.value < a.value and set(b.factors) == {"calibrated", "health", "visibility"}
    assert sc.score(0.9, SensorHealth(fault=True)).value == 0.0
    assert sc.health_score(SensorHealth(calibration_ok=False)) == pytest.approx(0.5)
    g = ConfidenceScorer(ConfidenceConfig(method="geometric")).score(0.8, healthy, 0.5)
    assert 0 < g.value < 0.8
    many = ConfidenceScorer(calibrator=PlattScaler(a=1.0, b=0.0)).score_many([0.2, 0.9], healthy, 0.9)
    np.testing.assert_allclose(many, np.array([0.2, 0.9]) * 0.9, atol=1e-6)
