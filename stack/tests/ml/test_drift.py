import numpy as np

from cheesestack.core import Severity
from cheesestack.ml.drift import DriftDetector, embedding_mean_shift, kolmogorov_sf, ks_2samp, psi


def test_psi_behaviour():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5000)
    assert psi(ref, rng.normal(size=5000)) < 0.02
    assert psi(ref, rng.normal(1.0, 1.0, 5000)) > 0.25
    assert psi(np.ones(10), np.ones(10)) < 1e-6


def test_ks_known_values():
    d, p = ks_2samp([1, 2, 3], [1, 2, 3])
    assert d == 0.0 and p == 1.0
    d, _ = ks_2samp([0, 1, 2, 3], [10, 11, 12])
    assert d == 1.0
    rng = np.random.default_rng(1)
    _, p_same = ks_2samp(rng.normal(size=500), rng.normal(size=500))
    _, p_diff = ks_2samp(rng.normal(size=500), rng.normal(0.5, 1, 500))
    assert p_same > 0.01 and p_diff < 1e-6
    assert np.isclose(kolmogorov_sf(1.36), 0.049, atol=2e-3)  # classic 5% critical value


def test_embedding_shift():
    rng = np.random.default_rng(2)
    ref = rng.normal(size=(1000, 8)) + 1
    same = embedding_mean_shift(ref, rng.normal(size=(500, 8)) + 1)
    moved = embedding_mean_shift(ref, rng.normal(size=(500, 8)) + 2)
    assert same.normalized_shift < 0.2 and moved.normalized_shift > 2.0


def test_detector_per_feature_alerts():
    rng = np.random.default_rng(3)
    ref = rng.normal(size=(2000, 3))
    det = DriftDetector().fit(ref, ["brightness", "speed", "blur"], embeddings=ref)
    cur = rng.normal(size=(1000, 3))
    cur[:, 0] += 1.0  # brightness drift (e.g. night)
    rep = det.check(cur, stamp=12.0, embeddings=cur)
    assert rep.drifted == ["brightness"]
    sev = {f.name: f.severity for f in rep.features}
    assert sev["brightness"] == Severity.DEGRADED and sev["speed"] == Severity.INFO
    codes = [f.code for f in rep.faults]
    assert "ML_DRIFT_BRIGHTNESS" in codes and all(f.stamp == 12.0 for f in rep.faults)
