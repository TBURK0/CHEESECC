import numpy as np
import pytest

from cheesestack.planning import LaneBoundaries, LaneCentering, LaneCenteringConfig
from cheesestack.planning.lane_centering import centerline_from_boundaries


def test_centerline_both_and_single():
    cfg = LaneCenteringConfig()
    c, w, r = centerline_from_boundaries(LaneBoundaries(np.array([1.9, 0, 0.001]), np.array([-1.7, 0, 0.001])), cfg)
    assert w == pytest.approx(3.6) and c[0] == pytest.approx(0.1) and r == "both boundaries"
    c, w, r = centerline_from_boundaries(LaneBoundaries(left=np.array([2.0, 0.0])), cfg)
    assert c[0] == pytest.approx(0.2) and r == "left only"
    c, _, r = centerline_from_boundaries(LaneBoundaries(np.array([1.8]), np.array([-1.8]), right_confidence=0.1), cfg)
    assert r == "left only"
    c, _, r = centerline_from_boundaries(LaneBoundaries(np.array([9.0]), np.array([-1.8]), left_confidence=0.5), cfg)
    assert r == "right only"  # implausible width -> trust more confident side
    assert centerline_from_boundaries(LaneBoundaries(), cfg)[0] is None


def test_target_hysteresis_and_rate_limit():
    lc = LaneCentering(LaneCenteringConfig(hysteresis_m=0.1, max_rate_mps=0.5))
    b = LaneBoundaries(np.array([1.8]), np.array([-1.8]))
    out = lc.update(b, 10.0, 0.1)
    assert out.valid and out.target_offset == pytest.approx(0.0)
    # small jitter is ignored
    out = lc.update(LaneBoundaries(np.array([1.85]), np.array([-1.75])), 10.0, 0.1)
    assert out.target_offset == pytest.approx(0.0)
    # large shift is followed at the rate limit
    shifted = LaneBoundaries(np.array([2.8]), np.array([-0.8]))
    out = lc.update(shifted, 10.0, 0.1)
    assert out.target_offset == pytest.approx(0.05)
    for _ in range(40):
        out = lc.update(shifted, 10.0, 0.1)
    assert out.target_offset == pytest.approx(1.0, abs=0.1)


def test_curvature_and_heading_at_lookahead():
    lc = LaneCentering()
    k = 0.01
    b = LaneBoundaries(np.array([1.8, 0.0, k / 2]), np.array([-1.8, 0.0, k / 2]))
    out = lc.update(b, 0.0, 0.1)  # lookahead clamps to 5 m
    assert out.lookahead_m == 5.0
    assert out.curvature == pytest.approx(k, rel=0.01)
    assert out.heading_error == pytest.approx(np.arctan(k * 5), rel=1e-6)
    invalid = lc.update(LaneBoundaries(), 10, 0.1)
    assert not invalid.valid
