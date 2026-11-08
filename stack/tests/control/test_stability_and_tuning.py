import math

import numpy as np
import pytest

from cheesestack.control import GainSchedule, StabilityConfig, StabilityLimiter, VehicleParams, step_response_metrics
from cheesestack.control.tuning import simulate_second_order_step


def test_stability_limits():
    lim = StabilityLimiter(VehicleParams(), StabilityConfig(max_lateral_accel=3.0, max_yaw_rate=0.6))
    r = lim.limit(0.1, 20.0)
    assert r.curvature == pytest.approx(3.0 / 400) and r.limited_by == "lateral_accel"
    r = lim.limit(-0.1, 2.0)
    assert r.curvature == pytest.approx(-0.1)  # yaw-rate limit 0.3 > 0.1, lat 0.75 > 0.1
    r = lim.limit(1.0, 0.0)
    assert r.limited_by == "mechanical"
    assert lim.max_speed_for_curvature(0.0) == 40.0
    assert lim.max_speed_for_curvature(0.01) == pytest.approx(math.sqrt(300))


def test_gain_schedule_interpolates_and_clamps():
    gs = GainSchedule([0, 10, 20], {"kp": [1.0, 2.0, 4.0], "ki": [0.1, 0.1, 0.2]})
    assert gs(5.0) == {"kp": 1.5, "ki": pytest.approx(0.1)}
    assert gs.get("kp", 100.0) == 4.0 and gs.get("kp", -1.0) == 1.0
    with pytest.raises(ValueError):
        GainSchedule([0, 0], {"kp": [1, 2]})


def test_step_metrics_second_order():
    zeta, wn = 0.5, 2.0
    t, y = simulate_second_order_step(wn, zeta, t_end=12.0)
    m = step_response_metrics(t, y, 1.0)
    expected_os = 100 * math.exp(-zeta * math.pi / math.sqrt(1 - zeta ** 2))
    assert m.overshoot_pct == pytest.approx(expected_os, abs=0.5)
    assert m.peak_time == pytest.approx(math.pi / (wn * math.sqrt(1 - zeta ** 2)), abs=0.02)
    assert m.settling_time == pytest.approx(4 / (zeta * wn), rel=0.25)
    assert 0.5 < m.rise_time < 1.0
    assert abs(m.steady_state_error) < 1e-3


def test_step_metrics_first_order():
    t = np.linspace(0, 10, 10001)
    y = 1 - np.exp(-t)
    m = step_response_metrics(t, y, 1.0)
    assert m.rise_time == pytest.approx(math.log(9), abs=1e-3)
    assert m.overshoot_pct == 0.0
    assert m.settling_time == pytest.approx(-math.log(0.02), abs=1e-2)
