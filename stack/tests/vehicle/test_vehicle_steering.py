import math

import pytest

from cheesestack.vehicle import (SteeringConfig, SteeringEstimator, VehicleParams, curvature_to_handwheel_deg,
                                 curvature_to_road_wheel, road_wheel_to_curvature)

P = VehicleParams(steering_ratio=12.0, wheelbase=2.7, max_steer_wheel_deg=480)


def test_ratio_conversion_and_curvature_roundtrip():
    est = SteeringEstimator(P, SteeringConfig(backlash_rad=0.0, deadband_rad=0.0))
    s = est.update(0.0, math.radians(120))
    assert s.road_wheel == pytest.approx(math.radians(10))
    assert road_wheel_to_curvature(curvature_to_road_wheel(0.05, P), P) == pytest.approx(0.05)
    assert curvature_to_handwheel_deg(10.0, P) == pytest.approx(480)  # clamped


def test_backlash_hysteresis():
    b = math.radians(4)
    est = SteeringEstimator(P, SteeringConfig(backlash_rad=b, deadband_rad=0.0))
    est.update(0.0, 0.0)
    assert est.update(0.1, math.radians(1.5)).handwheel_effective == pytest.approx(0.0)  # within play
    s = est.update(0.2, math.radians(10))
    assert s.handwheel_effective == pytest.approx(math.radians(8))
    # reversing: output holds until play is consumed
    assert est.update(0.3, math.radians(7)).handwheel_effective == pytest.approx(math.radians(8))
    assert est.update(0.4, math.radians(2)).handwheel_effective == pytest.approx(math.radians(4))


def test_deadband_and_rate():
    est = SteeringEstimator(P, SteeringConfig(backlash_rad=0.0, deadband_rad=math.radians(1), rate_tau_s=0.0))
    assert est.update(0.0, math.radians(0.5)).road_wheel == 0.0
    rate = math.radians(100)
    s = None
    for i in range(1, 50):
        s = est.update(i * 0.01, math.radians(5) + rate * i * 0.01)
    assert s.handwheel_rate == pytest.approx(rate, rel=1e-6)
    assert s.road_wheel_rate == pytest.approx(rate / 12.0, rel=1e-6)


def test_saturation_flag():
    est = SteeringEstimator(P, SteeringConfig(backlash_rad=0, deadband_rad=0))
    s = est.update(0.0, math.radians(600))
    assert s.saturated and s.handwheel_effective == pytest.approx(math.radians(480))
