import math
import random

import pytest

from sensor_bridge.axis_map import apply, parse_axis_map
from sensor_bridge.conversions import accel_si, gyro_si, us_range_m
from sensor_bridge.imu_calibration import GyroBiasCalibrator


def test_us_range_rep117():
    assert us_range_m('OK', 1234) == pytest.approx(1.234)
    assert us_range_m('TOO_CLOSE', 5) == -math.inf
    assert us_range_m('TOO_FAR', 5000) == math.inf
    assert us_range_m('TIMEOUT', None) == math.inf
    for s in ('NO_ECHO', 'STUCK_HIGH', 'UNASSIGNED'):
        assert math.isnan(us_range_m(s, None))


def test_imu_units():
    assert accel_si((9810, 0, -1000)) == pytest.approx((9.81, 0.0, -1.0))
    assert gyro_si((1_000_000, -500, 0)) == pytest.approx((1.0, -0.0005, 0.0))
    assert accel_si(None) is None


def test_axis_map_identity_and_rotation():
    assert apply(parse_axis_map([1, 0, 0, 0, 1, 0, 0, 0, 1]), (1, 2, 3)) == (1, 2, 3)
    # chip y -> forward, chip -x -> left (90 deg yaw): det +1
    m = parse_axis_map([0, 1, 0, -1, 0, 0, 0, 0, 1])
    assert apply(m, (1, 2, 3)) == (2, -1, 3)


@pytest.mark.parametrize('bad', [
    [1, 0, 0, 0, 1, 0, 0, 0, -1],   # reflection
    [1, 1, 0, 0, 1, 0, 0, 0, 1],    # two nonzeros in a row
    [2, 0, 0, 0, 1, 0, 0, 0, 1],    # not unit
    [1, 0, 0, 1, 0, 0, 0, 0, 1],    # repeated column
    [1, 0, 0, 0, 1, 0, 0, 0],       # wrong length
])
def test_axis_map_rejects_invalid(bad):
    with pytest.raises(ValueError):
        parse_axis_map(bad)


def test_gyro_bias_stationary():
    rng = random.Random(1)
    c = GyroBiasCalibrator(samples=100, max_std=0.02)
    for _ in range(100):
        c.add((0.01 + rng.gauss(0, 0.002), -0.02 + rng.gauss(0, 0.002), 0.005))
    assert c.calibrated
    assert c.bias == pytest.approx((0.01, -0.02, 0.005), abs=0.002)
    assert c.correct((0.01, -0.02, 0.005)) == pytest.approx((0, 0, 0), abs=0.002)


def test_gyro_bias_rejects_motion_then_retries():
    c = GyroBiasCalibrator(samples=50, max_std=0.02)
    for i in range(50):
        c.add((0.5 * (i % 2), 0.0, 0.0))   # vehicle rocking
    assert not c.calibrated and 'moving' in c.message
    for _ in range(50):
        c.add((0.0, 0.0, 0.0))
    assert c.calibrated
    c.restart()
    assert not c.calibrated
