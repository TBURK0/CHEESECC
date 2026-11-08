"""Turning raw sensor readings into calibrated body-frame values.

Applied inside fusion (not in the hardware code) so raw logs stay raw and you
can re-run old logs when calibration improves.
"""
from __future__ import annotations

import numpy as np


def rot_from_rpy_deg(rpy):
    """Sensor -> body rotation from mount roll/pitch/yaw in degrees (ZYX)."""
    r, p, y = np.radians(rpy)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def apply_accel_calib(raw, calib):
    """calib = {'offset': [3], 'scale': [3]}: a = (raw - offset) * scale.
    calib=None passes through unchanged."""
    raw = np.asarray(raw, float)
    if calib is None:
        return raw
    return (raw - np.asarray(calib["offset"])) * np.asarray(calib["scale"])


def to_body(vec, mount_rpy_deg):
    return rot_from_rpy_deg(mount_rpy_deg) @ np.asarray(vec, float)


def fit_six_position(readings_by_face, g=9.80665):
    """Step 1: fit per-axis offset and scale from six static poses.

    readings_by_face: dict like {'+x': mean_reading, '-x': ..., '+y': ...,
    '-y': ..., '+z': ..., '-z': ...}, where '+x' means the sensor's +x axis
    pointed straight UP and the value is the AVERAGE of a few seconds of
    still data in that pose.

    TODO (Step 1): implement. For each axis i:
        offset_i = (up_i + down_i) / 2
        scale_i  = 2 * g / (up_i - down_i)
    Then check the result: on all six faces the calibrated magnitude should
    be within about 0.5% of g. Record the residual in your test log.
    """
    raise NotImplementedError("Step 1: six-position accelerometer calibration")
