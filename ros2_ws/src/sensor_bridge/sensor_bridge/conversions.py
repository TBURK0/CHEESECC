"""Wire units -> SI / REP-117. Contract: docs/serial_protocol.md."""
import math

# REP-117: -inf = too close to measure, +inf = nothing detected, NaN = invalid reading.
_US_RANGE = {
    'TOO_CLOSE': -math.inf,
    'TOO_FAR': math.inf,
    'TIMEOUT': math.inf,
    'NO_ECHO': math.nan,
    'STUCK_HIGH': math.nan,
    'UNASSIGNED': math.nan,
}


def us_range_m(status, dist_mm):
    if status == 'OK':
        return dist_mm / 1000.0
    return _US_RANGE[status]


def accel_si(accel_mm_s2):
    return None if accel_mm_s2 is None else tuple(a / 1000.0 for a in accel_mm_s2)


def gyro_si(gyro_urad_s):
    return None if gyro_urad_s is None else tuple(g / 1e6 for g in gyro_urad_s)
