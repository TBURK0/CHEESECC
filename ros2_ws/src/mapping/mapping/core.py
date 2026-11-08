"""Conservative local obstacle clearance; unknown coverage never means free."""
import math


def clearance(depth_m, ranges, stamps, now, max_age=0.3, stop_distance=0.65):
    if len(ranges) != 4 or len(stamps) != 5:
        return False, 'MISSING_SENSOR'
    if any(not math.isfinite(t) or not 0 <= now-t <= max_age for t in stamps):
        return False, 'STALE_SENSOR'
    values = [depth_m] + list(ranges)
    if any(not math.isfinite(v) or not 0.02 <= v <= 8.0 for v in values):
        return False, 'INVALID_RANGE'
    distance = min(values)
    return distance > stop_distance, 'CLEAR' if distance > stop_distance else 'OBSTACLE'
