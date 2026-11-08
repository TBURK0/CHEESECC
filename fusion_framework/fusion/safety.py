"""Safety math and the host-side safety monitor.  (Step 7)

The first line of defense belongs on the microcontroller, not here:
  [ ] hardware kill switch that cuts throttle regardless of software
  [ ] MCU watchdog: neutral throttle if no host command for host_watchdog_s
  [ ] MCU speed cap
  [ ] MCU emergency stop from the front ultrasonic, using the formula below
This module is the second line: the same rule with richer inputs (camera,
curvature). It must never be the only thing that can stop the car.

The two formulas are implemented because they are plain physics; feed them
MEASURED numbers (latency_s and max_decel from step 7 braking tests).
"""
from __future__ import annotations

import math


def stopping_distance(v, latency_s, decel):
    """Distance covered during the reaction delay plus braking to a stop."""
    v = max(v, 0.0)
    return v * latency_s + v * v / (2.0 * decel)


def max_safe_speed(clear_m, latency_s, decel, margin_m=0.0):
    """Highest speed that can still stop within clear_m - margin_m."""
    d = clear_m - margin_m
    if d <= 0:
        return 0.0
    a, t = decel, latency_s
    return -a * t + math.sqrt((a * t) ** 2 + 2 * a * d)


class SafetyMonitor:
    """TODO (step 7): host-side check run at the control rate.

    evaluate(t, v, clear_distance) -> (speed_limit, estop, reason)
      - estop if clear_distance < stopping_distance(v) + margin
      - estop if any front sensor has not reported for ~2 cycles
      - otherwise speed_limit = min(max_speed, max_safe_speed(clear_distance))
    Test it on replayed logs from step 7 approach runs before using it live.
    """
