"""Fail-closed motion gate. All times are seconds in a single ROS clock domain."""
from dataclasses import dataclass
import math


@dataclass
class Evidence:
    stamp: float
    healthy: bool


class Supervisor:
    required = ('wheel', 'imu', 'localization', 'perception', 'depth', 'ultrasonic_0',
                'ultrasonic_1', 'ultrasonic_2', 'ultrasonic_3')

    def __init__(self, simulation=True, enabled=False, timeout=0.3, max_speed=0.3):
        if not 0 < timeout <= 1 or not 0 < max_speed <= 0.5:
            raise ValueError('invalid safety limits')
        self.simulation, self.enabled = simulation, enabled
        self.timeout, self.max_speed = timeout, max_speed
        self.armed = False
        self.evidence = {}
        self.highwater = {}
        self.last_now = None

    def observe(self, name, stamp, healthy, now):
        if name not in self.required:
            raise ValueError('unknown sensor')
        valid = math.isfinite(stamp) and 0 <= now-stamp <= self.timeout
        if stamp <= self.highwater.get(name, float('-inf')):
            valid = False
        if valid:
            self.highwater[name] = stamp
        # Bad or reordered input invalidates the source immediately.
        self.evidence[name] = Evidence(stamp if valid else float('-inf'), bool(healthy and valid))

    def decide(self, now, speed, yaw_rate, proposal_stamp, behavior_allows, obstacle_clear):
        reason = 'OK'
        if self.last_now is not None and now < self.last_now:
            self.evidence.clear()
            self.highwater.clear()
            self.armed = False
        self.last_now = now
        if not self.simulation:
            reason = 'PHYSICAL_MOTION_UNSUPPORTED'
        elif not self.enabled or not self.armed:
            reason = 'DISARMED'
        elif not all(math.isfinite(v) for v in (now, speed, yaw_rate, proposal_stamp)):
            reason = 'INVALID_COMMAND'
        elif not 0 <= now-proposal_stamp <= self.timeout:
            reason = 'STALE_COMMAND'
        elif any(name not in self.evidence or not self.evidence[name].healthy or
                 not 0 <= now-self.evidence[name].stamp <= self.timeout for name in self.required):
            missing = [name for name in self.required if name not in self.evidence or
                not self.evidence[name].healthy or not 0 <= now-self.evidence[name].stamp <= self.timeout]
            reason = 'SENSOR_UNAVAILABLE:' + ','.join(missing)
        elif not obstacle_clear:
            reason = 'OBSTACLE_OR_UNKNOWN'
        elif not behavior_allows:
            reason = 'BEHAVIOR_STOP'
        return (max(0, min(self.max_speed, speed)), max(-0.8, min(0.8, yaw_rate)), reason) if reason == 'OK' else (0.0, 0.0, reason)
