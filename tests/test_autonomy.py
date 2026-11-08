import json
import math
from pathlib import Path
import sys
import unittest
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
for name in ('localization', 'planning', 'mapping', 'perception', 'safety_supervisor'):
    sys.path.insert(0, str(ROOT/'ros2_ws/src'/name))
from localization.phone import PhoneParser, MotionParser, local_enu
from planning.core import Behavior, astar, pursuit
from mapping.core import clearance
from perception.core import signal_state
from safety_supervisor.core import Supervisor


def healthy_gate():
    gate = Supervisor(enabled=True)
    gate.armed = True
    for name in gate.required:
        gate.observe(name, 10.0, True, 10.0)
    return gate


class SafetyTests(unittest.TestCase):
    def test_default_disabled(self):
        self.assertEqual(Supervisor().decide(10, 1, 0, 10, True, True)[2], 'DISARMED')

    def test_physical_motion_disabled(self):
        gate = healthy_gate()
        gate.simulation = False
        self.assertEqual(gate.decide(10, 0.3, 0, 10, True, True)[0], 0)

    def test_limits(self):
        self.assertEqual(healthy_gate().decide(10, 5, 5, 10, True, True)[:2], (0.3, 0.8))

    def test_each_sensor_required(self):
        for name in Supervisor.required:
            gate = healthy_gate()
            del gate.evidence[name]
            self.assertEqual(gate.decide(10, .3, 0, 10, True, True)[0], 0, name)

    def test_stale_obstacle_localization(self):
        for name in ('depth', 'ultrasonic_0', 'localization'):
            gate = healthy_gate()
            gate.observe(name, 9, True, 10)
            self.assertEqual(gate.decide(10, .3, 0, 10, True, True)[0], 0)

    def test_unknown_behavior_and_obstacle(self):
        for behavior, clear in ((False, True), (True, False)):
            self.assertEqual(healthy_gate().decide(10, .3, 0, 10, behavior, clear)[0], 0)

    def test_nonfinite_future_stale_commands(self):
        for speed, stamp in ((math.nan, 10), (.3, 11), (.3, 9)):
            self.assertEqual(healthy_gate().decide(10, speed, 0, stamp, True, True)[0], 0)

    def test_reordered_sensor_not_resurrected(self):
        gate = healthy_gate()
        gate.observe('depth', 9.9, True, 10)
        gate.observe('depth', 9.95, True, 10)
        self.assertFalse(gate.evidence['depth'].healthy)

    def test_clock_reset_disarms(self):
        gate = healthy_gate()
        gate.decide(10, .3, 0, 10, True, True)
        self.assertEqual(gate.decide(1, .3, 0, 1, True, True)[0], 0)
        self.assertFalse(gate.armed)


class BehaviorTests(unittest.TestCase):
    def test_signal_transitions(self):
        b = Behavior()
        for signal in ('RED', 'YELLOW', 'UNKNOWN', 'invalid'):
            self.assertFalse(b.update(0, True, True, signal, False, 0))
        self.assertTrue(b.update(1, True, True, 'GREEN', False, 0))

    def test_stop_dwell_requires_stopped(self):
        b = Behavior()
        self.assertFalse(b.update(0, True, True, 'NONE', True, .3))
        self.assertFalse(b.update(1, True, True, 'NONE', True, 0))
        self.assertFalse(b.update(2.9, True, True, 'NONE', True, 0))
        self.assertTrue(b.update(3, True, True, 'NONE', True, 0))

    def test_failure_restarts_dwell(self):
        b = Behavior()
        b.update(0, True, True, 'NONE', True, 0)
        b.update(1, True, False, 'NONE', True, 0)
        self.assertFalse(b.update(3, True, True, 'NONE', True, 0))

    def test_route_and_tracking(self):
        route = astar((0, 0), (4, 0), [(2, 0)])
        self.assertNotIn((2, 0), route)
        self.assertEqual(pursuit(4, 0, 0, [(0, 0), (4, 0)]), (0, 0))
        with self.assertRaises(ValueError):
            astar((0, 0), (30, 1))


class SensorTests(unittest.TestCase):
    def test_motion_frame_and_quality(self):
        obj = dict(schema_version=1, type='phone.motion', simulation=True, sequence=1,
                   timestamp_utc='2026-11-07T19:00:00Z', frame='base_link_enu',
                   orientation_xyzw=[0, 0, 0, 1], angular_velocity_radps=[0, 0, 0],
                   orientation_variance_rad2=0.01)
        now = datetime(2026, 11, 7, 19, tzinfo=timezone.utc).timestamp()
        self.assertEqual(MotionParser().parse(json.dumps(obj), now)['q'], [0, 0, 0, 1])
        for change in ({'frame': 'device'}, {'orientation_xyzw': [0, 0, 0, 2]}, {'orientation_variance_rad2': 1}):
            with self.assertRaises(ValueError):
                MotionParser().parse(json.dumps(dict(obj, **change)), now)

    def test_obstacle_and_unknown_ranges(self):
        self.assertTrue(clearance(3, [3]*4, [10]*5, 10)[0])
        for ranges in ([.3]*4, [math.nan]*4, [3]*3):
            self.assertFalse(clearance(3, ranges, [10]*5, 10)[0])
        self.assertFalse(clearance(3, [3]*4, [9]*5, 10)[0])

    def test_ambiguous_light(self):
        self.assertEqual(signal_state([('GREEN', .9), ('RED', .9)]), 'UNKNOWN')
        self.assertEqual(signal_state([('GREEN', .2)]), 'UNKNOWN')
        self.assertEqual(signal_state([('GREEN', .9)]), 'GREEN')

    def test_phone_timestamp_accuracy_replay(self):
        obj = dict(schema_version=1, type='phone.gps', simulation=True, sequence=1,
                   timestamp_utc='2026-11-07T19:00:00Z', latitude_deg=0, longitude_deg=0,
                   altitude_m=0, horizontal_accuracy_m=3, vertical_accuracy_m=5)
        now = datetime(2026, 11, 7, 19, tzinfo=timezone.utc).timestamp()
        parser = PhoneParser()
        self.assertEqual(parser.parse(json.dumps(obj), now)['covariance'], [9, 9, 25])
        with self.assertRaises(ValueError):
            parser.parse(json.dumps(obj), now)
        for change in ({'horizontal_accuracy_m': -1}, {'latitude_deg': math.nan}, {'simulation': False}):
            with self.assertRaises(ValueError):
                PhoneParser().parse(json.dumps(dict(obj, **change)), now)
        with self.assertRaises(ValueError):
            PhoneParser().parse(json.dumps(obj), now+2)

    def test_geodetic_transform(self):
        self.assertEqual(local_enu(0, 0, 0, (0, 0, 0)), (0, 0, 0))
        east, north, up = local_enu(0, 0.00001, 0, (0, 0, 0))
        self.assertAlmostEqual(east, 1.11319, places=4)
        self.assertAlmostEqual(north, 0, places=4)


if __name__ == '__main__':
    unittest.main()
