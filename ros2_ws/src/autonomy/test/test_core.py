import json
import math
import unittest

from autonomy.core import CommandInbox, LocalMap, Point, depth_points, safe_command
from autonomy.simulation import run_scenarios


class CoreTests(unittest.TestCase):
    def test_depth_units_axes_and_invalid_returns(self):
        points = depth_points([[1000, 0, math.nan, 1000]], 100, 100, 1, 0, stride=1)
        self.assertEqual(points, [Point(1.0, 0.01), Point(1.0, -0.02)])
        self.assertEqual(depth_points([[1.0]], 100, 100, 0, 0, scale=1, stride=1), [Point(1, 0)])
        with self.assertRaises(ValueError):
            depth_points([[1000]], 0, 100, 0, 0)

    def test_floor_filter(self):
        self.assertEqual(depth_points([[1000]], 100, 100, 0, -20, stride=1), [])

    def test_unknown_and_inflation(self):
        grid = LocalMap()
        self.assertFalse(grid.plan(Point(1, 0)))
        grid.update([Point(1, i * 0.05) for i in range(-20, 21)])
        self.assertTrue(grid.plan(Point(0.5, 0)))
        self.assertFalse(grid.plan(Point(0.9, 0)))
        self.assertFalse(grid.plan(Point(2, 0)))
        grid.update([])
        self.assertTrue(all(v == -1 for v in grid.data))

    def test_occupied_wins_over_free_ray(self):
        grid = LocalMap()
        grid.update([Point(1, 0), Point(2, 0)])
        x, y = grid.cell(Point(1, 0))
        self.assertEqual(grid.data[y * grid.width + x], 100)

    def test_safety_faults(self):
        for kwargs in ({'now': 2}, {'observation_time': 2}, {'link_ok': False},
                       {'armed': False}, {'estop': True}, {'speed': math.nan},
                       {'points': [Point(math.inf, 0)]}, {'points': [Point(0.2, 0)]}):
            args = dict(throttle=0.2, steering=0, speed=0.2, points=[Point(2, 0)],
                        now=1, observation_time=1, link_ok=True, armed=True)
            args.update(kwargs)
            self.assertEqual(safe_command(**args)['brake'], 1)

    def test_comms_validation_does_not_consume_sequence(self):
        inbox = CommandInbox()
        valid = dict(seq=1, stamp=1.0, throttle=0.2, steering=0.0)
        for fields in ({'seq': True}, {'stamp': 2}, {'throttle': math.nan},
                       {'throttle': 1}, {'steering': 'left'}, {'arm': True}):
            msg = dict(valid, **fields)
            with self.assertRaises(ValueError):
                inbox.receive(json.dumps(msg), 1)
        self.assertEqual(inbox.receive(json.dumps(valid), 1)['seq'], 1)
        with self.assertRaises(ValueError):
            inbox.receive(json.dumps(valid), 1)
        for line in ('[]', 'null', '1', '"abc"', '{', 'x' * 1025):
            with self.assertRaises(ValueError):
                inbox.receive(line, 1)

    def test_scenarios(self):
        self.assertTrue(run_scenarios()['passed'])


if __name__ == '__main__':
    unittest.main()
