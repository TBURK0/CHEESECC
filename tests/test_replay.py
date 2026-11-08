from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'simulation'))
from replay import replay


class ReplayTests(unittest.TestCase):
    def test_repeatable_arrival(self):
        a = replay('clear')
        self.assertEqual(a, replay('clear'))
        self.assertGreater(a['final_x_m'], 3.85)

    def test_failures_stop(self):
        for scenario in ('obstacle', 'stale_sensor', 'lost_localization'):
            result = replay(scenario)
            self.assertTrue(all(r['speed_mps'] == 0 for r in result['rows'] if r['t_s'] >= 2.35))

    def test_traffic_unknown_stops(self):
        rows = replay('traffic_light')['rows']
        self.assertTrue(all(r['speed_mps'] == 0 for r in rows if r['t_s'] < 4))
        self.assertTrue(any(r['speed_mps'] > 0 for r in rows if r['t_s'] >= 4))

    def test_stop_sign_dwell(self):
        rows = replay('stop_sign')['rows']
        self.assertTrue(all(r['speed_mps'] == 0 for r in rows if 2 <= r['t_s'] < 4))
        self.assertTrue(any(r['speed_mps'] > 0 for r in rows if r['t_s'] > 4.1))
