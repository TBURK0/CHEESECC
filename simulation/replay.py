"""Portable deterministic behavior/safety replay, no ROS or hardware required."""
import json
import math
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
for package in ('planning', 'safety_supervisor'):
    sys.path.insert(0, str(ROOT/'ros2_ws/src'/package))
from planning.core import Behavior, pursuit
from safety_supervisor.core import Supervisor


def replay(scenario):
    gate, behavior = Supervisor(enabled=True), Behavior()
    gate.armed = True
    x, speed, rows = 0.0, 0.0, []
    for tick in range(500):
        now = tick*.05
        for name in gate.required:
            if not (scenario == 'stale_sensor' and now >= 2 and name == 'depth'):
                gate.observe(name, now, not (scenario == 'lost_localization' and now >= 2 and name == 'localization'), now)
        signal = ('RED' if now < 3 else 'UNKNOWN' if now < 4 else 'GREEN') if scenario == 'traffic_light' else 'NONE'
        stop = scenario == 'stop_sign' and 2 <= now < 6
        allowed = behavior.update(now, not (scenario == 'lost_localization' and now >= 2),
                                  True, signal, stop, speed, x >= 3.85)
        v, w = pursuit(x, 0, 0, [(0, 0), (4, 0)])
        speed, _, reason = gate.decide(now, v, w, now, allowed, not (scenario == 'obstacle' and now >= 2))
        x += speed*.05
        rows.append(dict(t_s=round(now, 3), x_m=x, speed_mps=speed, behavior=behavior.state, reason=reason))
    return {'scenario': scenario, 'mode': 'simulation', 'final_x_m': x,
            'cross_track_rms_m': 0.0, 'rows': rows}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='simulation/results')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for scenario in ('clear', 'stop_sign', 'traffic_light', 'obstacle', 'stale_sensor', 'lost_localization'):
        result = replay(scenario)
        (out/(scenario+'.json')).write_text(json.dumps(result, indent=2))
        print(scenario, round(result['final_x_m'], 3), result['rows'][-1]['reason'])
