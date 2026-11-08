"""Deterministic software scenarios; never opens a vehicle transport."""
import json
from .core import CommandInbox, LocalMap, Point, safe_command, scene_summary


def run_scenarios():
    wall = [Point(1.5, i * 0.05) for i in range(-30, 31)]
    grid = LocalMap()
    grid.update(wall)
    path = grid.plan(Point(0.8, 0))
    moving = safe_command(0.2, 0, 0.2, wall, 1.0, 1.0, True, True)
    close = safe_command(0.2, 0, 0.8, [Point(0.4, 0)], 1.0, 1.0, True, True)
    stale = safe_command(0.2, 0, 0.2, wall, 2.0, 1.0, True, True)
    inbox = CommandInbox()
    line = json.dumps(dict(seq=1, stamp=1.0, throttle=0.2, steering=0.0))
    inbox.receive(line, 1.0)
    try:
        inbox.receive(line, 1.1)
        replay_rejected = False
    except ValueError:
        replay_rejected = True
    checks = {'observed_route': bool(path),
              'wall_blocks_route': not grid.plan(Point(2.0, 0)),
              'fresh_command': moving['reason'] == 'ok',
              'obstacle_stop': close['brake'] == 1.0,
              'sensor_dropout_stop': stale['brake'] == 1.0,
              'replay_rejected': replay_rejected}
    return {'checks': checks, 'passed': all(checks.values()),
            'scene': scene_summary(wall), 'path_points': len(path)}


def main():
    result = run_scenarios()
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
