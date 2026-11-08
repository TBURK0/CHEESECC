"""Closed-loop bicycle / longitudinal dynamics with ideal known-map sensing.

This deterministic smoke simulator uses assumed physics, not measured vehicle data.
It never opens hardware. Unknown-space and sensor failure have separate scenarios.
"""
import json
import math

from .core import LocalMap, Point, safe_command
from .motion import VehicleGeometry, advance, lattice_plan, pursuit, swept_clear
from .spatial import Pose2


def drive(dropout_at=None, obstacle_at=None):
    geometry, grid = VehicleGeometry(), LocalMap(width=100, height=100)
    grid.data = [0] * len(grid.data)
    goal = Point(1.6, 0.4)
    route = lattice_plan(grid, goal, geometry)
    if not route:
        raise RuntimeError('simulation reference route unavailable')
    pose, speed, time, progress = Pose2(), 0.0, 0.0, 0
    last_observation, stopped_on_dropout, collision = 0.0, False, False
    history = []
    obstacle, obstacle_stopped = None, False
    for _ in range(300):
        time += 0.05
        if dropout_at is None or time < dropout_at:
            last_observation = time
        if obstacle_at is not None and time >= obstacle_at and obstacle is None:
            obstacle = pose.world(Point(0.8, 0))
            c = grid.cell(obstacle)
            grid.data[c[1] * grid.width + c[0]] = 100
        # Restrict search to forward progress to avoid jumping backwards on a curve.
        progress = min(range(progress, len(route)),
                       key=lambda i: math.hypot(route[i].x - pose.x, route[i].y - pose.y))
        local_route = [Pose2()]
        for p in route[progress + 1:]:
            q = pose.local(Point(p.x, p.y))
            local_route.append(Pose2(q.x, q.y, p.yaw - pose.yaw))
        throttle, brake, steer = pursuit(local_route, geometry)
        reached = math.hypot(pose.x - goal.x, pose.y - goal.y) < 0.16
        points = [pose.local(obstacle)] if obstacle is not None else []
        result = safe_command(throttle, steer, speed, points, time, last_observation, True, True)
        stopping_distance = 0.25 + abs(speed)*0.3 + speed*speed/(2*0.8)
        if not swept_clear(grid, pose, stopping_distance, steer * geometry.max_steer, geometry):
            result.update(throttle=0.0, brake=1.0, reason='swept_collision')
        if obstacle is not None and result['brake']:
            obstacle_stopped = True
        if reached or brake:
            result.update(throttle=0.0, brake=1.0)
        if result['reason'] == 'stale_scene':
            stopped_on_dropout = True
        # Assumed acceleration, rolling drag, and service-brake deceleration.
        acceleration = 2.0 * result['throttle'] - 0.15 * speed - 0.8 * result['brake']
        next_speed = min(0.6, max(0.0, speed + acceleration * 0.05))
        distance = (speed + next_speed) * 0.5 * 0.05
        steering = result['steering'] * geometry.max_steer
        if not swept_clear(grid, pose, distance, steering, geometry):
            collision = True
            break
        pose, speed = advance(pose, distance, steering, geometry), next_speed
        history.append(dict(time=time, x=pose.x, y=pose.y, speed=speed, reason=result['reason']))
        if speed == 0 and (reached or stopped_on_dropout or obstacle_stopped):
            break
    return dict(reached=math.hypot(pose.x-goal.x, pose.y-goal.y) < 0.2,
                collision=collision, stopped_on_dropout=stopped_on_dropout,
                obstacle_stopped=obstacle_stopped,
                final_speed=speed, samples=len(history), history=history)


def run():
    normal, dropout, obstacle = drive(), drive(dropout_at=1.0), drive(obstacle_at=1.0)
    checks = dict(goal_reached=normal['reached'], no_collision=not normal['collision'],
                  dropout_brakes=dropout['stopped_on_dropout'] and dropout['final_speed'] == 0,
                  dropout_before_goal=not dropout['reached'],
                  moving_obstacle_stop=obstacle['obstacle_stopped'] and obstacle['final_speed'] == 0 and not obstacle['collision'])
    return dict(passed=all(checks.values()), checks=checks, normal=normal, dropout=dropout, obstacle=obstacle)


def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='optional trajectory JSON, must not already exist')
    args = parser.parse_args()
    result = run()
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps(dict(passed=result['passed'], checks=result['checks']), indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
