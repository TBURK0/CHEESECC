"""Launch actual ROS processes, arm simulation, assert observation/stop behavior."""
import argparse
import json
import math
from pathlib import Path
import signal
import subprocess
import time
import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from std_srvs.srv import SetBool
from vehicle_interfaces.msg import SafetyStatus, MotionProposal


def run(scenario, enabled, output):
    output.mkdir(parents=True, exist_ok=True)
    with (output/(scenario+('_armed' if enabled else '_default')+'.log')).open('w') as log:
        proc = subprocess.Popen(['ros2', 'launch', 'planning', 'autonomy.launch.py',
            'scenario:='+scenario, 'autonomy_enabled:='+str(enabled).lower()], stdout=log, stderr=log,
            start_new_session=True)
        rclpy.init()
        n = Node('autonomy_acceptance_probe')
        speeds, statuses, poses, behaviors, latencies = [], [], [], [], []
        n.create_subscription(TwistStamped, '/simulation/desired_motion', lambda m: speeds.append((time.monotonic(), m.twist.linear.x)), 10)
        def status(m):
            statuses.append(m.reason)
            latencies.append(m.latency_ms)
        n.create_subscription(SafetyStatus, '/safety/status', status, 10)
        n.create_subscription(Odometry, '/localization/odom', lambda m: poses.append((m.pose.pose.position.x, m.pose.pose.position.y, m.header.stamp.sec+m.header.stamp.nanosec*1e-9)), 10)
        n.create_subscription(MotionProposal, '/planning/proposal', lambda m: behaviors.append(m.behavior_state), 10)
        arm = n.create_client(SetBool, '/safety/arm_simulation')
        started, future = time.monotonic(), None
        try:
            # Wait for node import/discovery separately from scenario observation.
            deadline = started+45
            ready_at = None
            while time.monotonic() < deadline:
                rclpy.spin_once(n, timeout_sec=0.05)
                if speeds and poses and ready_at is None:
                    ready_at = time.monotonic()
                    deadline = ready_at+8
                if enabled and future is None and arm.service_is_ready():
                    req = SetBool.Request()
                    req.data = True
                    future = arm.call_async(req)
            assert len(speeds) > 20, 'missing safety output'
            assert len(poses) > 20, 'missing EKF output'
            assert all(math.isfinite(v) for pose in poses for v in pose), 'invalid EKF pose'
            assert all(b[2] >= a[2] for a, b in zip(poses, poses[1:])), 'reordered EKF time'
            assert max(abs(p[1]) for p in poses) < 0.1, 'unstable straight-course lateral pose'
            if not enabled:
                assert all(v == 0 for _, v in speeds), 'default configuration moved'
            else:
                assert future is not None and future.done() and future.result().success, 'arming failed'
                if scenario == 'clear':
                    assert any(v > 0 for _, v in speeds), 'healthy simulation did not move: '+repr((set(statuses), set(behaviors)))
                elif scenario in ('obstacle', 'stale_sensor', 'lost_localization', 'perception_failure'):
                    tail = [v for t, v in speeds if ready_at and t > ready_at+6]
                    assert tail and all(v == 0 for v in tail), 'failure did not stop'
                elif scenario == 'stop_sign':
                    assert 'STOP_DWELL' in behaviors, 'stop dwell absent'
                elif scenario == 'traffic_light':
                    assert 'SIGNAL_STOP' in behaviors, 'signal stop absent'
                    assert any(v > 0 for _, v in speeds), 'green did not release'
            result = dict(scenario=scenario, enabled=enabled, command_samples=len(speeds),
                pose_samples=len(poses), max_speed_mps=max(v for _, v in speeds),
                final_pose_m=list(poses[-1][:2]), reasons=sorted(set(statuses)),
                proposal_age_p95_ms=sorted(latencies)[int(0.95*(len(latencies)-1))] if latencies else None,
                behaviors=sorted(set(behaviors)), passed=True)
            print(json.dumps(result), flush=True)
            return result
        finally:
            n.destroy_node()
            rclpy.shutdown()
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='/tmp/autonomy-smoke')
    parser.add_argument('--scenario', default='all')
    args = parser.parse_args()
    out = Path(args.output)
    cases = [('clear', False)]+[(s, True) for s in ('clear', 'obstacle', 'stale_sensor', 'lost_localization', 'perception_failure', 'stop_sign', 'traffic_light')]
    results = [run(s, enabled, out) for s, enabled in cases if args.scenario in ('all', s)]
    (out/'summary.json').write_text(json.dumps(results, indent=2))
