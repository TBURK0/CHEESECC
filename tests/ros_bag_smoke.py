"""Record synthetic observations and verify EKF pose during clocked bag replay."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import argparse
import tempfile
import shutil
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


def stop(proc):
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def check(output):
    output.mkdir(parents=True, exist_ok=True)
    if (output/'synthetic-bag').exists():
        raise ValueError('choose a fresh output directory')
    bag = Path(tempfile.mkdtemp(prefix='cheesecc-bag-'))/'synthetic-bag'
    log = (output/'bag-smoke.log').open('w')
    processes = []
    def start(args):
        p = subprocess.Popen(args, stdout=log, stderr=log, start_new_session=True)
        processes.append(p)
        return p
    rclpy.init()
    node = Node('bag_acceptance_probe')
    poses = []
    node.create_subscription(Odometry, '/localization/odom', lambda m: poses.append(m), 10)
    def spin(seconds):
        until = time.monotonic()+seconds
        while time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=0.05)
    try:
        launch = start(['ros2', 'launch', 'planning', 'autonomy.launch.py'])
        spin(5)
        assert poses, 'live EKF unavailable'
        topics = ['/sensors/wheel/odom', '/sensors/imu', '/phone/gps', '/perception/scene',
            '/camera/camera/aligned_depth_to_color/image_raw']+['/sensors/ultrasonic/sensor_{}'.format(i) for i in range(4)]
        record = start(['ros2', 'bag', 'record', '-o', str(bag)]+topics)
        spin(8)
        stop(record)
        stop(launch)
        assert (bag/'metadata.yaml').exists(), 'bag not written'
        poses.clear()
        launch = start(['ros2', 'launch', 'planning', 'autonomy.launch.py',
            'simulation:=false', 'use_sim_time:=true', 'synthetic_mounts:=true'])
        spin(4)
        poses.clear()
        play = start(['ros2', 'bag', 'play', str(bag), '--clock', '--delay', '2'])
        spin(16)
        assert len(poses) > 20, 'missing replay EKF samples'
        stamps = [m.header.stamp.sec+m.header.stamp.nanosec*1e-9 for m in poses]
        assert all(b >= a for a, b in zip(stamps, stamps[1:])), 'replay timestamp regression'
        assert all(math.hypot(m.pose.pose.position.x, m.pose.pose.position.y) < .05 for m in poses), 'stationary replay drift'
        result = dict(passed=True, pose_samples=len(poses), timestamp_span_s=stamps[-1]-stamps[0],
            max_stationary_drift_m=max(math.hypot(m.pose.pose.position.x, m.pose.pose.position.y) for m in poses))
        (output/'summary.json').write_text(json.dumps(result, indent=2))
        shutil.copytree(bag, output/'synthetic-bag')
        print(json.dumps(result))
    finally:
        for p in reversed(processes):
            stop(p)
        node.destroy_node()
        rclpy.try_shutdown()
        log.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    check(Path(p.parse_args().output))
