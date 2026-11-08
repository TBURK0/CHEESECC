"""Wheel-speed / calibrated gyro dead reckoning, without ROS dependencies."""
import math
from .spatial import Pose2


class DeadReckoner:
    def __init__(self):
        self.pose, self.last_stamp = Pose2(), None

    def update(self, speed, yaw_rate, stamp):
        if not all(math.isfinite(v) for v in (speed, yaw_rate, stamp)):
            raise ValueError('invalid odometry input')
        if self.last_stamp is None:
            self.last_stamp = stamp
            return self.pose
        dt = stamp - self.last_stamp
        if not 0 < dt <= 0.2:
            self.last_stamp = stamp
            raise ValueError('odometry gap or non-increasing time; pose continuity lost')
        self.last_stamp = stamp
        p = self.pose
        yaw = p.yaw + yaw_rate * dt
        if abs(yaw_rate) < 1e-9:
            x, y = p.x + speed * dt * math.cos(p.yaw), p.y + speed * dt * math.sin(p.yaw)
        else:
            x = p.x + speed / yaw_rate * (math.sin(yaw) - math.sin(p.yaw))
            y = p.y - speed / yaw_rate * (math.cos(yaw) - math.cos(p.yaw))
        self.pose = Pose2(x, y, math.atan2(math.sin(yaw), math.cos(yaw)))
        return self.pose
