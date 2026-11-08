import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range, Image, Imu
from std_srvs.srv import SetBool
from vehicle_interfaces.msg import Scene, ObstacleStatus, MotionProposal, SafetyStatus
from .core import Supervisor


def seconds(stamp):
    return stamp.sec + stamp.nanosec*1e-9


class SafetyNode(Node):
    def __init__(self):
        super().__init__('safety_supervisor')
        self.gate = Supervisor(self.declare_parameter('simulation', True).value,
                               self.declare_parameter('autonomy_enabled', False).value)
        self.proposal, self.obstacle = None, None
        self.pub = self.create_publisher(TwistStamped, '/simulation/desired_motion', 10)
        self.status = self.create_publisher(SafetyStatus, '/safety/status', 10)
        self.create_service(SetBool, '/safety/arm_simulation', self.arm)
        self.create_subscription(MotionProposal, '/planning/proposal', self.set_proposal, 10)
        self.create_subscription(ObstacleStatus, '/mapping/obstacles', self.set_obstacle, 10)
        self.create_subscription(Odometry, '/localization/odom', self.pose, 10)
        self.create_subscription(Odometry, '/sensors/wheel/odom', lambda m: self.observe('wheel', m.header.stamp, math.isfinite(m.twist.twist.linear.x)), 10)
        self.create_subscription(Imu, '/sensors/imu', lambda m: self.observe('imu', m.header.stamp, math.isfinite(m.angular_velocity.z)), qos_profile_sensor_data)
        self.create_subscription(Scene, '/perception/scene', lambda m: self.observe('perception', m.header.stamp, m.healthy and m.path_confidence >= 0.5), 10)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', lambda m: self.observe('depth', m.header.stamp, m.width > 0 and m.height > 0), qos_profile_sensor_data)
        for i in range(4):
            self.create_subscription(Range, '/sensors/ultrasonic/sensor_{}'.format(i),
                lambda m, index=i: self.observe('ultrasonic_{}'.format(index), m.header.stamp,
                    math.isfinite(m.range) and m.min_range <= m.range <= m.max_range), qos_profile_sensor_data)
        self.create_timer(0.05, self.tick)

    def now(self):
        return self.get_clock().now().nanoseconds*1e-9

    def observe(self, name, stamp, healthy):
        self.gate.observe(name, seconds(stamp), healthy, self.now())

    def pose(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        values = [p.x, p.y, q.x, q.y, q.z, q.w, m.pose.covariance[0], m.pose.covariance[7]]
        healthy = all(math.isfinite(v) for v in values) and 0 <= values[-2] < 0.25 and 0 <= values[-1] < 0.25 and abs(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w-1) < 0.01
        self.observe('localization', m.header.stamp, healthy)

    def set_proposal(self, m):
        self.proposal = m

    def set_obstacle(self, m):
        self.obstacle = m

    def arm(self, req, response):
        self.gate.armed = bool(req.data and self.gate.simulation and self.gate.enabled)
        response.success = self.gate.armed == req.data
        response.message = 'simulation only; autonomy_enabled must also be true'
        return response

    def tick(self):
        now = self.now()
        p, o = self.proposal, self.obstacle
        clear = bool(o and o.clear and 0 <= now-seconds(o.header.stamp) <= 0.3
                     and all(0 <= now-seconds(t) <= 0.3 for t in o.source_stamps))
        v, w, reason = self.gate.decide(now, p.speed_mps if p else 0, p.yaw_rate_radps if p else 0,
            seconds(p.header.stamp) if p else float('-inf'), bool(p and p.behavior_allows), clear)
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x, msg.twist.angular.z = float(v), float(w)
        self.pub.publish(msg)
        status = SafetyStatus()
        status.header = msg.header
        status.simulation, status.stop_requested, status.reason = self.gate.simulation, reason != 'OK', reason
        status.latency_ms = float(max(0, now-seconds(p.header.stamp))*1000) if p else 0.0
        self.status.publish(status)


def main():
    rclpy.init()
    node = SafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
