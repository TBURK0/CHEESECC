"""Dead reckoning from measured wheel velocity and calibrated IMU yaw rate.

    wheel/velocity must be measured geometry_msgs/TwistStamped in base_link.
    No estimated speed is fabricated from throttle commands.
"""
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster

from .odometry import DeadReckoner


def seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class OdometryNode(Node):
    def __init__(self):
        super().__init__('wheel_imu_odometry')
        self.declare_parameter('imu_axes_verified', False)
        self.integrator = DeadReckoner()
        self.imu = None
        self.failed = False
        self.pub = self.create_publisher(Odometry, 'odom', 10)
        self.broadcaster = TransformBroadcaster(self)
        self.create_subscription(Imu, 'imu/mpu6050', self.on_imu, 10)
        self.create_subscription(TwistStamped, 'wheel/velocity', self.on_velocity, 10)

    def on_imu(self, msg):
        self.imu = msg

    def on_velocity(self, msg):
        if self.failed or not self.get_parameter('imu_axes_verified').value or self.imu is None:
            return
        t = seconds(msg.header.stamp)
        now = self.get_clock().now().nanoseconds * 1e-9
        if (msg.header.frame_id != 'base_link' or not 0 <= now - t <= 0.2 or
                abs(t - seconds(self.imu.header.stamp)) > 0.05):
            return
        try:
            p = self.integrator.update(msg.twist.linear.x, self.imu.angular_velocity.z, t)
        except ValueError as exc:
            # Never resume a discontinuous pose stream without explicit restart/relocalization.
            self.failed = True
            self.get_logger().error(str(exc))
            return
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = msg.header.stamp, 'odom', 'base_link'
        odom.pose.pose.position.x, odom.pose.pose.position.y = p.x, p.y
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = math.sin(p.yaw / 2), math.cos(p.yaw / 2)
        odom.twist.twist.linear.x, odom.twist.twist.angular.z = msg.twist.linear.x, self.imu.angular_velocity.z
        # Nonzero conservative uncertainty; this estimator has no absolute drift correction.
        for index in (0, 7, 35):
            odom.pose.covariance[index] = 0.1
            odom.twist.covariance[index] = 0.05
        self.pub.publish(odom)
        tf = TransformStamped()
        tf.header, tf.child_frame_id = odom.header, 'base_link'
        tf.transform.translation.x, tf.transform.translation.y = p.x, p.y
        tf.transform.rotation = odom.pose.pose.orientation
        self.broadcaster.sendTransform(tf)


def main():
    rclpy.init()
    node = OdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
