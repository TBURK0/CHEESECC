"""Deterministic synthetic observations; no device access."""
import math
import struct
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, NavSatFix, Image, CameraInfo, Range
from vehicle_interfaces.msg import Scene, Detection


class Simulator(Node):
    def __init__(self):
        super().__init__('synthetic_course')
        self.scenario = self.declare_parameter('scenario', 'clear').value
        self.x, self.y, self.yaw, self.speed, self.rate = 0.0, 0.0, 0.0, 0.0, 0.0
        self.elapsed, self.command_stamp = 0.0, float('-inf')
        self.odom = self.create_publisher(Odometry, '/sensors/wheel/odom', 10)
        self.imu = self.create_publisher(Imu, '/sensors/imu', 10)
        self.gps = self.create_publisher(NavSatFix, '/phone/gps', 10)
        self.scene = self.create_publisher(Scene, '/perception/scene', 10)
        self.depth = self.create_publisher(Image, '/camera/camera/aligned_depth_to_color/image_raw', 10)
        self.rgb = self.create_publisher(Image, '/camera/camera/color/image_raw', 10)
        self.info = self.create_publisher(CameraInfo, '/camera/camera/color/camera_info', 10)
        self.ranges = [self.create_publisher(Range, '/sensors/ultrasonic/sensor_{}'.format(i), 10) for i in range(4)]
        self.create_subscription(TwistStamped, '/simulation/desired_motion', self.command, 10)
        self.create_timer(0.05, self.tick)

    def command(self, msg):
        now = self.get_clock().now().nanoseconds*1e-9
        stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        if 0 <= now-stamp <= 0.3 and all(math.isfinite(v) for v in (msg.twist.linear.x, msg.twist.angular.z)):
            self.speed = max(0.0, min(0.3, msg.twist.linear.x))
            self.rate = max(-0.8, min(0.8, msg.twist.angular.z))
            self.command_stamp = stamp

    def tick(self):
        stamp = self.get_clock().now().to_msg()
        now = stamp.sec+stamp.nanosec*1e-9
        self.elapsed += 0.05
        if now-self.command_stamp > 0.3 or now < self.command_stamp:
            self.speed, self.rate = 0.0, 0.0
        self.x += self.speed*math.cos(self.yaw)*0.05
        self.y += self.speed*math.sin(self.yaw)*0.05
        self.yaw += self.rate*0.05
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = stamp, 'odom', 'base_link'
        odom.pose.pose.position.x, odom.pose.pose.position.y = self.x, self.y
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = math.sin(self.yaw/2), math.cos(self.yaw/2)
        odom.twist.twist.linear.x, odom.twist.twist.angular.z = self.speed, self.rate
        for i in (0, 7, 14, 21, 28, 35):
            odom.pose.covariance[i] = 0.01
            odom.twist.covariance[i] = 0.01
        if self.scenario != 'lost_localization' or self.elapsed < 2:
            self.odom.publish(odom)
            imu = Imu()
            imu.header.stamp, imu.header.frame_id = stamp, 'imu_link'
            imu.orientation = odom.pose.pose.orientation
            imu.angular_velocity.z = self.rate
            for i in (0, 4, 8):
                imu.orientation_covariance[i] = 0.01
                imu.angular_velocity_covariance[i] = 0.01
            imu.linear_acceleration_covariance[0] = -1.0
            self.imu.publish(imu)
        gps = NavSatFix()
        gps.header.stamp, gps.header.frame_id = stamp, 'gps_link'
        gps.latitude, gps.longitude, gps.altitude = 45.0+self.y/111319.49, -93.0+self.x/(111319.49*math.cos(math.pi/4)), 0.0
        gps.status.status = 0
        gps.position_covariance_type = 2
        gps.position_covariance = [4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 9.0]
        self.gps.publish(gps)
        scene = Scene()
        scene.header.stamp, scene.header.frame_id = stamp, 'camera_color_optical_frame'
        scene.healthy, scene.path_confidence, scene.signal_state = True, 1.0, 'NONE'
        if self.scenario == 'stop_sign' and 2 <= self.elapsed < 6:
            scene.stop_sign = True
        if self.scenario == 'traffic_light':
            scene.signal_state = 'RED' if self.elapsed < 3 else 'UNKNOWN' if self.elapsed < 4 else 'GREEN'
        if self.scenario == 'perception_failure' and self.elapsed >= 2:
            scene.healthy = False
        if scene.stop_sign or scene.signal_state != 'NONE':
            detection = Detection()
            detection.class_name = 'stop sign' if scene.stop_sign else 'traffic light'
            detection.confidence, detection.signal_confidence, detection.signal_state = 1.0, 1.0, scene.signal_state
            scene.detections = [detection]
        self.scene.publish(scene)
        image = Image()
        image.header = scene.header
        image.width, image.height, image.encoding, image.step = 8, 8, '16UC1', 16
        distance = 300 if self.scenario == 'obstacle' and self.elapsed >= 2 else 3000
        image.data = struct.pack('<64H', *([distance]*64))
        if self.scenario != 'stale_sensor' or self.elapsed < 2:
            self.depth.publish(image)
            for i, publisher in enumerate(self.ranges):
                r = Range()
                r.header.stamp, r.header.frame_id = stamp, 'ultrasonic_{}'.format(i)
                r.radiation_type, r.field_of_view = Range.ULTRASOUND, 0.3
                r.min_range, r.max_range, r.range = 0.02, 4.0, distance/1000.0
                publisher.publish(r)
        color = Image()
        color.header, color.width, color.height = scene.header, 8, 8
        color.encoding, color.step, color.data = 'bgr8', 24, bytes([255]*192)
        self.rgb.publish(color)
        info = CameraInfo()
        info.header, info.width, info.height = scene.header, 8, 8
        info.k = [8.0, 0.0, 4.0, 0.0, 8.0, 4.0, 0.0, 0.0, 1.0]
        self.info.publish(info)


def main():
    rclpy.init()
    n = Simulator()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()
