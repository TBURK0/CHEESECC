import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import NavSatFix, Imu
from .phone import PhoneParser, MotionParser


class PhoneNode(Node):
    def __init__(self):
        super().__init__('phone_fixture_ingestion')
        self.parser = PhoneParser()
        self.motion_parser = MotionParser()
        self.pub = self.create_publisher(NavSatFix, '/phone/gps', 10)
        self.motion_pub = self.create_publisher(Imu, '/phone/imu', 10)
        self.create_subscription(String, '/simulation/phone_json', self.receive, 10)

    def receive(self, msg):
        try:
            if len(msg.data) > 8192:
                raise ValueError('oversized telemetry')
            import json
            if json.loads(msg.data).get('type') == 'phone.motion':
                motion = self.motion_parser.parse(msg.data, self.get_clock().now().nanoseconds*1e-9)
                imu = Imu()
                imu.header.frame_id = 'base_link'
                imu.header.stamp.sec = int(motion['stamp'])
                imu.header.stamp.nanosec = int((motion['stamp'] % 1)*1e9)
                imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w = map(float, motion['q'])
                imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = map(float, motion['rate'])
                imu.linear_acceleration_covariance[0] = -1.0
                for i in (0, 4, 8):
                    imu.orientation_covariance[i] = motion['variance']
                    imu.angular_velocity_covariance[i] = 0.01
                self.motion_pub.publish(imu)
                return
            fix = self.parser.parse(msg.data, self.get_clock().now().nanoseconds*1e-9)
            out = NavSatFix()
            out.header.frame_id = 'gps_link'
            out.header.stamp.sec = int(fix['stamp'])
            out.header.stamp.nanosec = int((fix['stamp'] % 1)*1e9)
            out.latitude, out.longitude, out.altitude = map(float, (fix['latitude'], fix['longitude'], fix['altitude']))
            out.status.status, out.position_covariance_type = 0, 2
            for i, variance in zip((0, 4, 8), fix['covariance']):
                out.position_covariance[i] = variance
            self.pub.publish(out)
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            self.get_logger().warning('Rejected phone sample: {}'.format(error))


def main():
    rclpy.init()
    n = PhoneNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()
