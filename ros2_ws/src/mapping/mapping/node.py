import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, Range
from vehicle_interfaces.msg import ObstacleStatus
from .core import clearance


class MappingNode(Node):
    def __init__(self):
        super().__init__('local_obstacle_mapping')
        self.depth, self.ranges = None, [None]*4
        self.pub = self.create_publisher(ObstacleStatus, '/mapping/obstacles', 10)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.image, qos_profile_sensor_data)
        for i in range(4):
            self.create_subscription(Range, '/sensors/ultrasonic/sensor_{}'.format(i), lambda m, index=i: self.range(m, index), qos_profile_sensor_data)
        self.create_timer(0.05, self.tick)

    def range(self, msg, index):
        self.ranges[index] = msg

    def image(self, msg):
        # Conservative nearest valid pixel across the image; no holes authorize motion.
        import struct
        size = 2 if msg.encoding == '16UC1' else 4 if msg.encoding == '32FC1' else 0
        value = float('nan')
        if size and msg.width > 0 and msg.height > 0 and msg.step >= msg.width*size and len(msg.data) >= msg.step*msg.height:
            values = []
            data = bytes(msg.data)
            fmt = ('>' if msg.is_bigendian else '<') + ('H' if size == 2 else 'f')
            for row in range(msg.height):
                for col in range(msg.width):
                    v = struct.unpack_from(fmt, data, row*msg.step+col*size)[0]
                    values.append(v*0.001 if size == 2 else v)
            if values and all(math.isfinite(v) and 0.02 <= v <= 8 for v in values):
                value = min(values)
        self.depth = (msg.header.stamp, value)

    def tick(self):
        now = self.get_clock().now()
        msg = ObstacleStatus()
        msg.header.stamp, msg.header.frame_id = now.to_msg(), 'base_link'
        if self.depth and all(self.ranges):
            stamps = [self.depth[0]]+[r.header.stamp for r in self.ranges]
            msg.clear, msg.reason = clearance(self.depth[1], [r.range for r in self.ranges],
                [s.sec+s.nanosec*1e-9 for s in stamps], now.nanoseconds*1e-9)
            msg.source_stamps = stamps
        else:
            msg.clear, msg.reason = False, 'MISSING_SENSOR'
        self.pub.publish(msg)


def main():
    rclpy.init()
    n = MappingNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()
