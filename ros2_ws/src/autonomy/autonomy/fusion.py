"""Fusion (SKELETON) — Jetson.

Combine camera obstacles + ultrasonic ranges + IMU into one obstacle list in base_link.
The actual fusion is developed on another branch; this node is the interface it plugs into.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range, Imu
from geometry_msgs.msg import PoseArray


class Fusion(Node):
    def __init__(self):
        super().__init__('fusion')
        self.create_subscription(PoseArray, 'perception/obstacles', self.on_obstacles, 5)
        for name in ('fl', 'fr', 'rl', 'rr'):
            self.create_subscription(Range, f'ultrasonic/{name}', self.on_range, 5)
        self.create_subscription(Imu, 'imu/mpu6050', self.on_imu, 10)
        self.world = self.create_publisher(PoseArray, 'fusion/obstacles', 5)
        self.ranges = {}

    def on_range(self, msg):
        self.ranges[msg.header.frame_id] = msg

    def on_imu(self, msg):
        pass  # TODO(build-off): ego-motion / orientation for temporal association

    def on_obstacles(self, msg):
        # TODO(build-off): transform camera obstacles into base_link, merge with the
        # ultrasonic ranges (near-field), associate/track over time, publish the fused set.
        self.get_logger().warn('fusion: merge logic lives on another branch — not implemented here',
                               throttle_duration_sec=5.0)


def main():
    rclpy.init()
    node = Fusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
