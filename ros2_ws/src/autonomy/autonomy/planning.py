"""Planning (SKELETON) — Jetson.

Fused obstacles + a goal (from the iPhone GPS/destination app) -> a path in base_link.
Legacy entry point: emits an empty path. Use autonomy_pipeline for local planning.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, PoseStamped
from nav_msgs.msg import Path


class Planning(Node):
    def __init__(self):
        super().__init__('planning')
        self.create_subscription(PoseArray, 'fusion/obstacles', self.on_world, 5)
        self.create_subscription(PoseStamped, 'goal', self.on_goal, 5)
        self.path_pub = self.create_publisher(Path, 'planning/path', 5)
        self.world = None
        self.goal = None
        self.create_timer(0.1, self.plan)   # 10 Hz

    def on_world(self, msg):
        self.world = msg

    def on_goal(self, msg):
        self.goal = msg

    def plan(self):
        path = Path()
        path.header.frame_id = 'base_link'
        path.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(path)
        self.get_logger().warn('legacy planner disabled; launch autonomy_pipeline',
                               throttle_duration_sec=5.0)


def main():
    rclpy.init()
    node = Planning()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
