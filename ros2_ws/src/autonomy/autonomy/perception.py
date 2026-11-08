"""Perception (SKELETON) — Jetson.

RealSense D405 color+depth -> obstacle positions in the camera frame.
The detector (e.g. Ultralytics YOLO) and the depth deprojection are NOT implemented here;
this node is the wiring + the interface to build on. Camera feeds come from another branch.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray


class Perception(Node):
    def __init__(self):
        super().__init__('perception')
        self.create_subscription(Image, 'd405/color/image_raw', self.on_color, 5)
        self.create_subscription(Image, 'd405/depth/image_rect_raw', self.on_depth, 5)
        self.obstacles = self.create_publisher(PoseArray, 'perception/obstacles', 5)
        self.depth = None

    def on_depth(self, msg):
        self.depth = msg

    def on_color(self, msg):
        # TODO(build-off): run the detector on `msg`; for each detection take the box
        # center, read range from the aligned `self.depth`, deproject with the D405
        # intrinsics to (x,y,z) in the camera frame, and publish as PoseArray.
        self.get_logger().warn('perception: detector not implemented — publishing nothing',
                               throttle_duration_sec=5.0)


def main():
    rclpy.init()
    node = Perception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
