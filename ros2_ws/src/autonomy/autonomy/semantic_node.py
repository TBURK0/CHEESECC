"""Optional YOLO RGB/aligned-depth semantic node. No actuation decisions."""
import json

import message_filters
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from .depth_io import decode_depth
from .semantic import BoxTracker, detection_position
from .spatial import RigidTransform


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class SemanticNode(Node):
    def __init__(self):
        super().__init__('semantic_perception')
        self.declare_parameter('model', 'yolo11n.pt')
        self.declare_parameter('confidence', 0.25)
        from ultralytics import YOLO
        self.model = YOLO(self.get_parameter('model').value)
        self.info = None
        self.bridge, self.tracker = CvBridge(), BoxTracker()
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.pub = self.create_publisher(String, 'perception/semantic', 5)
        self.create_subscription(CameraInfo, 'd405/color/camera_info', self.on_info, 5)
        self.color = message_filters.Subscriber(self, Image, 'd405/color/image_raw')
        self.depth = message_filters.Subscriber(self, Image, 'd405/aligned_depth_to_color/image_raw')
        self.sync = message_filters.ApproximateTimeSynchronizer([self.color, self.depth], 5, 0.05)
        self.sync.registerCallback(self.on_pair)

    def on_info(self, msg):
        self.info = msg

    def on_pair(self, color, depth):
        info = self.info
        stamp = stamp_seconds(color.header.stamp)
        age = self.get_clock().now().nanoseconds * 1e-9 - stamp
        if (info is None or not 0 <= age <= 0.3 or color.width != depth.width or
                color.height != depth.height or info.width != color.width or info.height != color.height or
                info.header.frame_id != color.header.frame_id or depth.header.frame_id != color.header.frame_id):
            return
        try:
            tf = self.tf.lookup_transform('base_link', color.header.frame_id,
                                          rclpy.time.Time.from_msg(color.header.stamp)).transform
            t, q = tf.translation, tf.rotation
            transform = RigidTransform((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))
            rows, scale = decode_depth(depth)
            image = self.bridge.imgmsg_to_cv2(color, desired_encoding='bgr8')
            result = self.model.predict(image, conf=self.get_parameter('confidence').value, verbose=False)[0]
            detections = []
            for box, conf, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                                      result.boxes.cls.cpu().tolist()):
                name = result.names[int(cls)]
                position = detection_position(box, rows, scale,
                                              (info.k[0], info.k[4], info.k[2], info.k[5]), transform)
                detections.append(dict(name=name, confidence=conf, xyxy=box,
                                       position_base_link=position))
            tracked = self.tracker.update(detections, stamp)
            self.pub.publish(String(data=json.dumps(dict(stamp=stamp, frame_id='base_link',
                                                          detections=tracked), allow_nan=False)))
        except (ValueError, TransformException) as exc:
            self.get_logger().warn('semantic frame rejected: ' + str(exc), throttle_duration_sec=2.0)


def main():
    rclpy.init()
    node = SemanticNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
