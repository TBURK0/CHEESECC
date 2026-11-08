import time
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vehicle_interfaces.msg import Scene, Detection
from .core import PretrainedDetector, classical_path


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('rgb_perception')
        self.detector = None
        weights = self.declare_parameter('weights', '').value
        try:
            if weights:
                self.detector = PretrainedDetector(weights)
        except Exception as error:
            self.get_logger().error('Detector unavailable: {}'.format(error))
        self.pub = self.create_publisher(Scene, '/perception/scene', 10)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image, qos_profile_sensor_data)

    def image(self, image):
        msg = Scene()
        msg.header, msg.signal_state = image.header, 'UNKNOWN'
        started = time.perf_counter()
        try:
            from cv_bridge import CvBridge
            bgr = CvBridge().imgmsg_to_cv2(image, desired_encoding='bgr8')
            msg.path_confidence, _ = classical_path(bgr)
            if self.detector is None:
                raise ValueError('reviewed pretrained weights not configured')
            rows = self.detector.detect(bgr, image.header.stamp)
            lights = []
            for row in rows:
                d = Detection()
                d.class_name, d.confidence = row['class_name'], row['confidence']
                d.signal_state, d.signal_confidence = row['signal'], row['signal_confidence']
                d.bbox_xyxy = [float(v) for v in row['box']]
                msg.detections.append(d)
                msg.stop_sign |= d.class_name == 'stop sign' and d.confidence >= 0.8
                if d.class_name == 'traffic light':
                    lights.append(d.signal_state)
            # Missing light detection cannot establish an intersection is clear.
            msg.signal_state = lights[0] if len(lights) == 1 else 'UNKNOWN'
            msg.healthy = msg.path_confidence >= 0.5
        except Exception as error:
            msg.healthy = False
            self.get_logger().warning('Perception stopped: {}'.format(error))
        self.get_logger().debug('inference_ms={:.2f}'.format((time.perf_counter()-started)*1000))
        self.pub.publish(msg)


def main():
    rclpy.init()
    n = PerceptionNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()
