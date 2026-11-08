"""RC telemetry publisher — assembles the dashboard signal frame and sends it over BLE/serial.

Subscribes the RC car's existing ROS topics (vehicle_bridge / sensor_bridge) and emits one
JSON Telemetry frame per tick. Live fields are filled from real messages; pending-hardware
fields stay None (see schema.py). Default transport is 'stdout' so you can run it with no
phone attached.

    ros2 run rc_telemetry rc_telemetry --ros-args -p transport:=serial -p port:=/dev/ttyACM0
"""
import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range, Imu
from vehicle_interfaces.msg import VehicleStatus

from rc_telemetry.schema import Telemetry
from rc_telemetry.transport import make_transport


class TelemetryNode(Node):
    def __init__(self):
        super().__init__("rc_telemetry")
        self.declare_parameter("transport", "stdout")      # stdout | serial
        self.declare_parameter("port", "/dev/ttyACM0")
        self.declare_parameter("baud", 115200)
        self.declare_parameter("rate_hz", 10.0)

        self.tx = make_transport(
            self.get_parameter("transport").value,
            self.get_parameter("port").value,
            int(self.get_parameter("baud").value),
            self.get_logger(),
        )
        self.status = None
        self.us = {}
        self.yaw_rate = None

        self.create_subscription(VehicleStatus, "vehicle/status", self._on_status, 5)
        for name in ("fl", "fr", "rl", "rr"):
            self.create_subscription(Range, f"ultrasonic/{name}", self._on_range, 5)
        self.create_subscription(Imu, "imu/mpu6050", self._on_imu, 10)
        self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self._tick)

    def _on_status(self, m):
        self.status = m

    def _on_range(self, m):
        self.us[m.header.frame_id] = m.range if math.isfinite(m.range) else None

    def _on_imu(self, m):
        self.yaw_rate = m.angular_velocity.z

    def _tick(self):
        t = Telemetry(ts=time.time())
        s = self.status
        if s is not None:
            t.state = s.state
            t.armed = s.armed
            t.link_ok = s.link_ok
            t.throttle = s.throttle
            t.brake = s.brake
            t.steering = s.steering
            t.faults = list(s.fault_names)
            t.firmware = s.firmware_version
            t.outputs_enabled = s.outputs_enabled_build
            t.hw_watchdog_verified = s.hw_watchdog_verified
        t.us_fl = self.us.get("us_fl")
        t.us_fr = self.us.get("us_fr")
        t.us_rl = self.us.get("us_rl")
        t.us_rr = self.us.get("us_rr")
        t.yaw_rate = self.yaw_rate
        try:
            t.validate()
        except AssertionError as e:
            self.get_logger().warn(f"dropping bad telemetry frame: {e}")
            return
        self.tx.send(t.to_json())


def main():
    rclpy.init()
    node = TelemetryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
