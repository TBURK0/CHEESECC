"""ROS 2 node that owns the UNO Q serial link. Contract: docs/serial_protocol.md."""
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from vehicle_interfaces.msg import ActuatorCommand, UnoQFrame, VehicleStatus

from . import protocol
from .bridge_logic import BridgeLogic
from .link import Link


class VehicleBridge(Node):
    def __init__(self):
        super().__init__('vehicle_bridge')
        p = {
            'port': '/dev/ttyUSB0', 'baud': 460800, 'cmd_rate_hz': 50.0,
            'cmd_input_timeout_s': 0.2, 'telemetry_timeout_s': 0.5, 'allow_arm': False,
            'max_throttle': 0.25, 'max_brake': 1.0, 'max_steering': 1.0,
        }
        for k, v in p.items():
            self.declare_parameter(k, v)
        g = {k: self.get_parameter(k).value for k in p}

        self.logic = BridgeLogic(g['allow_arm'], g['cmd_rate_hz'], g['cmd_input_timeout_s'],
                                 g['telemetry_timeout_s'], g['max_throttle'], g['max_brake'],
                                 g['max_steering'])
        self.link = Link(g['port'], g['baud'])
        self.decoder = protocol.LineDecoder()
        self.boot = None
        self.host_bad_frames = 0
        self._next_open = 0.0

        self.status_pub = self.create_publisher(VehicleStatus, 'vehicle/status', 10)
        self.frame_pub = self.create_publisher(UnoQFrame, 'uno_q/frames', 100)
        self.diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        self.create_subscription(ActuatorCommand, 'vehicle/cmd', self._on_cmd, 10)
        self.create_subscription(Bool, 'vehicle/estop', self._on_estop, 10)
        for name, kind in (('arm', 'ARM'), ('disarm', 'DISARM'), ('estop', 'ESTOP'),
                           ('reset_estop', 'RESET'), ('i2c_scan', 'SCAN')):
            self.create_service(Trigger, f'vehicle/{name}', self._service(kind))

        # Poll faster than cmd_rate so rx stamps are tight; BridgeLogic rate-limits CMD itself.
        self.create_timer(1.0 / max(g['cmd_rate_hz'], 200.0), self._tick)
        self.create_timer(1.0, self._publish_diagnostics)
        if g['allow_arm']:
            self.get_logger().warn('allow_arm is TRUE: vehicle/arm can enable actuator commands')

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _send(self, bodies):
        for b in bodies:
            self.link.write(protocol.encode(b))

    # ---- inputs ----

    def _on_cmd(self, msg):
        err = self.logic.on_input(msg.throttle, msg.brake, msg.steering, self._now())
        if err:
            self.get_logger().error(err)

    def _on_estop(self, msg):
        if msg.data:
            self._send(self.logic.request('ESTOP')[2])

    def _service(self, kind):
        def cb(_req, res):
            ok, reason, bodies = self.logic.request(kind)
            self._send(bodies)
            res.success = ok and self.link.connected
            res.message = reason or (f'{kind} sent; MCU result is in vehicle/status'
                                     if self.link.connected else 'link down')
            return res
        return cb

    # ---- periodic ----

    def _tick(self):
        now = self._now()
        if not self.link.connected and now >= self._next_open:
            self._next_open = now + 1.0
            if self.link.open():
                self.get_logger().info(f'link open: {self.link.port}')
                self.decoder = protocol.LineDecoder()
            else:
                self.get_logger().warn(f'link open failed: {self.link.last_error}',
                                       throttle_duration_sec=10.0)
        for kind, item in self.decoder.feed(self.link.read()):
            if kind == 'bad':
                self.host_bad_frames += 1
                self.get_logger().warn(f'bad frame from MCU: {item}', throttle_duration_sec=2.0)
            else:
                self._handle(item, now)
        self._send(self.logic.tick(now))

    def _handle(self, body, now):
        try:
            m = protocol.parse(body)
        except protocol.ProtocolError as e:
            self.host_bad_frames += 1
            self.get_logger().warn(f'unparseable frame {body!r}: {e}', throttle_duration_sec=2.0)
            return
        if isinstance(m, protocol.Status):
            self.logic.on_status(m, now)
            self._publish_status()
        elif isinstance(m, (protocol.Ultrasonic, protocol.Imu, protocol.I2cIdent, protocol.I2cEnd)):
            f = UnoQFrame()
            f.header.stamp = self.get_clock().now().to_msg()
            f.body = body
            self.frame_pub.publish(f)
            if isinstance(m, (protocol.I2cIdent, protocol.I2cEnd)):
                self.get_logger().info(f'i2c scan: {body}')
        elif isinstance(m, protocol.Boot):
            self.boot = m
            self.get_logger().info(f'MCU boot: {body}')
        elif isinstance(m, protocol.Ack):
            if not m.ok:
                self.get_logger().warn(f'MCU rejected seq {m.seq}: {m.reason}')
        elif isinstance(m, protocol.Log):
            self.get_logger().info(f'MCU: {m.text}')

    def _publish_status(self):
        s = self.logic.status
        msg = VehicleStatus()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.link_ok = self.logic.link_ok
        msg.state = s.state if s and self.logic.link_ok else 'UNKNOWN'
        msg.armed = self.logic.mcu_armed
        if s:
            msg.faults = s.faults
            msg.fault_names = protocol.fault_names(s.faults)
            msg.cmd_age_s = s.cmd_age_ms / 1000.0
            msg.throttle, msg.brake, msg.steering = (s.throttle_pm / 1000.0, s.brake_pm / 1000.0,
                                                     s.steer_pm / 1000.0)
            msg.bad_frames = s.bad_frames
        if self.boot:
            msg.firmware_version = self.boot.fw_version
            msg.outputs_enabled_build = self.boot.outputs_enabled
            msg.hw_watchdog_verified = self.boot.hw_watchdog_verified
        self.status_pub.publish(msg)

    def _publish_diagnostics(self):
        s = self.logic.status
        d = DiagnosticStatus(name='vehicle_bridge: UNO Q link', hardware_id=self.link.port)
        if not self.logic.link_ok:
            d.level, d.message = DiagnosticStatus.ERROR, 'no telemetry from MCU'
            self._publish_status()  # keep vehicle/status alive with link_ok=False
        elif s.state == 'ESTOP' or s.faults & (protocol.FAULT['ESTOP_INPUT'] | protocol.FAULT['IMU_FAULT']
                                                | protocol.FAULT['US_FAULT']):
            d.level, d.message = DiagnosticStatus.WARN, s.state
        else:
            d.level, d.message = DiagnosticStatus.OK, s.state
        d.values = [KeyValue(key='faults', value=','.join(protocol.fault_names(s.faults)) if s else ''),
                    KeyValue(key='mcu_bad_frames', value=str(s.bad_frames if s else '')),
                    KeyValue(key='host_bad_frames', value=str(self.host_bad_frames)),
                    KeyValue(key='link_error', value=self.link.last_error or '')]
        arr = DiagnosticArray(status=[d])
        arr.header.stamp = self.get_clock().now().to_msg()
        self.diag_pub.publish(arr)

    def shutdown(self):
        self._send(self.logic.request('DISARM')[2])
        self.link.close()


def main():
    rclpy.init()
    node = VehicleBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
