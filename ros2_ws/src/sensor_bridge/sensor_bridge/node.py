"""Turns UNO Q sensor frames (from vehicle_bridge on uno_q/frames) into ROS sensor messages."""
import rclpy
from builtin_interfaces.msg import Time
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from sensor_msgs.msg import Imu, Range
from std_srvs.srv import Trigger
from vehicle_bridge import protocol
from vehicle_bridge.time_sync import McuClock
from vehicle_interfaces.msg import UnoQFrame

from .axis_map import apply, parse_axis_map
from .conversions import accel_si, gyro_si, us_range_m
from .imu_calibration import GyroBiasCalibrator

IMUS = ('MPU6050', 'ITG3200')
_IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1]


def _stamp(t_s):
    sec = int(t_s // 1)
    return Time(sec=sec, nanosec=int((t_s - sec) * 1e9))


class SensorBridge(Node):
    def __init__(self):
        super().__init__('sensor_bridge')
        p = {
            'us_min_range': 0.02, 'us_max_range': 4.0,
            'us_field_of_view': 0.26,   # ~15 deg, HC-SR04 nominal, UNVERIFIED for the fitted sensor
            'stale_timeout_s': 1.0,
            'gyro_calib_samples': 200, 'gyro_calib_max_std': 0.02,
            'mpu6050_axis_map': _IDENTITY, 'itg3200_axis_map': _IDENTITY,  # UNVERIFIED mounting
        }
        for k, v in p.items():
            self.declare_parameter(k, v)
        g = {k: self.get_parameter(k).value for k in p}
        self.cfg = g
        # Raises at startup on an invalid map: better than publishing wrongly rotated data.
        self.axis = {'MPU6050': parse_axis_map(g['mpu6050_axis_map']),
                     'ITG3200': parse_axis_map(g['itg3200_axis_map'])}
        self.calib = {s: GyroBiasCalibrator(g['gyro_calib_samples'], g['gyro_calib_max_std'])
                      for s in IMUS}
        self.clock = McuClock()

        self.us_pub = {i: self.create_publisher(Range, f'ultrasonic/{n}', 10)
                       for i, n in protocol.US_IDS.items()}
        self.imu_pub = {s: self.create_publisher(Imu, f'imu/{s.lower()}', 50) for s in IMUS}
        self.diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        self.create_subscription(UnoQFrame, 'uno_q/frames', self._on_frame, 200)
        self.create_service(Trigger, 'imu/calibrate', self._on_calibrate)
        self.create_timer(1.0, self._publish_diagnostics)

        # per-sensor: [last host time, last status, frames since last diag]
        names = [f'us_{n}' for n in protocol.US_IDS.values()] + [s.lower() for s in IMUS]
        self.track = {n: [None, 'NO_DATA', 0] for n in names}

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_calibrate(self, _req, res):
        for c in self.calib.values():
            c.restart()
        res.success, res.message = True, 'gyro calibration restarted; keep the car still'
        return res

    def _on_frame(self, frame):
        try:
            m = protocol.parse(frame.body)
        except protocol.ProtocolError as e:
            self.get_logger().warn(f'bad frame {frame.body!r}: {e}', throttle_duration_sec=2.0)
            return
        rx = frame.header.stamp.sec + frame.header.stamp.nanosec * 1e-9
        if isinstance(m, protocol.Ultrasonic):
            self._on_us(m, self.clock.observe(m.t_us, rx))
        elif isinstance(m, protocol.Imu):
            self._on_imu(m, self.clock.observe(m.t_us, rx))

    def _note(self, name, status):
        t = self.track[name]
        t[0], t[1], t[2] = self._now(), status, t[2] + 1

    def _on_us(self, m, t_s):
        name = protocol.US_IDS[m.id]
        self._note(f'us_{name}', m.status)
        r = Range()
        r.header.stamp = _stamp(t_s)
        r.header.frame_id = f'us_{name}'
        r.radiation_type = Range.ULTRASOUND
        r.field_of_view = float(self.cfg['us_field_of_view'])
        r.min_range = float(self.cfg['us_min_range'])
        r.max_range = float(self.cfg['us_max_range'])
        r.range = float(us_range_m(m.status, m.dist_mm))
        self.us_pub[m.id].publish(r)

    def _on_imu(self, m, t_s):
        self._note(m.src.lower(), m.status)
        if m.status != 'OK' or m.gyro_urad_s is None:
            return
        cal = self.calib[m.src]
        gyro = gyro_si(m.gyro_urad_s)
        if not cal.calibrated:
            cal.add(gyro)
            return  # never publish an uncalibrated gyro
        ax = self.axis[m.src]
        msg = Imu()
        msg.header.stamp = _stamp(t_s)
        msg.header.frame_id = f'imu_{m.src.lower()}'
        msg.orientation_covariance[0] = -1.0  # no orientation estimate
        g = apply(ax, cal.correct(gyro))
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = g
        accel = accel_si(m.accel_mm_s2)
        if accel is None:
            msg.linear_acceleration_covariance[0] = -1.0  # ITG3200: gyro only
        else:
            a = apply(ax, accel)
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = a
        self.imu_pub[m.src].publish(msg)

    def _publish_diagnostics(self):
        now = self._now()
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        for name, t in self.track.items():
            last, status, count = t
            d = DiagnosticStatus(name=f'sensor_bridge: {name}', hardware_id=name)
            age = None if last is None else now - last
            if age is None or age > self.cfg['stale_timeout_s']:
                d.level, d.message = DiagnosticStatus.ERROR, 'stale: no frames'
            elif status in ('OK', 'TOO_FAR', 'TIMEOUT', 'TOO_CLOSE'):
                d.level, d.message = DiagnosticStatus.OK, status
            else:
                d.level, d.message = DiagnosticStatus.ERROR, status
            src = name.upper()
            if src in self.calib and d.level == DiagnosticStatus.OK and not self.calib[src].calibrated:
                d.level, d.message = DiagnosticStatus.WARN, self.calib[src].message
            d.values = [KeyValue(key='last_status', value=status),
                        KeyValue(key='age_s', value='' if age is None else f'{age:.3f}'),
                        KeyValue(key='rate_hz', value=str(count))]
            t[2] = 0
            arr.status.append(d)
        self.diag_pub.publish(arr)


def main():
    rclpy.init()
    node = SensorBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
