"""Depth -> local map/scene/path and a separately gated command output."""
import json
import math
import os
import struct

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import CameraInfo, Image, Range
from std_msgs.msg import String
from vehicle_interfaces.msg import ActuatorCommand, VehicleStatus
from tf2_ros import Buffer, TransformListener, TransformException

from .core import CommandInbox, LocalMap, Point, depth_points, safe_command, scene_summary
from .comms import AuthenticatedInbox
from .depth_io import decode_depth
from .mapping import EvidenceMap
from .motion import VehicleGeometry, lattice_plan, pursuit, command_sweep_clear
from .spatial import (Pose2, RigidTransform, ObstacleMemory, calibrated_depth,
                      ultrasonic_endpoint, fuse_points, mark_memory_occupied)


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class Pipeline(Node):
    def __init__(self):
        super().__init__('autonomy_pipeline')
        self.declare_parameter('mount_calibrated', False)
        self.declare_parameter('camera_height_m', 0.2)
        self.declare_parameter('use_tf', True)
        self.declare_parameter('geometry_calibrated', False)
        self.declare_parameter('command_source', 'candidate')
        self.declare_parameter('operator', 'driver')
        self.declare_parameter('require_ultrasonics', False)
        self.declare_parameter('wheelbase_m', 0.26)
        self.declare_parameter('front_m', 0.32)
        self.declare_parameter('rear_m', 0.10)
        self.declare_parameter('half_width_m', 0.12)
        self.declare_parameter('max_steer_rad', 0.45)
        self.geometry = VehicleGeometry(*[self.get_parameter(k).value for k in
                                         ('wheelbase_m', 'front_m', 'rear_m', 'half_width_m', 'max_steer_rad')])
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.pose = None
        self.memory = ObstacleMemory()
        self.range_points = {}
        self.camera_points = []
        self.camera_pose = None
        self.camera_origin = Point(0, 0)
        self.range_health = {}
        self.range_hazards = {}
        self.source = self.get_parameter('command_source').value
        if self.source not in ('candidate', 'json', 'authenticated', 'controller'):
            raise ValueError('unknown command_source')
        self.info = self.goal = self.status = self.candidate = self.speed = None
        self.scene_time = self.status_time = self.speed_time = self.command_time = -math.inf
        self.points = []
        self.map = LocalMap()
        self.evidence = EvidenceMap()
        self.last_mapped_scene = -math.inf
        self.inbox = CommandInbox()
        if self.source == 'authenticated':
            try:
                key = bytes.fromhex(os.environ.get('AUTONOMY_COMMAND_KEY', ''))
            except ValueError as exc:
                raise ValueError('AUTONOMY_COMMAND_KEY must be hex encoded') from exc
            self.inbox = AuthenticatedInbox(key, self.get_parameter('operator').value)
            self.get_logger().info('Receiver command session: ' + self.inbox.session)
        self.map_pub = self.create_publisher(OccupancyGrid, 'mapping/local', 5)
        self.history_pub = self.create_publisher(OccupancyGrid, 'mapping/odom', 5)
        self.scene_pub = self.create_publisher(String, 'scene/summary', 5)
        self.obstacle_pub = self.create_publisher(PoseArray, 'fusion/obstacles', 5)
        self.path_pub = self.create_publisher(Path, 'planning/path', 5)
        self.command_pub = self.create_publisher(ActuatorCommand, 'vehicle/cmd', 5)
        self.safety_pub = self.create_publisher(String, 'safety/reason', 5)
        self.create_subscription(CameraInfo, 'd405/depth/camera_info', self.on_info, 5)
        self.create_subscription(Image, 'd405/depth/image_rect_raw', self.on_depth, 5)
        self.create_subscription(PoseStamped, 'goal', self.on_goal, 5)
        self.create_subscription(Odometry, 'odom', self.on_speed, 5)
        self.create_subscription(VehicleStatus, 'vehicle/status', self.on_status, 5)
        if self.source == 'candidate':
            self.create_subscription(ActuatorCommand, 'autonomy/cmd_candidate', self.on_command, 5)
        elif self.source in ('json', 'authenticated'):
            self.create_subscription(String, 'autonomy/cmd_json', self.on_json_command, 5)
        for name in ('fl', 'fr', 'rl', 'rr'):
            self.create_subscription(Range, 'ultrasonic/' + name,
                                     lambda msg, name=name: self.on_range(name, msg), 5)
        self.create_timer(0.05, self.tick)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_info(self, msg):
        self.info = msg

    def on_goal(self, msg):
        self.goal = msg if msg.header.frame_id == 'base_link' else None

    def on_speed(self, msg):
        self.pose = None
        if msg.header.frame_id != 'odom' or msg.child_frame_id != 'base_link':
            return
        q, p = msg.pose.pose.orientation, msg.pose.pose.position
        norm = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w
        if not all(math.isfinite(v) for v in (p.x, p.y, norm)) or abs(norm - 1) > 0.01:
            return
        yaw = math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2 * (q.y*q.y + q.z*q.z))
        self.pose = Pose2(p.x, p.y, yaw)
        # Magnitude avoids treating reverse motion as stationary.
        v = msg.twist.twist.linear
        self.speed = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
        self.speed_time = stamp_seconds(msg.header.stamp)

    def transform(self, frame, stamp):
        tf = self.tf.lookup_transform('base_link', frame, rclpy.time.Time.from_msg(stamp)).transform
        t, q = tf.translation, tf.rotation
        return RigidTransform((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))

    def on_range(self, name, msg):
        self.range_points.pop(name, None)
        self.range_health.pop(name, None)
        self.range_hazards.pop(name, None)
        t = stamp_seconds(msg.header.stamp)
        if msg.range == -math.inf:
            self.range_hazards[name] = t
            return
        if self.pose is None or abs(t - self.speed_time) > 0.05:
            return
        try:
            transform = self.transform(msg.header.frame_id, msg.header.stamp)
            if msg.range == math.inf:
                self.range_health[name] = t
                return
            point = ultrasonic_endpoint(msg.range, msg.min_range, msg.max_range,
                                        transform)
            if point is not None:
                self.range_points[name] = (self.pose.world(point), t)
                self.range_health[name] = t
        except (TransformException, ValueError):
            return

    def on_status(self, msg):
        self.status, self.status_time = msg, stamp_seconds(msg.header.stamp)

    def on_command(self, msg):
        self.candidate, self.command_time = msg, stamp_seconds(msg.header.stamp)

    def on_json_command(self, msg):
        try:
            data = self.inbox.receive(msg.data, self.now())
        except ValueError as exc:
            self.candidate = None
            self.get_logger().warn(str(exc), throttle_duration_sec=2.0)
            return
        cmd = ActuatorCommand()
        cmd.throttle, cmd.steering = float(data['throttle']), float(data['steering'])
        cmd.brake = float(data.get('brake', 0.0))
        self.candidate, self.command_time = cmd, data['stamp']

    def on_depth(self, msg):
        self.scene_time = -math.inf
        info = self.info
        t = stamp_seconds(msg.header.stamp)
        if not self.get_parameter('mount_calibrated').value or info is None:
            return
        if self.pose is None or abs(t - self.speed_time) > 0.05:
            return
        if (info.header.frame_id != msg.header.frame_id or info.width != msg.width or
                info.height != msg.height):
            return
        try:
            rows, scale = decode_depth(msg)
            # Require valid depth across the central driving corridor; holes are unknown.
            samples = [rows[v][u] * scale
                       for v in range(msg.height // 3, max(msg.height // 3 + 1, 2 * msg.height // 3), 4)
                       for u in range(msg.width // 3, max(msg.width // 3 + 1, 2 * msg.width // 3), 4)]
            if not samples or not all(math.isfinite(z) and 0.1 <= z <= 4.0 for z in samples):
                return
            if self.get_parameter('use_tf').value:
                transform = self.transform(msg.header.frame_id, msg.header.stamp)
                self.camera_points = calibrated_depth(rows, info.k[0], info.k[4], info.k[2], info.k[5],
                                                      scale, transform)
                x, y, _ = transform.apply((0, 0, 0))
                self.camera_origin = Point(x, y)
            else:
                self.camera_origin = Point(0, 0)
                self.camera_points = depth_points(rows, info.k[0], info.k[4], info.k[2], info.k[5],
                                                 scale=scale, camera_height=self.get_parameter('camera_height_m').value)
        except (ValueError, struct.error, TransformException) as exc:
            self.get_logger().warn(str(exc), throttle_duration_sec=2.0)
            return
        self.scene_time = stamp_seconds(msg.header.stamp)
        self.camera_pose = self.pose
        self.camera_points = fuse_points(self.camera_points, [], voxel=self.map.resolution / 2)
        self.memory.update(self.camera_points, self.camera_pose, self.scene_time)

    def tick(self):
        now = self.now()
        header = PoseStamped().header
        header.frame_id, header.stamp = 'base_link', self.get_clock().now().to_msg()
        fresh = 0 <= now - self.scene_time <= 0.3
        ranges = [self.pose.local(p) for p, t in self.range_points.values()
                  if self.pose is not None and 0 <= now - t <= 0.3]
        healthy_ranges = sum(0 <= now - t <= 0.3 for t in self.range_health.values())
        if self.get_parameter('require_ultrasonics').value and healthy_ranges != 4:
            fresh = False
        camera = ([self.pose.local(self.camera_pose.world(p)) for p in self.camera_points]
                  if self.pose is not None and self.camera_pose is not None else [])
        self.points = fuse_points(camera, ranges) if fresh else []
        origin = (self.pose.local(self.camera_pose.world(self.camera_origin))
                  if self.pose is not None and self.camera_pose is not None else Point(0, 0))
        self.map.update(camera if fresh else [], origin)
        mark_memory_occupied(self.map, ranges)
        if fresh and self.pose and 0 <= now - self.speed_time <= 0.2:
            memory = self.memory.snapshot(self.pose, now)
            mark_memory_occupied(self.map, memory)
        grid = OccupancyGrid()
        grid.header = header
        grid.info.resolution = self.map.resolution
        grid.info.width, grid.info.height = self.map.width, self.map.height
        grid.info.origin.position.x, grid.info.origin.position.y = self.map.origin.x, self.map.origin.y
        grid.info.origin.orientation.w = 1.0
        grid.data = self.map.data if fresh else [-1] * len(self.map.data)
        self.map_pub.publish(grid)
        if fresh and self.pose and self.scene_time > self.last_mapped_scene:
            self.evidence.integrate(self.map, self.pose, self.scene_time)
            self.last_mapped_scene = self.scene_time
        if self.pose:
            history = self.evidence.snapshot(self.pose, now)
            history_msg = OccupancyGrid()
            history_msg.header.stamp, history_msg.header.frame_id = header.stamp, 'odom'
            history_msg.info.resolution = history.resolution
            history_msg.info.width, history_msg.info.height = history.width, history.height
            history_msg.info.origin.position.x, history_msg.info.origin.position.y = history.origin.x, history.origin.y
            history_msg.info.origin.orientation.w = 1.0
            history_msg.data = history.data
            self.history_pub.publish(history_msg)
        scene = scene_summary(self.points) if fresh else {'obstacle_count': None}
        scene['fresh'] = fresh
        self.scene_pub.publish(String(data=json.dumps(scene, allow_nan=False)))
        obstacles = PoseArray()
        obstacles.header = header
        if fresh:
            for point in self.points:
                p = Pose()
                p.position.x, p.position.y, p.orientation.w = point.x, point.y, 1.0
                obstacles.poses.append(p)
        self.obstacle_pub.publish(obstacles)
        path = Path()
        path.header = header
        # Goals in a moving base frame expire; reissue from a localization-aware upstream node.
        route = []
        if (fresh and self.get_parameter('geometry_calibrated').value and self.goal and
                0 <= now - stamp_seconds(self.goal.header.stamp) <= 0.5):
            route = lattice_plan(self.map, Point(self.goal.pose.position.x, self.goal.pose.position.y), self.geometry)
            for point in route:
                p = PoseStamped()
                p.header = header
                p.pose.position.x, p.pose.position.y, p.pose.orientation.w = point.x, point.y, 1.0
                p.pose.orientation.z, p.pose.orientation.w = math.sin(point.yaw / 2), math.cos(point.yaw / 2)
                path.poses.append(p)
        self.path_pub.publish(path)
        if self.source == 'controller':
            self.candidate = ActuatorCommand()
            self.candidate.throttle, self.candidate.brake, self.candidate.steering = pursuit(route, self.geometry)
            self.command_time = now
        ready = (self.status is not None and self.candidate is not None and
                 bool(path.poses) and
                 self.pose is not None and
                 self.speed is not None and 0 <= now - self.speed_time <= 0.2 and
                 0 <= now - self.status_time <= 0.3 and 0 <= now - self.command_time <= 0.2)
        result = safe_command(self.candidate.throttle if ready else 0.0,
                              self.candidate.steering if ready else 0.0,
                              self.speed if ready else 0.0, self.points, now, self.scene_time,
                              ready and self.status.link_ok,
                              ready and self.status.armed,
                              estop=bool(ready and (self.status.faults or self.status.state != 'ARMED')))
        # A candidate brake always overrides throttle, including invalid brake inputs.
        if ready and (not math.isfinite(self.candidate.brake) or self.candidate.brake != 0):
            result.update(throttle=0.0, brake=1.0, steering=0.0, reason='candidate_brake')
        if result['reason'] == 'ok' and not command_sweep_clear(self.map, self.speed, result['steering'], self.geometry):
            result.update(throttle=0.0, brake=1.0, steering=0.0, reason='swept_collision_or_unknown')
        if any(0 <= now - t <= 0.3 for t in self.range_hazards.values()):
            result.update(throttle=0.0, brake=1.0, steering=0.0, reason='ultrasonic_too_close')
        cmd = ActuatorCommand()
        cmd.header = header
        cmd.throttle, cmd.brake, cmd.steering = result['throttle'], result['brake'], result['steering']
        self.command_pub.publish(cmd)
        self.safety_pub.publish(String(data=result['reason']))


def main():
    rclpy.init()
    node = Pipeline()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
