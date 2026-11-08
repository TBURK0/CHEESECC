import json
import math
import unittest
import struct
from types import SimpleNamespace

from autonomy.comms import AuthenticatedInbox, sign_command
from autonomy.core import LocalMap, Point
from autonomy.motion import (VehicleGeometry, advance, command_sweep_clear,
                             footprint_clear, lattice_plan, pursuit, swept_clear)
from autonomy.semantic import BoxTracker, detection_position
from autonomy.depth_io import decode_depth
from autonomy.mapping import EvidenceMap
from autonomy.odometry import DeadReckoner
from autonomy.physics_sim import run as run_physics
from autonomy.spatial import (ObstacleMemory, Pose2, RigidTransform, calibrated_depth,
                              fuse_points, ultrasonic_endpoint)


class SpatialTests(unittest.TestCase):
    def test_quaternion_transform_and_pose_inverse(self):
        tf = RigidTransform((1, 2, 3), (0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)))
        x, y, z = tf.apply((1, 0, 0))
        self.assertAlmostEqual(x, 1)
        self.assertAlmostEqual(y, 3)
        self.assertAlmostEqual(z, 3)
        pose = Pose2(3, -1, 0.7)
        p = pose.local(pose.world(Point(2, 4)))
        self.assertAlmostEqual(p.x, 2)
        self.assertAlmostEqual(p.y, 4)
        with self.assertRaises(ValueError):
            RigidTransform((0, 0, 0), (0, 0, 0, 0))

    def test_calibrated_depth(self):
        # Optical -> base: z -> x, x -> -y, y -> -z.
        tf = RigidTransform((0.1, 0, 0.2), (-0.5, 0.5, -0.5, 0.5))
        p = calibrated_depth([[1000]], 100, 100, 0, 0, 0.001, tf)[0]
        self.assertAlmostEqual(p.x, 1.1)
        self.assertAlmostEqual(p.y, 0)

    def test_ultrasonic_invalid_and_transformed(self):
        tf = RigidTransform((0.2, 0.1, 0), (0, 0, 0, 1))
        self.assertEqual(ultrasonic_endpoint(1, 0.02, 4, tf), Point(1.2, 0.1))
        for distance in (math.nan, math.inf, -math.inf, 5, 0):
            self.assertIsNone(ultrasonic_endpoint(distance, 0.02, 4, tf))
        fused = fuse_points([Point(2, 0)], [Point(0.2, 0)])
        self.assertEqual(len(fused), 2)

    def test_ego_motion_memory_expiry_and_clock_reset(self):
        memory = ObstacleMemory()
        memory.update([Point(1, 0)], Pose2(), 1)
        points = memory.update([], Pose2(0.2, 0, 0), 1.1)
        self.assertAlmostEqual(points[0].x, 0.8)
        self.assertEqual(memory.update([], Pose2(), 1.6), [])
        memory.update([Point(1, 0)], Pose2(), 2)
        self.assertEqual(memory.update([], Pose2(), 1), [])


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.grid = LocalMap()
        self.grid.data = [0] * len(self.grid.data)
        self.geometry = VehicleGeometry()

    def test_bicycle_curvature_and_pursuit(self):
        pose = advance(Pose2(), 1, 0.3, self.geometry)
        self.assertGreater(pose.y, 0)
        self.assertAlmostEqual(pose.yaw, math.tan(0.3) / self.geometry.wheelbase)
        throttle, brake, steer = pursuit([Pose2(), Pose2(1, 0.5, 0)], self.geometry)
        self.assertGreater(steer, 0)
        self.assertGreater(throttle, 0)
        self.assertEqual(brake, 0)
        self.assertEqual(pursuit([], self.geometry), (0, 1, 0))

    def test_unknown_footprint_stops(self):
        self.assertTrue(footprint_clear(self.grid, Pose2(), self.geometry))
        self.grid.data = [-1] * len(self.grid.data)
        self.assertFalse(command_sweep_clear(self.grid, 0.1, 0, self.geometry))
        self.assertEqual(lattice_plan(self.grid, Point(1, 0), self.geometry), [])

    def test_swept_collision_between_endpoints(self):
        c = self.grid.cell(Point(0.8, 0))
        self.grid.data[c[1] * self.grid.width + c[0]] = 100
        self.assertTrue(footprint_clear(self.grid, Pose2(), self.geometry))
        self.assertTrue(footprint_clear(self.grid, Pose2(1.5, 0, 0), self.geometry))
        self.assertFalse(swept_clear(self.grid, Pose2(), 1.5, 0, self.geometry))

    def test_lattice_car_route(self):
        route = lattice_plan(self.grid, Point(1.2, 0.4), self.geometry, max_expansions=400)
        self.assertTrue(route)
        self.assertLess(math.hypot(route[-1].x - 1.2, route[-1].y - 0.4), 0.15)
        for a, b in zip(route, route[1:]):
            self.assertLessEqual(abs(b.yaw - a.yaw), 0.2 * math.tan(self.geometry.max_steer) / self.geometry.wheelbase + 1e-8)


class AuthenticationTests(unittest.TestCase):
    def setUp(self):
        # Ephemeral test key only; never persisted or used by a running receiver.
        import secrets
        self.key = secrets.token_bytes(32)
        self.receiver = AuthenticatedInbox(self.key, 'driver')
        self.payload = dict(seq=1, stamp=1.0, throttle=0.1, steering=0.0)

    def signed(self, **changes):
        return sign_command(dict(self.payload, **changes), self.key, self.receiver.session, 'driver')

    def test_valid_and_replay(self):
        line = self.signed()
        self.assertEqual(self.receiver.receive(line, 1)['throttle'], 0.1)
        with self.assertRaises(ValueError):
            self.receiver.receive(line, 1)

    def test_tamper_operator_session_and_expiry(self):
        tampered = json.loads(self.signed())
        tampered['command']['throttle'] = 0.2
        lines = [json.dumps(tampered), self.signed(stamp=0),
                 sign_command(self.payload, self.key, 'other-session', 'driver'),
                 sign_command(self.payload, self.key, self.receiver.session, 'other-operator'),
                 '[]', '{', 'x' * 2049]
        for line in lines:
            with self.assertRaises(ValueError):
                self.receiver.receive(line, 1)
        self.receiver.receive(self.signed(), 1)

    def test_restart_invalidates_previous_session(self):
        receiver = AuthenticatedInbox(self.key, 'driver')
        with self.assertRaises(ValueError):
            receiver.receive(self.signed(), 1)
        with self.assertRaises(ValueError):
            AuthenticatedInbox(b'short', 'driver')


class SemanticTests(unittest.TestCase):
    def detection(self, x=0, name='stop sign'):
        return dict(name=name, confidence=0.9, xyxy=[x, 0, x + 10, 10])

    def test_tracking_class_identity_expiry_and_unique_assignment(self):
        tracker = BoxTracker()
        first = tracker.update([self.detection()], 1)[0]['track_id']
        result = tracker.update([self.detection(1), self.detection(2)], 1.1)
        self.assertEqual(result[0]['track_id'], first)
        self.assertNotEqual(result[1]['track_id'], first)
        changed = tracker.update([self.detection(1, 'traffic light')], 1.2)[0]['track_id']
        self.assertNotEqual(changed, first)
        expired = tracker.update([self.detection()], 2)[0]['track_id']
        self.assertNotEqual(expired, first)

    def test_depth_median_and_missing_depth(self):
        tf = RigidTransform((0, 0, 0), (0, 0, 0, 1))
        depth = [[1000] * 10 for _ in range(10)]
        depth[5][5] = 4000
        position = detection_position([0, 0, 10, 10], depth, 0.001, (100, 100, 5, 5), tf)
        self.assertEqual(position, (0, 0, 1))
        self.assertIsNone(detection_position([0, 0, 10, 10], [[0] * 10] * 10, 0.001, (100, 100, 5, 5), tf))


class IntegrationCoreTests(unittest.TestCase):
    def test_depth_endianness_row_padding_and_truncation(self):
        msg = SimpleNamespace(encoding='16UC1', width=2, height=2, step=6, is_bigendian=True,
                              data=struct.pack('>HHH', 100, 200, 0) + struct.pack('>HHH', 300, 400, 0))
        rows, scale = decode_depth(msg)
        self.assertEqual(rows, [(100, 200), (300, 400)])
        self.assertEqual(scale, 0.001)
        msg.data = b'\x00'
        with self.assertRaises(ValueError):
            decode_depth(msg)
        msg = SimpleNamespace(encoding='32FC1', width=1, height=1, step=4, is_bigendian=False,
                              data=struct.pack('<f', 1.5))
        self.assertEqual(decode_depth(msg), ([(1.5,)], 1.0))

    def test_dead_reckoning_straight_turn_and_gap(self):
        odom = DeadReckoner()
        odom.update(1, 0, 0)
        pose = odom.update(1, 0, 0.1)
        self.assertAlmostEqual(pose.x, 0.1)
        pose = odom.update(1, 1, 0.2)
        self.assertGreater(pose.y, 0)
        self.assertAlmostEqual(pose.yaw, 0.1)
        with self.assertRaises(ValueError):
            odom.update(1, 0, 1)

    def test_mapping_world_alignment_expiry_and_observed_clear(self):
        local = LocalMap()
        local.update([Point(1, 0)])
        memory = EvidenceMap(ttl=1)
        memory.integrate(local, Pose2(1, 0, 0), 0)
        snapshot = memory.snapshot(Pose2(), 0.1)
        c = snapshot.cell(Point(2.05, 0.05))
        self.assertEqual(snapshot.data[c[1] * snapshot.width + c[0]], 100)
        self.assertTrue(all(v == -1 for v in memory.snapshot(Pose2(), 1.1).data))
        local.data = [0] * len(local.data)
        memory.integrate(local, Pose2(1, 0, 0), 0.2)
        snapshot = memory.snapshot(Pose2(), 0.2)
        c = snapshot.cell(Point(2.05, 0.05))
        self.assertEqual(snapshot.data[c[1] * snapshot.width + c[0]], 0)

    def test_closed_loop_physics(self):
        result = run_physics()
        self.assertTrue(result['passed'], result['checks'])

    def test_camera_origin_does_not_clear_behind_mount(self):
        grid = LocalMap()
        grid.update([Point(1, 0)], origin=Point(0.4, 0))
        x, y = grid.cell(Point(0.1, 0))
        self.assertEqual(grid.data[y * grid.width + x], -1)


if __name__ == '__main__':
    unittest.main()
