"""Run in a sourced ROS workspace; explicitly skipped on non-ROS hosts."""
import importlib.util
import time
import unittest


@unittest.skipUnless(importlib.util.find_spec('rclpy'), 'ROS 2 runtime is not installed')
class RosSmokeTests(unittest.TestCase):
    def test_default_node_publishes_brake_and_unknown_map(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from nav_msgs.msg import OccupancyGrid
        from vehicle_interfaces.msg import ActuatorCommand
        from autonomy.pipeline import Pipeline
        rclpy.init()
        node = observer = executor = None
        try:
            node = Pipeline()
            observer = rclpy.create_node('autonomy_smoke_observer')
            commands, maps = [], []
            observer.create_subscription(ActuatorCommand, 'vehicle/cmd', commands.append, 10)
            observer.create_subscription(OccupancyGrid, 'mapping/local', maps.append, 10)
            executor = SingleThreadedExecutor()
            executor.add_node(node)
            executor.add_node(observer)
            deadline = time.monotonic() + 3
            while (not commands or not maps) and time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.05)
            self.assertTrue(commands)
            self.assertTrue(maps)
            self.assertEqual(commands[-1].throttle, 0.0)
            self.assertEqual(commands[-1].brake, 1.0)
            self.assertTrue(all(value == -1 for value in maps[-1].data))
        finally:
            if executor:
                executor.shutdown()
            if observer:
                observer.destroy_node()
            if node:
                node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    unittest.main()
