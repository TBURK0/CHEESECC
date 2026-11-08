import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from vehicle_interfaces.msg import Scene, MotionProposal
from .core import Behavior, astar, pursuit


class PlannerNode(Node):
    def __init__(self):
        super().__init__('waypoint_planner')
        self.behavior, self.pose, self.scene = Behavior(), None, None
        self.route = astar((0, 0), (4, 0))
        self.pub = self.create_publisher(MotionProposal, '/planning/proposal', 10)
        self.path_pub = self.create_publisher(Path, '/planning/global_path', 10)
        self.local_pub = self.create_publisher(Path, '/planning/local_path', 10)
        self.create_subscription(Odometry, '/localization/odom', self.set_pose, 10)
        self.create_subscription(Scene, '/perception/scene', self.set_scene, 10)
        self.create_subscription(PoseStamped, '/mission/goal', self.goal, 10)
        self.create_timer(0.05, self.tick)

    def set_pose(self, m):
        self.pose = m

    def set_scene(self, m):
        self.scene = m

    def goal(self, m):
        try:
            if m.header.frame_id != 'map' or self.pose is None:
                raise ValueError('map goal and localization required')
            p = m.pose.position
            current = self.pose.pose.pose.position
            self.route = astar((round(current.x), round(current.y)), (round(p.x), round(p.y)))
            self.behavior = Behavior()
        except (ValueError, OverflowError):
            self.route = []

    def tick(self):
        now = self.get_clock().now()
        t = now.nanoseconds*1e-9
        fresh = lambda m: m is not None and 0 <= t-(m.header.stamp.sec+m.header.stamp.nanosec*1e-9) <= 0.3
        localized, seen = fresh(self.pose), fresh(self.scene)
        v, w, arrived = 0.0, 0.0, False
        if localized and self.route:
            p, q = self.pose.pose.pose.position, self.pose.pose.pose.orientation
            yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
            v, w = pursuit(p.x, p.y, yaw, self.route)
            arrived = math.hypot(p.x-self.route[-1][0], p.y-self.route[-1][1]) < 0.15
        allowed = self.behavior.update(t, localized and bool(self.route), bool(seen and self.scene.healthy),
            self.scene.signal_state if seen else 'UNKNOWN', bool(seen and self.scene.stop_sign),
            self.pose.twist.twist.linear.x if localized else 0, arrived)
        msg = MotionProposal()
        msg.header.stamp, msg.header.frame_id = now.to_msg(), 'base_link'
        msg.speed_mps, msg.yaw_rate_radps = float(v), float(w)
        msg.behavior_allows, msg.behavior_state = allowed, self.behavior.state
        self.pub.publish(msg)
        path = Path()
        path.header.stamp, path.header.frame_id = now.to_msg(), 'map'
        for x, y in self.route:
            p = PoseStamped()
            p.header = path.header
            p.pose.position.x, p.pose.position.y, p.pose.orientation.w = float(x), float(y), 1.0
            path.poses.append(p)
        self.path_pub.publish(path)
        # Reference segment, not a guarantee of collision-free swept trajectory.
        self.local_pub.publish(path)


def main():
    rclpy.init()
    n = PlannerNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown()
