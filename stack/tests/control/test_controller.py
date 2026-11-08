import math

import pytest

from cheesestack.core import Header, MessageBus, Pose2D, SimClock, Trajectory, TrajectoryPoint
from cheesestack.control import (
    ControlNode, ControllerMode, TrajectoryController, TrajectoryControllerConfig, curvature_to_steer,
)
from _sim import circle_xy, state, step_kinematic, straight_path


def make(lateral="pure_pursuit"):
    c = TrajectoryController(config=TrajectoryControllerConfig(lateral=lateral))
    c.sm.enable(0.0)
    c.sm.request_engage(0.0)
    return c


def test_not_engaged_outputs_zero():
    c = TrajectoryController()
    c.set_trajectory(straight_path(v=5.0))
    cmd = c.compute(state(Pose2D(0, 1, 0), 5.0), 0.05)
    assert cmd.throttle == 0 and cmd.steer_wheel_deg == 0 and cmd.source == "control:off"


@pytest.mark.parametrize("lateral", ["pure_pursuit", "stanley", "rear_wheel", "lqr"])
def test_closed_loop_on_arc_with_speed(lateral):
    c = make(lateral)
    xy, th = circle_xy(40.0, n=800, arc=math.pi)
    pts = [TrajectoryPoint(0.0, float(x), float(y), float(a), 8.0, 0.0, 1 / 40.0) for (x, y), a in zip(xy, th)]
    c.set_trajectory(Trajectory(Header(0.0, "map"), pts))
    pose, v, dt = Pose2D(0.0, 1.0, 0.0), 6.0, 0.05
    k_actual = 0.0
    errs = []
    for i in range(240):
        cmd = c.compute(state(pose, v, i * dt, yaw_rate=v * k_actual, curvature=k_actual), dt)
        k_actual = -cmd.curvature  # app sign (+right) -> stack (+left)
        pose = step_kinematic(pose, v, k_actual, dt)
        v += (cmd.throttle * 2.0 - cmd.brake * 6.5) * dt * 0.5
        if i > 140:
            errs.append(abs(c.last_debug["cross_track"]))
    assert max(errs) < 0.5
    assert cmd.steer_wheel_deg < 0  # left turn -> negative in app convention
    assert cmd.target_speed_mps == pytest.approx(8.0)


def test_node_publishes_command():
    bus, clock = MessageBus(), SimClock()
    node = ControlNode(make(), bus, clock)
    bus.publish(ControlNode.TRAJ_TOPIC, straight_path(v=5.0))
    bus.publish(ControlNode.STATE_TOPIC, state(Pose2D(0, 0.5, 0), 3.0))
    node.tick()
    cmd = bus.latest(ControlNode.CMD_TOPIC)
    assert cmd is not None and cmd.throttle > 0 and cmd.steer_wheel_deg > 0  # left of path -> steer right (+)
