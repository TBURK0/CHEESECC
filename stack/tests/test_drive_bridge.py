import math

from cheesestack.core import Pose2D
from cheesestack.comms import FrameDecoder
from cheesestack.pipeline import DriveBridge
from cheesestack.safety import WatchdogState


APP_CMD = {
    "id": 7,
    "steerWheelDeg": 12.0,
    "curvature": 0.01,
    "targetSpeedMps": 8.0,
    "throttle": 0.2,
    "brake": 0.0,
    "lookaheadS": 0.5,
    "source": "app",
    "carId": "hyundai-elantra-2019",
    "latitude": 39.132,
    "longitude": -84.515,
    "fromLatitude": 39.131,
    "fromLongitude": -84.516,
}


def test_app_command_is_checked_and_published():
    bridge = DriveBridge()
    tick = bridge.accept(APP_CMD, t=0.2)
    assert tick.used_fallback is False
    assert tick.command.steer_wheel_deg == 12.0
    assert tick.watchdog == WatchdogState.OK
    names = [bridge.decoder.decode(frame).name for frame in tick.frames]
    assert names[:3] == ["SteeringCommand", "SpeedCommand", "DriveCommandMeta"]
    assert "DriveCommandGeo" in names
    echoed = bridge.received()
    assert len(echoed) == len(tick.frames)
    steering = FrameDecoder(bridge.database).decode(echoed[0])
    assert steering.valid
    assert steering.signals["SteerWheelDeg"] == pytest_approx(12.0)


def test_bad_command_stops_the_car():
    bridge = DriveBridge()
    tick = bridge.accept({**APP_CMD, "steerWheelDeg": float("nan"), "speedMps": 10.0}, t=0.2)
    assert tick.used_fallback is True
    assert tick.command.throttle == 0.0
    assert tick.command.brake > 0.0


def test_phone_route_reports_progress_and_lookahead():
    bridge = DriveBridge()
    route = bridge.set_route([(39.13, -84.52), (39.131, -84.52), (39.132, -84.52)])
    assert route.length > 100.0
    s, d, left = bridge.progress(39.131, -84.52)
    assert s == pytest_approx(route.length / 2, rel=0.05)
    assert abs(d) < 1.0
    assert left == pytest_approx(route.length - s, abs=1.0)
    ahead = bridge.lookahead_point(s, 20.0)
    assert math.hypot(ahead[0], ahead[1]) > 0.0


def test_sim_scene_fills_the_occupancy_grid():
    bridge = DriveBridge()
    result = bridge.run_sim("straight_follow", duration=0.4, ego_speed=8.0, lead_speed=6.0, gap=20.0)
    assert result.steps > 0
    assert result.trace
    assert bridge.grid.occupied_mask().any()
    ego = result.trace[-1].state.pose
    assert isinstance(ego, Pose2D)


def pytest_approx(value, **kw):
    import pytest
    return pytest.approx(value, **kw)
