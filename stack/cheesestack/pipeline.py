"""One drive tick that the phone and the autonomy stack share.

The app sends a DriveCommand (steer, speed, lookahead, and optional GPS).
This module checks it with the safety layer, puts a minimal-risk stop on the
bus when the command is not usable, encodes the CHEESECC CAN frames, projects
the phone's route into the mapping layer, and can drop a simulated scene into
the occupancy grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from cheesestack.comms import (
    DriveCommandCodec,
    FrameDecoder,
    VirtualBus,
    VirtualBusConfig,
    VirtualCanNetwork,
    load_default_database,
)
from cheesestack.core import BoundingBox3D, ControlCommand, Detection, Header, ObjectClass, Pose2D
from cheesestack.localization.gps import GeoPoint, equirectangular_to_local
from cheesestack.mapping.route import Route
from cheesestack.mapping.topology import Lane, LaneMap, Polyline
from cheesestack.perception.occupancy_grid import OccupancyGrid, OccupancyGridConfig
from cheesestack.safety import InputValidator, MinimalRiskManeuver, Watchdog, WatchdogState
from cheesestack.sim import ClosedLoopConfig, SimResult, make_scenario, run_scenario


@dataclass
class DriveTick:
    """What one app command became after safety and encoding."""

    command: ControlCommand
    frames: list
    used_fallback: bool
    watchdog: WatchdogState
    faults: list = field(default_factory=list)


class DriveBridge:
    """Safety-checked path from an app DriveCommand dict onto the CAN bus."""

    def __init__(self, car_id: str = "hyundai-elantra-2019", watchdog_timeout_s: float = 1.0) -> None:
        self.car_id = car_id
        self.database = load_default_database()
        self.codec = DriveCommandCodec(self.database)
        self.decoder = FrameDecoder(self.database)
        self.validator = InputValidator()
        self.fallback = MinimalRiskManeuver()
        self.watchdog = Watchdog()
        self.network = VirtualCanNetwork()
        self.bus = VirtualBus(self.network, VirtualBusConfig(receive_own_messages=True))
        self.grid = OccupancyGrid(OccupancyGridConfig(size_m=40.0, resolution=0.25))
        self.route: Optional[Route] = None
        self._origin: Optional[GeoPoint] = None
        self._prev: Optional[ControlCommand] = None
        self.watchdog.watch_named("drive", timeout_s=watchdog_timeout_s, t=0.0)

    def set_route(self, lat_lon: Sequence[Sequence[float]]) -> Route:
        """Build a one-lane map route from the phone's polyline. Points are (lat, lon) degrees."""
        pts = [(float(p[0]), float(p[1])) for p in lat_lon]
        if len(pts) < 2:
            raise ValueError("a route needs at least two coordinates")
        origin = GeoPoint(pts[0][0], pts[0][1])
        local = [equirectangular_to_local(lat, lon, origin) for lat, lon in pts]
        lane = Lane("route", Polyline(np.asarray(local, dtype=float)))
        lane_map = LaneMap([lane], name="phone-route")
        self._origin = origin
        self.route = Route.from_lane_ids(lane_map, ["route"])
        return self.route

    def progress(self, lat: float, lon: float) -> tuple[float, float, float]:
        """Return station, lateral offset, and remaining metres for a GPS fix on the route."""
        if self.route is None or self._origin is None:
            raise RuntimeError("set a route before asking for progress")
        east, north = equirectangular_to_local(lat, lon, self._origin)
        s, d = self.route.to_frenet(float(east), float(north))
        return s, d, self.route.remaining_distance(s)

    def lookahead_point(self, s: float, horizon_m: float) -> np.ndarray:
        if self.route is None:
            raise RuntimeError("set a route before asking for a lookahead point")
        return self.route.from_frenet(min(self.route.length, max(0.0, s) + horizon_m), 0.0)

    def accept(self, cmd: dict, t: float) -> DriveTick:
        """Validate one app command and publish its CAN frames. Bad commands become a stop."""
        proposed = _control_from_app(cmd, t)
        result = self.validator.validate_command(proposed, self._prev, t)
        used_fallback = not result.ok
        if used_fallback:
            speed = _finite_or(cmd.get("speedMps", cmd.get("targetSpeedMps", 0.0)), 0.0)
            hold = proposed.steer_wheel_deg if math.isfinite(proposed.steer_wheel_deg) else 0.0
            if not self.fallback.active:
                self.fallback.start(t, speed)
            # The jerk ramp is flat at the start instant. Publish the first ramp sample
            # so a rejected command already asks for brake.
            command = self.fallback.step(t + self.fallback.config.plan_dt_s, speed, hold)
        else:
            command = result.value
        self._prev = command
        self.watchdog.kick("drive", t)
        faults = list(result.faults) + self.watchdog.check(t)
        payload = _app_payload(cmd, command, self.car_id)
        frames = self.codec.encode_drive_command(payload, t, drive_mode="Fault" if used_fallback else "Auto")
        for frame in frames:
            self.bus.send(frame)
        return DriveTick(command, frames, used_fallback, self.watchdog.state("drive"), faults)

    def received(self) -> list:
        frames = []
        while True:
            frame = self.bus.recv()
            if frame is None:
                break
            frames.append(frame)
        return frames

    def observe_actors(self, ego: Pose2D, actors: Sequence, stamp: float = 0.0) -> int:
        """Insert simulated or tracked actors into the ego occupancy grid. Returns how many landed."""
        detections = []
        for actor in actors:
            pose = getattr(actor, "pose", None)
            if pose is None:
                continue
            dx, dy = pose.x - ego.x, pose.y - ego.y
            c, s = math.cos(ego.yaw), math.sin(ego.yaw)
            body = Pose2D(c * dx + s * dy, -s * dx + c * dy, pose.yaw - ego.yaw)
            length = float(getattr(actor, "length", 4.5))
            width = float(getattr(actor, "width", 1.8))
            cls = getattr(actor, "cls", ObjectClass.CAR)
            detections.append(Detection(
                Header(stamp, source="pipeline.scene"),
                cls,
                1.0,
                box3d=BoundingBox3D(body, length, width, float(getattr(actor, "height", 1.5))),
            ))
        if detections:
            self.grid.insert_detections(detections)
        return len(detections)

    def run_sim(self, name: str = "straight_follow", duration: float = 2.0, **params) -> SimResult:
        """Run a built-in closed-loop scenario, then mark the last scene in the occupancy grid."""
        scenario = make_scenario(name, duration=duration, **params)
        result = run_scenario(scenario, config=ClosedLoopConfig(duration=duration))
        if result.trace:
            last = result.trace[-1]
            self.observe_actors(last.state.pose, last.actors, stamp=last.t)
        return result


def _finite_or(value, default):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _control_from_app(cmd: dict, t: float) -> ControlCommand:
    return ControlCommand(
        header=Header(stamp=float(t), seq=int(cmd.get("id", 0) or 0), source="app"),
        steer_wheel_deg=_finite_or(cmd.get("steerWheelDeg", 0.0), float("nan")),
        throttle=_finite_or(cmd.get("throttle", 0.0), float("nan")),
        brake=_finite_or(cmd.get("brake", 0.0), float("nan")),
        curvature=_finite_or(cmd.get("curvature", 0.0), float("nan")),
        target_speed_mps=_finite_or(cmd.get("targetSpeedMps", 0.0), float("nan")),
        source=str(cmd.get("source", "app")),
    )


def _app_payload(cmd: dict, command: ControlCommand, car_id: str) -> dict:
    payload = {
        "id": int(cmd.get("id", 0) or 0),
        "steerWheelDeg": command.steer_wheel_deg,
        "curvature": command.curvature,
        "targetSpeedMps": command.target_speed_mps,
        "throttle": command.throttle,
        "brake": command.brake,
        "lookaheadS": _finite_or(cmd.get("lookaheadS", 0.0), 0.0),
        "source": cmd.get("source", "app"),
        "carId": cmd.get("carId", car_id),
    }
    for key in ("latitude", "longitude", "fromLatitude", "fromLongitude"):
        if key in cmd:
            payload[key] = cmd[key]
    return payload
