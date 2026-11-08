"""Closed-loop simulation: scenario + twin + sensors + pluggable controller on a SimClock.

Plug-in point: any callable `controller(state: VehicleState, view: WorldView) -> ControlCommand`
(see `Controller`). Two adapters are provided:
  * `PurePursuitDriver` - built-in reference driver (pure-pursuit steering on the
    lane centre + IDM longitudinal with in-path object selection). Uses ground
    truth, so it is an "oracle" baseline for regression tests of the sim itself.
  * `BusController` - for the real stack: ticks a list of `core.Node`s that read
    sensor/state topics from the bus and publish a `ControlCommand` on a topic.

Each physics step (scenario.dt): sensors that are due publish on the bus; at the
control rate the ego `VehicleState` and ground truth are published, the
controller is called and its command latched into the twin (zero-order hold);
then the world steps and ego collisions are checked.

Bus topics: /sim/ego_state, /sim/ground_truth, /control/command, /sensors/*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence

import numpy as np

from cheesestack.core import ControlCommand, Header, MessageBus, Node, ObjectClass, SimClock, VehicleState

from .physics import ActorTruth, boxes_overlap
from .roads import Road
from .scenarios import Scenario
from .sensors import Sensor, TruthSnapshot, default_sensor_suite
from .vehicle_twin import TwinConfig, VehicleParams, VehicleTwin

EGO_STATE_TOPIC = "/sim/ego_state"
GROUND_TRUTH_TOPIC = "/sim/ground_truth"
COMMAND_TOPIC = "/control/command"


@dataclass
class WorldView:
    """What a controller may look at. `actors` excludes the ego. Real stacks should use `bus`."""

    t: float
    road: Road
    actors: list[ActorTruth]
    bus: MessageBus
    scenario: Scenario
    ego: ActorTruth
    vehicle_params: VehicleParams
    target_speed: float
    target_offset: float


class Controller(Protocol):
    def __call__(self, state: VehicleState, view: WorldView) -> ControlCommand: ...


# --------------------------------------------------------- reference driver
@dataclass
class PurePursuitConfig:
    lookahead_time: float = 0.9
    lookahead_min: float = 5.0
    time_headway: float = 1.4
    min_gap: float = 4.0
    comfort_decel: float = 3.0
    idm_delta: float = 4.0
    max_accel: float = 2.0
    emergency_ttc: float = 1.2
    corridor_margin: float = 0.3
    reaction_horizon: float = 4.0  # s, used to predict crossing objects into the corridor


class PurePursuitDriver:
    """Pure pursuit + IDM. Outputs pedals/steering-wheel and curvature/target speed."""

    def __init__(self, config: PurePursuitConfig = PurePursuitConfig()) -> None:
        self.config = config
        self.last_lead: Optional[str] = None

    def __call__(self, state: VehicleState, view: WorldView) -> ControlCommand:
        c, p, road = self.config, view.vehicle_params, view.road
        s_ego, _ = road.project(state.pose.x, state.pose.y)
        # ---- lateral: pure pursuit on target lane centre
        ld = max(c.lookahead_min, c.lookahead_time * state.v)
        target = road.pose_at(s_ego + ld, view.target_offset)
        local = state.pose.inverse().transform_point([target.x, target.y])
        dist = max(1e-3, float(np.hypot(*local)))
        kappa = 2.0 * local[1] / (dist * dist)
        delta = float(np.clip(math.atan(kappa * p.wheelbase), -p.max_steer, p.max_steer))
        # ---- longitudinal: IDM on in-path lead
        gap, v_lead, lead = self._lead(state, view, s_ego)
        v0 = max(0.1, view.target_speed)
        a_free = c.max_accel * (1.0 - (state.v / v0) ** c.idm_delta)
        a_des = a_free
        if lead is not None:
            dv = state.v - v_lead
            s_star = c.min_gap + max(0.0, state.v * c.time_headway + state.v * dv / (2 * math.sqrt(c.max_accel * c.comfort_decel)))
            a_des = c.max_accel * (1.0 - (state.v / v0) ** c.idm_delta - (s_star / max(gap, 0.1)) ** 2)
            if dv > 0 and gap / dv < c.emergency_ttc:
                a_des = -p.max_decel
        self.last_lead = lead
        a_des = float(np.clip(a_des, -p.max_decel, p.max_accel))
        throttle = max(0.0, a_des) / p.max_accel
        brake = max(0.0, -a_des) / p.max_decel
        if state.v < 0.05 and a_des <= 0:
            brake = max(brake, 0.3)  # hold at standstill
        return ControlCommand(
            header=Header(stamp=view.t, frame_id="base_link", source="sim.pure_pursuit"),
            steer_wheel_deg=math.degrees(delta * p.steering_ratio),
            throttle=throttle, brake=brake, curvature=kappa,
            target_speed_mps=max(0.0, state.v + a_des * 1.0), source="sim.pure_pursuit",
        )

    def _lead(self, state: VehicleState, view: WorldView, s_ego: float):
        c, p, road = self.config, view.vehicle_params, view.road
        front = p.length - p.rear_overhang
        best = (math.inf, 0.0, None)
        for a in view.actors:
            s_a, d_a = road.project(a.pose.x, a.pose.y)
            gap = s_a - s_ego - front - 0.5 * a.length
            if gap < -front:
                continue
            yaw = road.heading_at(s_a)
            n = np.array([-math.sin(yaw), math.cos(yaw)])
            t_hat = np.array([math.cos(yaw), math.sin(yaw)])
            v_long, v_lat = float(a.velocity @ t_hat), float(a.velocity @ n)
            half = 0.5 * (p.width + a.width) + c.corridor_margin
            t_reach = min(c.reaction_horizon, max(0.0, gap) / max(state.v, 0.5))
            d_pred = d_a + v_lat * t_reach
            lo, hi = sorted((d_a, d_pred))
            in_path = hi >= view.target_offset - half and lo <= view.target_offset + half
            if in_path and gap < best[0]:
                best = (gap, v_long, a.name)
        return best


class BusController:
    """Adapter for real stack nodes: tick `nodes`, then read the latest command from the bus.

    The runner has already published /sim/ego_state, /sim/ground_truth and sensor
    topics for this tick. If no command is available, a full-brake command is returned
    (fail-safe) and `missing_count` is incremented; the same happens when the latest
    command is older than `max_age` seconds (stalled planner/controller).
    """

    def __init__(self, nodes: Sequence[Node] = (), command_topic: str = COMMAND_TOPIC,
                 hook: Optional[Callable[[VehicleState, WorldView], None]] = None, max_age: float = 0.5) -> None:
        self.nodes, self.command_topic, self.hook = list(nodes), command_topic, hook
        self.max_age = max_age
        self.missing_count = 0

    def __call__(self, state: VehicleState, view: WorldView) -> ControlCommand:
        if self.hook is not None:
            self.hook(state, view)
        for n in self.nodes:
            n.tick()
        cmd = view.bus.latest(self.command_topic)
        if cmd is None or view.t - cmd.header.stamp > self.max_age:
            self.missing_count += 1
            return ControlCommand(Header(view.t, source="sim.bus_controller"), brake=1.0, source="sim.failsafe")
        return cmd


# -------------------------------------------------------------------- runner
@dataclass
class ClosedLoopConfig:
    control_rate_hz: float = 20.0
    enable_sensors: bool = True
    publish: bool = True
    publish_command: bool = True
    stop_on_collision: bool = True
    duration: Optional[float] = None  # override scenario.duration


@dataclass
class TraceSample:
    t: float
    state: VehicleState
    ego: ActorTruth
    command: ControlCommand
    actors: list[ActorTruth]
    s: float
    d: float
    lateral_error: float


@dataclass
class SimResult:
    scenario: str
    params: dict
    trace: list[TraceSample] = field(default_factory=list)
    collided: bool = False
    collision_with: Optional[str] = None
    collision_cls: Optional[ObjectClass] = None
    sim_time: float = 0.0
    steps: int = 0
    vehicle_params: VehicleParams = field(default_factory=VehicleParams)

    def series(self, name: str) -> np.ndarray:
        """Time series by name: t, v, a, yaw_rate, steer, s, d, lateral_error, x, y."""
        get = {
            "t": lambda x: x.t, "v": lambda x: x.state.v, "a": lambda x: x.state.a,
            "yaw_rate": lambda x: x.state.yaw_rate, "steer": lambda x: x.state.steer,
            "s": lambda x: x.s, "d": lambda x: x.d, "lateral_error": lambda x: x.lateral_error,
            "x": lambda x: x.state.pose.x, "y": lambda x: x.state.pose.y,
        }[name]
        return np.array([get(x) for x in self.trace])


class ClosedLoopRunner:
    def __init__(self, scenario: Scenario, controller: Controller,
                 config: ClosedLoopConfig = ClosedLoopConfig(),
                 vehicle_params: VehicleParams = VehicleParams(), twin_config: TwinConfig = TwinConfig(),
                 bus: Optional[MessageBus] = None, clock: Optional[SimClock] = None,
                 sensors: Optional[list[Sensor]] = None,
                 hooks: Sequence[Callable[["ClosedLoopRunner", TraceSample], None]] = ()) -> None:
        self.scenario, self.controller, self.config = scenario, controller, config
        self.vehicle_params, self.twin_config = vehicle_params, twin_config
        self.clock = clock or getattr(bus, "clock", None) or SimClock()
        self.bus = bus if bus is not None else MessageBus()
        self.world, self.twin = scenario.build_world(vehicle_params, twin_config, self.clock)
        self.sensors = sensors if sensors is not None else (
            default_sensor_suite(self.world.rng) if config.enable_sensors else [])
        self.hooks = list(hooks)

    def _snapshot(self, t: float) -> TruthSnapshot:
        return TruthSnapshot(t, self.twin.vehicle_state(t), self.twin.accel_body.copy(),
                             self.world.truths(exclude=[self.twin.name]), self.vehicle_params.track)

    def run(self) -> SimResult:
        sc, cfg, w, twin, bus = self.scenario, self.config, self.world, self.twin, self.bus
        duration = cfg.duration if cfg.duration is not None else sc.duration
        n_steps = int(round(duration / sc.dt))
        ctrl_every = max(1, int(round(1.0 / (cfg.control_rate_hz * sc.dt))))
        result = SimResult(sc.name, dict(sc.params), vehicle_params=self.vehicle_params)
        d_target = sc.ego_d()
        cmd = ControlCommand(Header(self.clock.now()))
        for k in range(n_steps):
            t = self.clock.now()
            if hasattr(bus, "pump"):
                bus.pump(t)
            snap = self._snapshot(t)
            for sensor in self.sensors:
                msg = sensor.sample(snap)
                if msg is not None:
                    bus.publish(sensor.topic, msg)
            if k % ctrl_every == 0:
                state = snap.ego
                ego_truth = twin.truth()
                if cfg.publish:
                    bus.publish(EGO_STATE_TOPIC, state)
                    bus.publish(GROUND_TRUTH_TOPIC, snap.actors)
                view = WorldView(t, sc.road, snap.actors, bus, sc, ego_truth, self.vehicle_params,
                                 sc.ego.target_speed, d_target)
                cmd = self.controller(state, view)
                if cfg.publish and cfg.publish_command and not isinstance(self.controller, BusController):
                    bus.publish(COMMAND_TOPIC, cmd)
                twin.apply(cmd)
                s, d = sc.road.project(state.pose.x, state.pose.y)
                sample = TraceSample(t, state, ego_truth, cmd, snap.actors, s, d, d - d_target)
                result.trace.append(sample)
                for h in self.hooks:
                    h(self, sample)
            w.step()
            result.steps += 1
            hit = self._check_collision()
            if hit is not None:
                result.collided, result.collision_with, result.collision_cls = True, hit.name, hit.cls
                if cfg.stop_on_collision:
                    break
        result.sim_time = self.clock.now()
        return result

    def _check_collision(self) -> Optional[ActorTruth]:
        ego = self.twin.truth()
        for a in self.world.truths(exclude=[self.twin.name]):
            if boxes_overlap(ego, a):
                return a
        return None


def run_scenario(scenario: Scenario, controller: Optional[Controller] = None, **kw) -> SimResult:
    """One-liner: run a scenario with the reference driver (or a given controller)."""
    return ClosedLoopRunner(scenario, controller or PurePursuitDriver(), **kw).run()
