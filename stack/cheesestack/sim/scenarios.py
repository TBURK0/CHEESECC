"""Scenario generation: road + ego start + scripted actors, built-ins and parametric variation.

Scripted actors are `FrenetActor`s: point-mass state (s, d, v) on a `Road`
driven by a `Behavior` = longitudinal profile (-> accel) + lateral profile
(-> lateral velocity). Profiles receive (t, actor, world) so they may trigger
on time or on world state (e.g. ego gap).

Built-ins (all parameterised, SI units):
  straight_follow, cut_in, hard_brake_lead, curve, pedestrian_crossing

Parametric variation: `ScenarioVariation(factory, base, ranges)` samples or
grids the factory's keyword parameters deterministically from a seed.

# TODO(sim): OpenSCENARIO import, reactive (IDM/MOBIL) traffic agents,
# map-based routes across junctions, trajectory-replay actors from logs.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np

from cheesestack.core import ObjectClass, Pose2D, SimClock

from .physics import ActorTruth, World, WorldConfig, rk4_step  # noqa: F401
from .roads import Road
from .vehicle_twin import TwinConfig, VehicleParams, VehicleTwin


# ---------------------------------------------------------------- behaviours
class LongitudinalProfile(Protocol):
    def __call__(self, t: float, actor: "FrenetActor", world: Optional[World]) -> float: ...


class LateralProfile(Protocol):
    def __call__(self, t: float, actor: "FrenetActor", world: Optional[World]) -> float: ...


@dataclass
class HoldSpeed:
    """P-control to a constant speed with accel/decel limits."""

    speed: float
    kp: float = 1.5
    max_accel: float = 2.5
    max_decel: float = 6.0

    def __call__(self, t, actor, world) -> float:
        return float(np.clip(self.kp * (self.speed - actor.v), -self.max_decel, self.max_accel))


@dataclass
class SpeedSchedule:
    """Track a piecewise-linear target speed over time [(t, v), ...]."""

    points: list[tuple[float, float]]
    kp: float = 1.5
    max_accel: float = 3.0
    max_decel: float = 8.0

    def __call__(self, t, actor, world) -> float:
        ts, vs = zip(*self.points)
        v_ref = float(np.interp(t, ts, vs))
        return float(np.clip(self.kp * (v_ref - actor.v), -self.max_decel, self.max_accel))


@dataclass
class BrakeAt:
    """Follow `base` until `t_start`, then brake at `decel` to a stop."""

    t_start: float
    decel: float
    base: LongitudinalProfile

    def __call__(self, t, actor, world) -> float:
        if t < self.t_start:
            return self.base(t, actor, world)
        return -self.decel if actor.v > 0.0 else 0.0


@dataclass
class Stationary:
    def __call__(self, t, actor, world) -> float:
        return -actor.v / 0.5 if actor.v > 0 else 0.0


@dataclass
class KeepOffset:
    """Hold the actor's lateral offset at `d` (None -> its initial offset)."""

    d: Optional[float] = None
    kp: float = 1.0

    def __call__(self, t, actor, world) -> float:
        target = actor.d0 if self.d is None else self.d
        return self.kp * (target - actor.d)


def _smoothstep5(x: float) -> tuple[float, float]:
    """Quintic smoothstep value and derivative on [0, 1] (zero vel/accel at ends)."""
    x = min(max(x, 0.0), 1.0)
    return x**3 * (10 - 15 * x + 6 * x * x), 30 * x * x * (1 - x) ** 2


@dataclass
class LaneChange:
    """Quintic lateral move to `d_target` over `duration`.

    Starts at `t_start`, or when `trigger(t, actor, world)` first returns True.
    """

    d_target: float
    duration: float = 3.0
    t_start: float = 0.0
    trigger: Optional[Callable] = None
    kp: float = 2.0
    _t0: Optional[float] = field(default=None, repr=False)
    _d_from: float = field(default=0.0, repr=False)

    def __call__(self, t, actor, world) -> float:
        if self._t0 is None:
            fire = self.trigger(t, actor, world) if self.trigger is not None else t >= self.t_start
            if not fire:
                return KeepOffset()(t, actor, world)
            self._t0, self._d_from = t, actor.d
        x = (t - self._t0) / self.duration
        val, der = _smoothstep5(x)
        d_ref = self._d_from + (self.d_target - self._d_from) * val
        vd_ref = (self.d_target - self._d_from) * der / self.duration
        return vd_ref + self.kp * (d_ref - actor.d)


@dataclass
class Cross:
    """Walk laterally at `speed` toward `d_end`, starting at `t_start`."""

    d_end: float
    speed: float = 1.4
    t_start: float = 0.0

    def __call__(self, t, actor, world) -> float:
        if t < self.t_start:
            return 0.0
        err = self.d_end - actor.d
        return math.copysign(min(self.speed, abs(err) / 0.2), err) if abs(err) > 1e-3 else 0.0


def gap_trigger(max_gap: float, ego_name: str = "ego") -> Callable:
    """Trigger that fires when the actor is less than `max_gap` metres ahead of the ego (in s)."""

    def trig(t, actor, world) -> bool:
        if world is None or ego_name not in {a.name for a in world.actors}:
            return False
        ego = world.get(ego_name).truth()
        s_ego, _ = actor.road.project(ego.pose.x, ego.pose.y)
        return 0.0 <= actor.s - s_ego <= max_gap

    return trig


@dataclass
class Behavior:
    longitudinal: LongitudinalProfile
    lateral: LateralProfile = field(default_factory=KeepOffset)

    def __call__(self, t, actor, world) -> tuple[float, float]:
        return float(self.longitudinal(t, actor, world)), float(self.lateral(t, actor, world))


# -------------------------------------------------------------------- actors
class FrenetActor:
    """Scripted actor moving in road Frenet coordinates (s, d) with speed v along s."""

    def __init__(self, name: str, road: Road, behavior: Behavior, s0: float, d0: float, v0: float = 0.0,
                 cls: ObjectClass = ObjectClass.CAR, length: float = 4.5, width: float = 1.8,
                 height: float = 1.5) -> None:
        self.name, self.road, self.behavior, self.cls = name, road, behavior, cls
        self.length, self.width, self.height = length, width, height
        self.state = np.array([s0, d0, v0], dtype=float)
        self.d0 = d0
        self.vd = 0.0
        self.accel = 0.0

    s = property(lambda self: float(self.state[0]))
    d = property(lambda self: float(self.state[1]))
    v = property(lambda self: float(self.state[2]))

    def step(self, t: float, dt: float, world: Optional[World] = None) -> None:
        self.accel, self.vd = self.behavior(t, self, world)
        a, vd = self.accel, self.vd

        # TODO(sim): ds/dt = v / (1 - kappa*d) for actors far from the reference line.
        def f(_t, x):
            return np.array([x[2], vd, a if (x[2] > 0 or a > 0) else 0.0])

        self.state = rk4_step(f, self.state, t, dt)
        self.state[2] = max(0.0, self.state[2])

    def truth(self) -> ActorTruth:
        ref = self.road.pose_at(self.s, self.d)
        v, vd = self.v, self.vd
        c, s = math.cos(ref.yaw), math.sin(ref.yaw)
        vel = np.array([c * v - s * vd, s * v + c * vd])
        yaw = ref.yaw + math.atan2(vd, v) if (abs(v) + abs(vd)) > 0.05 else ref.yaw
        if self.cls == ObjectClass.PEDESTRIAN and v < 0.05 and abs(vd) <= 0.05:
            yaw = ref.yaw + math.pi / 2
        return ActorTruth(self.name, self.cls, Pose2D(ref.x, ref.y, yaw), vel, self.length, self.width,
                          self.height, accel=self.accel,
                          yaw_rate=v * self.road.curvature_at(self.s))


# ------------------------------------------------------------------ scenario
@dataclass
class EgoSpec:
    s0: float = 0.0
    lane: int = 0
    v0: float = 15.0
    target_speed: float = 15.0
    d0: Optional[float] = None  # overrides lane centre


@dataclass
class ActorSpec:
    name: str
    behavior: Behavior
    s0: float
    d0: float
    v0: float = 0.0
    cls: ObjectClass = ObjectClass.CAR
    length: float = 4.5
    width: float = 1.8
    height: float = 1.5


@dataclass
class Scenario:
    name: str
    road: Road
    ego: EgoSpec = field(default_factory=EgoSpec)
    actors: list[ActorSpec] = field(default_factory=list)
    duration: float = 10.0
    dt: float = 0.02
    seed: int = 0
    params: dict = field(default_factory=dict)
    description: str = ""

    def ego_d(self) -> float:
        return self.ego.d0 if self.ego.d0 is not None else self.road.lane_offset(self.ego.lane)

    def ego_start_pose(self) -> Pose2D:
        return self.road.pose_at(self.ego.s0, self.ego_d())

    def build_world(self, vehicle_params: VehicleParams = VehicleParams(),
                    twin_config: TwinConfig = TwinConfig(), clock: Optional[SimClock] = None) -> tuple[World, VehicleTwin]:
        """Fresh world (ego twin first, then actors). Behaviours are deep-copied so runs are independent."""
        import copy

        world = World(WorldConfig(dt=self.dt, seed=self.seed), clock)
        world.context["road"] = self.road
        world.context["scenario"] = self
        twin = VehicleTwin(vehicle_params, twin_config, self.ego_start_pose(), self.ego.v0)
        world.add(twin)
        for spec in self.actors:
            world.add(FrenetActor(spec.name, self.road, copy.deepcopy(spec.behavior), spec.s0, spec.d0, spec.v0,
                                  spec.cls, spec.length, spec.width, spec.height))
        return world, twin


# ---------------------------------------------------------------- built-ins
def straight_follow(gap: float = 30.0, ego_speed: float = 20.0, lead_speed: float = 15.0,
                    duration: float = 12.0, num_lanes: int = 1, seed: int = 0) -> Scenario:
    road = Road.straight(max(400.0, gap + ego_speed * duration * 1.5), num_lanes=num_lanes)
    d = road.lane_offset(0)
    return Scenario(
        "straight_follow", road, EgoSpec(10.0, 0, ego_speed, ego_speed),
        [ActorSpec("lead", Behavior(HoldSpeed(lead_speed)), 10.0 + gap, d, lead_speed)],
        duration, seed=seed, params=dict(gap=gap, ego_speed=ego_speed, lead_speed=lead_speed),
        description="Ego approaches a slower lead in the same lane.",
    )


def cut_in(gap: float = 15.0, ego_speed: float = 20.0, actor_speed: float = 17.0, cut_in_time: float = 2.0,
           lane_change_duration: float = 2.5, duration: float = 12.0, seed: int = 0) -> Scenario:
    road = Road.straight(600.0, num_lanes=2)
    d0, d1 = road.lane_offset(0), road.lane_offset(1)
    beh = Behavior(HoldSpeed(actor_speed), LaneChange(d0, lane_change_duration, t_start=cut_in_time))
    return Scenario(
        "cut_in", road, EgoSpec(10.0, 0, ego_speed, ego_speed),
        [ActorSpec("cutter", beh, 10.0 + gap, d1, actor_speed)],
        duration, seed=seed,
        params=dict(gap=gap, ego_speed=ego_speed, actor_speed=actor_speed, cut_in_time=cut_in_time),
        description="Actor in the left lane cuts into the ego lane ahead.",
    )


def hard_brake_lead(gap: float = 30.0, speed: float = 20.0, decel: float = 6.0, brake_time: float = 2.0,
                    duration: float = 10.0, seed: int = 0) -> Scenario:
    road = Road.straight(500.0)
    beh = Behavior(BrakeAt(brake_time, decel, HoldSpeed(speed)))
    return Scenario(
        "hard_brake_lead", road, EgoSpec(10.0, 0, speed, speed),
        [ActorSpec("lead", beh, 10.0 + gap, road.lane_offset(0), speed)],
        duration, seed=seed, params=dict(gap=gap, speed=speed, decel=decel, brake_time=brake_time),
        description="Lead vehicle brakes hard to a stop.",
    )


def curve(radius: float = 120.0, speed: float = 15.0, angle_deg: float = 90.0, duration: float = 14.0,
          seed: int = 0) -> Scenario:
    road = Road.curve(radius, math.radians(angle_deg), lead_in=40.0, transition=25.0, lead_out=150.0)
    return Scenario(
        "curve", road, EgoSpec(5.0, 0, speed, speed), [], duration, seed=seed,
        params=dict(radius=radius, speed=speed, angle_deg=angle_deg),
        description="Lane keeping through a clothoid-entered curve.",
    )


def pedestrian_crossing(distance: float = 45.0, ego_speed: float = 12.0, walk_speed: float = 1.4,
                        start_time: float = 0.5, duration: float = 10.0, seed: int = 0) -> Scenario:
    road = Road.straight(300.0, num_lanes=1)
    d_start, d_end = -road.half_width - 1.5, road.half_width + 1.5
    beh = Behavior(Stationary(), Cross(d_end, walk_speed, start_time))
    return Scenario(
        "pedestrian_crossing", road, EgoSpec(10.0, 0, ego_speed, ego_speed),
        [ActorSpec("ped", beh, 10.0 + distance, d_start, 0.0, ObjectClass.PEDESTRIAN, 0.5, 0.5, 1.7)],
        duration, seed=seed,
        params=dict(distance=distance, ego_speed=ego_speed, walk_speed=walk_speed, start_time=start_time),
        description="Pedestrian crosses the ego lane from the right.",
    )


BUILTIN_SCENARIOS: dict[str, Callable[..., Scenario]] = {
    "straight_follow": straight_follow,
    "cut_in": cut_in,
    "hard_brake_lead": hard_brake_lead,
    "curve": curve,
    "pedestrian_crossing": pedestrian_crossing,
}


def make_scenario(name: str, **params) -> Scenario:
    return BUILTIN_SCENARIOS[name](**params)


# ---------------------------------------------------------------- variation
@dataclass
class ParamRange:
    """Distribution of one scenario parameter: uniform [low, high], integer, or choice."""

    low: float = 0.0
    high: float = 1.0
    kind: str = "uniform"  # "uniform" | "int" | "choice"
    choices: tuple = ()

    def sample(self, rng: np.random.Generator):
        if self.kind == "choice":
            return self.choices[int(rng.integers(len(self.choices)))]
        if self.kind == "int":
            return int(rng.integers(int(self.low), int(self.high) + 1))
        return float(rng.uniform(self.low, self.high))

    def grid(self, n: int) -> list:
        if self.kind == "choice":
            return list(self.choices)
        vals = np.linspace(self.low, self.high, max(1, n))
        return sorted({int(round(v)) for v in vals}) if self.kind == "int" else [float(v) for v in vals]


@dataclass
class ScenarioVariation:
    """Parametric family of scenarios around `factory(**base)`."""

    factory: Callable[..., Scenario]
    base: dict = field(default_factory=dict)
    ranges: dict[str, ParamRange] = field(default_factory=dict)

    def build(self, params: dict, seed: int = 0) -> Scenario:
        sc = self.factory(**{**self.base, **params})
        sc.seed = seed
        sc.params = {**sc.params, **params}
        return sc

    def sample(self, rng: np.random.Generator, seed: Optional[int] = None) -> tuple[dict, Scenario]:
        params = {k: r.sample(rng) for k, r in sorted(self.ranges.items())}
        seed = int(rng.integers(2**31)) if seed is None else seed
        return params, self.build(params, seed)

    def grid(self, points_per_dim: int = 3) -> list[dict]:
        keys = sorted(self.ranges)
        return [dict(zip(keys, combo)) for combo in itertools.product(*(self.ranges[k].grid(points_per_dim) for k in keys))]
