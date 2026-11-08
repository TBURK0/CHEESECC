"""Physics simulation: fixed-step world, RK4 integrator, actors, collisions.

The `World` owns a `core.SimClock` and a set of actors. Each `step()` advances
every actor by exactly `dt` (deterministic insertion order) and then the clock.
Randomness is deterministic: `world.rng(name)` returns a generator seeded from
(world seed, crc32(name)), so adding an actor never perturbs another actor's or
sensor's random stream.

Actors integrate their own continuous dynamics, typically with `rk4_step`.
Collision checking uses oriented-rectangle separating-axis tests on the
ground plane.

Limits: no actor-actor interaction forces (collisions are detected, not
resolved), single global step size.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Protocol, runtime_checkable

import numpy as np

from cheesestack.core import ObjectClass, Pose2D, SimClock


def rk4_step(f: Callable[[float, np.ndarray], np.ndarray], x: np.ndarray, t: float, dt: float) -> np.ndarray:
    """Classic 4th-order Runge-Kutta step for dx/dt = f(t, x)."""
    k1 = f(t, x)
    k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2)
    k4 = f(t + dt, x + dt * k3)
    return x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def euler_step(f: Callable[[float, np.ndarray], np.ndarray], x: np.ndarray, t: float, dt: float) -> np.ndarray:
    return x + dt * f(t, x)


@dataclass
class ActorTruth:
    """Ground-truth snapshot of an actor. `pose` is the geometric box centre in `map`."""

    name: str
    cls: ObjectClass
    pose: Pose2D
    velocity: np.ndarray  # (vx, vy) in map
    length: float
    width: float
    height: float = 1.5
    accel: float = 0.0
    yaw_rate: float = 0.0

    @property
    def speed(self) -> float:
        return float(np.hypot(self.velocity[0], self.velocity[1]))


@runtime_checkable
class Actor(Protocol):
    name: str

    def step(self, t: float, dt: float, world: "World") -> None: ...

    def truth(self) -> ActorTruth: ...


def box_corners(pose: Pose2D, length: float, width: float) -> np.ndarray:
    """(4, 2) corners of an oriented rectangle centred at pose, CCW from front-left."""
    hl, hw = 0.5 * length, 0.5 * width
    local = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
    return pose.transform_points(local)


def boxes_overlap(a: ActorTruth, b: ActorTruth, margin: float = 0.0) -> bool:
    """Separating-axis test for two oriented rectangles (optionally inflated by margin)."""
    ca = box_corners(a.pose, a.length + 2 * margin, a.width + 2 * margin)
    cb = box_corners(b.pose, b.length + 2 * margin, b.width + 2 * margin)
    for yaw in (a.pose.yaw, b.pose.yaw):
        for axis in (np.array([math.cos(yaw), math.sin(yaw)]), np.array([-math.sin(yaw), math.cos(yaw)])):
            pa, pb = ca @ axis, cb @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


@dataclass
class WorldConfig:
    dt: float = 0.01
    seed: int = 0
    collision_margin: float = 0.0


class World:
    """Fixed-step multi-actor world on a SimClock."""

    def __init__(self, config: WorldConfig = WorldConfig(), clock: Optional[SimClock] = None) -> None:
        if config.dt <= 0:
            raise ValueError("dt must be positive")
        self.config = config
        self.clock = clock or SimClock()
        self._actors: dict[str, Actor] = {}
        self.step_count = 0
        self._t0 = self.clock.now()
        self.context: dict = {}  # free-form shared data (e.g. the road)

    @property
    def dt(self) -> float:
        return self.config.dt

    @property
    def t(self) -> float:
        return self.clock.now()

    def rng(self, name: str) -> np.random.Generator:
        """Deterministic, independent random stream for a named consumer."""
        return np.random.default_rng(np.random.SeedSequence([self.config.seed & 0xFFFFFFFF, zlib.crc32(name.encode())]))

    def add(self, actor: Actor) -> Actor:
        if actor.name in self._actors:
            raise ValueError(f"duplicate actor name {actor.name!r}")
        self._actors[actor.name] = actor
        return actor

    def remove(self, name: str) -> None:
        self._actors.pop(name, None)

    def get(self, name: str) -> Actor:
        return self._actors[name]

    @property
    def actors(self) -> list[Actor]:
        return list(self._actors.values())

    def truths(self, exclude: Iterable[str] = ()) -> list[ActorTruth]:
        ex = set(exclude)
        return [a.truth() for n, a in self._actors.items() if n not in ex]

    def step(self) -> float:
        t = self.clock.now()
        for actor in list(self._actors.values()):
            actor.step(t, self.dt, self)
        self.step_count += 1
        # t0 + n*dt (not repeated +=) avoids float drift in long runs.
        self.clock.set(max(self.clock.now(), self._t0 + self.step_count * self.dt))
        return self.clock.now()

    def run(self, duration: float, callback: Optional[Callable[["World"], None]] = None) -> None:
        n = int(round(duration / self.dt))
        for _ in range(n):
            self.step()
            if callback is not None:
                callback(self)

    def collisions(self, name: Optional[str] = None) -> list[tuple[str, str]]:
        """Overlapping actor pairs (all pairs, or only those involving `name`)."""
        truths = {n: a.truth() for n, a in self._actors.items()}
        names = list(truths)
        out = []
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                if name is not None and name not in (a, b):
                    continue
                if boxes_overlap(truths[a], truths[b], self.config.collision_margin):
                    out.append((a, b))
        return out


class ConstantVelocityActor:
    """Simplest actor: constant velocity in map frame (useful for tests/static obstacles)."""

    def __init__(self, name: str, pose: Pose2D, velocity=(0.0, 0.0), cls: ObjectClass = ObjectClass.CAR,
                 length: float = 4.5, width: float = 1.8) -> None:
        self.name, self.cls, self.length, self.width = name, cls, length, width
        self.state = np.array([pose.x, pose.y], dtype=float)
        self.yaw = pose.yaw
        self.velocity = np.asarray(velocity, dtype=float)

    def step(self, t: float, dt: float, world: World) -> None:
        self.state = rk4_step(lambda _t, x: self.velocity, self.state, t, dt)

    def truth(self) -> ActorTruth:
        return ActorTruth(self.name, self.cls, Pose2D(self.state[0], self.state[1], self.yaw),
                          self.velocity.copy(), self.length, self.width)
