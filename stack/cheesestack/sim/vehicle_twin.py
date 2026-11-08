"""Vehicle digital twin: parameterized bicycle model with actuator dynamics.

State vector (all SI): [x, y, yaw, vx, vy, r, delta, a]
  x, y   rear-axle centre in `map` (matches `base_link`)
  yaw    heading
  vx, vy body-frame velocity of the centre of gravity
  r      yaw rate
  delta  actual road-wheel angle (after actuator lag / rate limit)
  a      actual longitudinal drive/brake acceleration (after actuator lag / jerk limit)

Models:
  * kinematic bicycle (rear-axle reference, no slip) - valid at low speed / low lateral accel.
  * dynamic bicycle with linear tyres saturated at mu*Fz (friction circle ignored).
    Below `dynamic_min_speed` the twin falls back to the kinematic model (the dynamic
    model is singular at vx -> 0), with hysteresis.

Actuators: first-order lag + rate limit on steering, first-order lag + jerk limit
on acceleration, rolling resistance and aerodynamic drag, no reverse gear.

Command mapping (`TwinConfig.command_mode`):
  "pedals":          steer_wheel_deg / steering_ratio -> delta_cmd,
                     throttle * max_accel - brake * max_decel -> a_cmd
  "curvature_speed": atan(curvature * L) -> delta_cmd, P-loop on target_speed_mps -> a_cmd

# TODO(sim): Pacejka tyres + combined slip, load transfer, powertrain/brake maps,
# steering-column compliance, road grade, calibration from logged data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from cheesestack.core import ControlCommand, Header, ObjectClass, Pose2D, VehicleState, wrap_angle

from .physics import ActorTruth, World, rk4_step

G = 9.81
IX, IY, IYAW, IVX, IVY, IR, IDELTA, IA = range(8)


@dataclass
class VehicleParams:
    wheelbase: float = 2.7
    lf: float = 1.2  # CG -> front axle
    mass: float = 1500.0
    iz: float = 2250.0
    cf: float = 80000.0  # front axle cornering stiffness N/rad
    cr: float = 80000.0
    mu: float = 0.9
    length: float = 4.7
    width: float = 1.85
    height: float = 1.5
    rear_overhang: float = 1.0  # rear bumper -> rear axle
    track: float = 1.6
    wheel_radius: float = 0.32
    steering_ratio: float = 15.0
    max_steer: float = 0.55  # rad road-wheel
    max_steer_rate: float = 0.8  # rad/s
    steer_tau: float = 0.08  # s
    max_accel: float = 3.0
    max_decel: float = 8.0
    accel_tau: float = 0.25
    max_jerk: float = 20.0
    drag_coeff: float = 0.4  # 0.5*rho*Cd*A, N/(m/s)^2
    rolling_coeff: float = 0.012
    max_speed: float = 50.0

    @property
    def lr(self) -> float:
        return self.wheelbase - self.lf

    @property
    def center_offset(self) -> float:
        """Distance from rear axle forward to the geometric box centre."""
        return 0.5 * self.length - self.rear_overhang


@dataclass
class TwinConfig:
    name: str = "ego"
    model: str = "dynamic"  # "dynamic" | "kinematic"
    dynamic_min_speed: float = 4.0
    dynamic_hysteresis: float = 0.5
    command_mode: str = "pedals"  # "pedals" | "curvature_speed"
    speed_kp: float = 1.2
    substeps: int = 1


class VehicleTwin:
    """Digital twin that consumes `ControlCommand` and produces `VehicleState`.

    Implements the `physics.Actor` protocol, so it can live in a `World`.
    """

    cls = ObjectClass.CAR

    def __init__(
        self,
        params: VehicleParams = VehicleParams(),
        config: TwinConfig = TwinConfig(),
        pose: Pose2D = Pose2D(),
        v0: float = 0.0,
    ) -> None:
        self.params = params
        self.config = config
        self.name = config.name
        self.length, self.width = params.length, params.width
        self.x = np.zeros(8)
        self._u = (0.0, 0.0)
        self._dynamic = False
        self.last_command: Optional[ControlCommand] = None
        self.accel_body = np.zeros(2)  # CG acceleration (ax, ay) in body frame, for the IMU
        self._seq = 0
        self.reset(pose, v0)

    # ------------------------------------------------------------- interface
    def reset(self, pose: Pose2D, v0: float = 0.0) -> None:
        self.x[:] = 0.0
        self.x[IX], self.x[IY], self.x[IYAW], self.x[IVX] = pose.x, pose.y, pose.yaw, v0
        self._u = (0.0, 0.0)
        self._dynamic = self.config.model == "dynamic" and v0 >= self.config.dynamic_min_speed

    def command_to_inputs(self, cmd: ControlCommand) -> tuple[float, float]:
        p = self.params
        if self.config.command_mode == "curvature_speed":
            delta = math.atan(cmd.curvature * p.wheelbase)
            a = self.config.speed_kp * (cmd.target_speed_mps - self.x[IVX])
        else:
            delta = math.radians(cmd.steer_wheel_deg) / p.steering_ratio
            a = float(np.clip(cmd.throttle, 0, 1)) * p.max_accel - float(np.clip(cmd.brake, 0, 1)) * p.max_decel
        return float(np.clip(delta, -p.max_steer, p.max_steer)), float(np.clip(a, -p.max_decel, p.max_accel))

    def apply(self, cmd: ControlCommand) -> None:
        """Latch a command (zero-order hold until the next one)."""
        self.last_command = cmd
        self._u = self.command_to_inputs(cmd)

    def step(self, t: float, dt: float, world: Optional[World] = None) -> None:
        self._update_mode()
        h = dt / max(1, self.config.substeps)
        for i in range(max(1, self.config.substeps)):
            xd = self.derivative(t + i * h, self.x)
            self.x = rk4_step(self.derivative, self.x, t + i * h, h)
            self._post_step()
        self.accel_body = np.array([xd[IVX] - self.x[IR] * self.x[IVY], xd[IVY] + self.x[IR] * self.x[IVX]])
        self._last_vx_dot = float(xd[IVX])

    def vehicle_state(self, t: float) -> VehicleState:
        p = self.params
        self._seq += 1
        delta = self.x[IDELTA]
        return VehicleState(
            header=Header(stamp=t, frame_id="map", seq=self._seq, source="sim.twin"),
            pose=self.pose,
            v=float(self.x[IVX]),
            a=float(getattr(self, "_last_vx_dot", self.x[IA])),
            yaw_rate=float(self.x[IR]),
            steer=float(delta),
            curvature=float(math.tan(delta) / p.wheelbase),
        )

    def truth(self) -> ActorTruth:
        p = self.params
        yaw = self.x[IYAW]
        c, s = math.cos(yaw), math.sin(yaw)
        off = p.center_offset
        center = Pose2D(float(self.x[IX] + off * c), float(self.x[IY] + off * s), wrap_angle(float(yaw)))
        # Velocity of the box centre: rear-axle velocity + r x offset.
        vbx, vby = self.x[IVX], self.x[IVY] - p.lr * self.x[IR] + off * self.x[IR]
        vel = np.array([c * vbx - s * vby, s * vbx + c * vby])
        return ActorTruth(self.name, self.cls, center, vel, p.length, p.width, p.height,
                          accel=float(self.x[IA]), yaw_rate=float(self.x[IR]))

    @property
    def pose(self) -> Pose2D:
        return Pose2D(float(self.x[IX]), float(self.x[IY]), wrap_angle(float(self.x[IYAW])))

    @property
    def speed(self) -> float:
        return float(self.x[IVX])

    # -------------------------------------------------------------- dynamics
    def _update_mode(self) -> None:
        c = self.config
        if c.model != "dynamic":
            self._dynamic = False
            return
        vx = self.x[IVX]
        if self._dynamic and vx < c.dynamic_min_speed - c.dynamic_hysteresis:
            self._dynamic = False
            self._project_kinematic()
        elif not self._dynamic and vx > c.dynamic_min_speed:
            self._dynamic = True

    def _project_kinematic(self) -> None:
        p = self.params
        r = self.x[IVX] * math.tan(self.x[IDELTA]) / p.wheelbase
        self.x[IR], self.x[IVY] = r, p.lr * r

    def _post_step(self) -> None:
        p = self.params
        self.x[IVX] = float(np.clip(self.x[IVX], 0.0, p.max_speed))  # no reverse gear
        self.x[IDELTA] = float(np.clip(self.x[IDELTA], -p.max_steer, p.max_steer))
        if not self._dynamic:
            self._project_kinematic()

    def derivative(self, t: float, x: np.ndarray) -> np.ndarray:
        p = self.params
        delta_cmd, a_cmd = self._u
        yaw, vx, vy, r, delta, a = x[IYAW], x[IVX], x[IVY], x[IR], x[IDELTA], x[IA]
        dx = np.zeros(8)
        dx[IDELTA] = float(np.clip((delta_cmd - delta) / p.steer_tau, -p.max_steer_rate, p.max_steer_rate))
        dx[IA] = float(np.clip((a_cmd - a) / p.accel_tau, -p.max_jerk, p.max_jerk))
        resist = (G * p.rolling_coeff + p.drag_coeff * vx * vx / p.mass) if vx > 1e-3 else 0.0
        if self._dynamic and vx > 0.5:
            alpha_f = delta - math.atan2(vy + p.lf * r, vx)
            alpha_r = -math.atan2(vy - p.lr * r, vx)
            fzf, fzr = p.mass * G * p.lr / p.wheelbase, p.mass * G * p.lf / p.wheelbase
            fyf = float(np.clip(p.cf * alpha_f, -p.mu * fzf, p.mu * fzf))
            fyr = float(np.clip(p.cr * alpha_r, -p.mu * fzr, p.mu * fzr))
            dx[IVX] = a - resist + r * vy - fyf * math.sin(delta) / p.mass
            dx[IVY] = (fyf * math.cos(delta) + fyr) / p.mass - r * vx
            dx[IR] = (p.lf * fyf * math.cos(delta) - p.lr * fyr) / p.iz
            vbx, vby, yaw_rate = vx, vy - p.lr * r, r
        else:
            yaw_rate = vx * math.tan(delta) / p.wheelbase
            dx[IVX] = a - resist
            vbx, vby = vx, 0.0
        if vx <= 0.0 and dx[IVX] < 0.0:
            dx[IVX] = 0.0
        c, s = math.cos(yaw), math.sin(yaw)
        dx[IX] = c * vbx - s * vby
        dx[IY] = s * vbx + c * vby
        dx[IYAW] = yaw_rate
        return dx

    def with_params(self, **changes) -> "VehicleTwin":
        """Copy of this twin with modified parameters (for Monte Carlo variation)."""
        twin = VehicleTwin(replace(self.params, **changes), self.config, self.pose, self.speed)
        return twin
