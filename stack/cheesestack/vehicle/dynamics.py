"""Vehicle dynamics models: kinematic bicycle, linear dynamic bicycle, longitudinal.

- `KinematicBicycle`: rear-axle reference, state [x, y, yaw, v], input (delta, a).
  Exact at low speed / low lateral acceleration; no tire slip.
- `DynamicBicycle`: linear single-track model with linear tires, CG reference,
  state [x, y, yaw, vx, vy, r], input (delta, ax). Valid for |a_y| < ~0.4 g.
  Below `min_speed` the tire-slip terms are singular, so it blends to kinematic.
- `LongitudinalModel`: drive force (torque / power / traction limited, after the
  app's LongitudinalModel.swift) minus aero drag, rolling resistance and grade.
- `rk4_step`: generic fixed-step RK4 integrator used by all models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from cheesestack.core import wrap_angle

from .params import AIR_DENSITY, GRAVITY, VehicleParams

Deriv = Callable[[np.ndarray, np.ndarray], np.ndarray]


def rk4_step(f: Deriv, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """One classical Runge-Kutta 4 step of dx/dt = f(x, u) with zero-order-hold u."""
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def understeer_gradient(params: VehicleParams) -> float:
    """K = m/L (lr/Cf - lf/Cr) [rad/(m/s^2)]."""
    return params.understeer_gradient


@dataclass
class KinematicBicycle:
    """Rear-axle kinematic bicycle. x = [x, y, yaw, v]; u = [delta, a]."""

    params: VehicleParams

    def deriv(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        _, _, yaw, v = x
        delta, a = u
        return np.array([v * math.cos(yaw), v * math.sin(yaw), v * math.tan(delta) / self.params.wheelbase, a])

    def step(self, x: np.ndarray, delta: float, a: float, dt: float) -> np.ndarray:
        lim = self.params.max_road_wheel_angle
        u = np.array([float(np.clip(delta, -lim, lim)), a])
        out = rk4_step(self.deriv, np.asarray(x, dtype=float), u, dt)
        out[2] = wrap_angle(out[2])
        out[3] = max(0.0, out[3])  # TODO(vehicle): reverse gear support
        return out

    def yaw_rate(self, v: float, delta: float) -> float:
        return v * math.tan(delta) / self.params.wheelbase


@dataclass
class DynamicBicycleConfig:
    min_speed: float = 2.0  # m/s; below this use kinematic lateral behavior
    max_slip_angle: float = math.radians(12.0)  # clamp linear tire beyond this (crude saturation)


class DynamicBicycle:
    """Linear dynamic single-track model at the CG. x = [x, y, yaw, vx, vy, r]; u = [delta, ax]."""

    def __init__(self, params: VehicleParams, config: Optional[DynamicBicycleConfig] = None) -> None:
        self.params = params
        self.config = config or DynamicBicycleConfig()

    def slip_angles(self, vx: float, vy: float, r: float, delta: float) -> tuple[float, float]:
        p = self.params
        vxs = max(abs(vx), self.config.min_speed)
        af = delta - math.atan2(vy + p.lf * r, vxs)
        ar = -math.atan2(vy - p.lr * r, vxs)
        return af, ar

    def tire_forces(self, vx: float, vy: float, r: float, delta: float) -> tuple[float, float]:
        p = self.params
        af, ar = self.slip_angles(vx, vy, r, delta)
        m = self.config.max_slip_angle
        fyf = p.cornering_stiffness_front * float(np.clip(af, -m, m))
        fyr = p.cornering_stiffness_rear * float(np.clip(ar, -m, m))
        # Friction circle cap per axle. TODO(vehicle): Pacejka / brush tire, load transfer.
        fyf = float(np.clip(fyf, -p.tire_friction * p.axle_load_front, p.tire_friction * p.axle_load_front))
        fyr = float(np.clip(fyr, -p.tire_friction * p.axle_load_rear, p.tire_friction * p.axle_load_rear))
        return fyf, fyr

    def deriv(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        p = self.params
        _, _, yaw, vx, vy, r = x
        delta, ax = u
        c, s = math.cos(yaw), math.sin(yaw)
        dx = vx * c - vy * s
        dy = vx * s + vy * c
        if abs(vx) < self.config.min_speed:
            # Kinematic lateral states: relax vy, r toward kinematic values.
            beta = math.atan(p.lr * math.tan(delta) / p.wheelbase)
            r_kin = vx * math.cos(beta) * math.tan(delta) / p.wheelbase
            vy_kin = vx * math.tan(beta)
            tau = 0.05
            return np.array([dx, dy, r, ax, (vy_kin - vy) / tau, (r_kin - r) / tau])
        fyf, fyr = self.tire_forces(vx, vy, r, delta)
        dvx = ax + vy * r - fyf * math.sin(delta) / p.mass
        dvy = (fyf * math.cos(delta) + fyr) / p.mass - vx * r
        dr = (p.lf * fyf * math.cos(delta) - p.lr * fyr) / p.yaw_inertia
        return np.array([dx, dy, r, dvx, dvy, dr])

    def step(self, x: np.ndarray, delta: float, ax: float, dt: float, substeps: int = 1) -> np.ndarray:
        lim = self.params.max_road_wheel_angle
        u = np.array([float(np.clip(delta, -lim, lim)), ax])
        out = np.asarray(x, dtype=float).copy()
        h = dt / max(1, substeps)
        for _ in range(max(1, substeps)):
            out = rk4_step(self.deriv, out, u, h)
        out[2] = wrap_angle(out[2])
        return out

    def lateral_state_space(self, vx: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Small-angle LTI model at speed vx for lateral states [vy, r] and input delta.

        Returns (A, B, C_ay, D_ay) with d[vy, r]/dt = A x + B delta and a_y = C_ay x + D_ay delta.
        """
        p = self.params
        v = max(abs(vx), self.config.min_speed)
        cf, cr, m, iz, lf, lr = p.cornering_stiffness_front, p.cornering_stiffness_rear, p.mass, p.yaw_inertia, p.lf, p.lr
        a11 = -(cf + cr) / (m * v)
        a12 = -(lf * cf - lr * cr) / (m * v) - v
        a21 = -(lf * cf - lr * cr) / (iz * v)
        a22 = -(lf * lf * cf + lr * lr * cr) / (iz * v)
        A = np.array([[a11, a12], [a21, a22]])
        B = np.array([cf / m, lf * cf / iz])
        C = np.array([a11, a12 + v])
        D = float(cf / m)
        return A, B, C, D

    def steady_state(self, vx: float, delta: float) -> tuple[float, float]:
        """Steady-state (vy, r) for constant delta at speed vx (linear model)."""
        A, B, _, _ = self.lateral_state_space(vx)
        x = -np.linalg.solve(A, B * delta)
        return float(x[0]), float(x[1])


@dataclass
class LongitudinalConfig:
    min_mass: float = 400.0
    torque_multiplier: float = 9.0  # overall gear * final drive in first gear (as in app)
    traction_accel_cap: float = 6.0  # m/s^2
    max_brake_decel: float = 6.5  # m/s^2 at brake=1
    power_min_speed: float = 3.0  # m/s, avoids P/v blow-up


class LongitudinalModel:
    """Longitudinal acceleration from pedals and resistive forces."""

    def __init__(self, params: VehicleParams, config: Optional[LongitudinalConfig] = None) -> None:
        self.params = params
        self.config = config or LongitudinalConfig()

    def drag_force(self, v: float) -> float:
        p = self.params
        return 0.5 * AIR_DENSITY * p.drag_coefficient * p.frontal_area * v * abs(v)

    def rolling_force(self, v: float, grade_rad: float = 0.0) -> float:
        p = self.params
        if abs(v) < 1e-3:
            return 0.0
        return math.copysign(p.rolling_resistance * p.mass * GRAVITY * math.cos(grade_rad), v)

    def grade_force(self, grade_rad: float) -> float:
        return self.params.mass * GRAVITY * math.sin(grade_rad)

    def resistive_force(self, v: float, grade_rad: float = 0.0) -> float:
        return self.drag_force(v) + self.rolling_force(v, grade_rad) + self.grade_force(grade_rad)

    def drive_force(self, v: float, throttle: float) -> float:
        p, c = self.params, self.config
        th = float(np.clip(throttle, 0.0, 1.0))
        mass = max(c.min_mass, p.mass)
        torque_force = p.torque_nm * c.torque_multiplier * th / max(0.2, p.tire_radius)
        power_force = p.horsepower * 745.7 * th / max(abs(v), c.power_min_speed)
        return min(mass * c.traction_accel_cap, torque_force, power_force)

    def acceleration(self, v: float, throttle: float, brake: float, grade_rad: float = 0.0) -> float:
        p, c = self.params, self.config
        mass = max(c.min_mass, p.mass)
        a = (self.drive_force(v, throttle) - self.resistive_force(v, grade_rad)) / mass
        a -= float(np.clip(brake, 0.0, 1.0)) * c.max_brake_decel
        # TODO(vehicle): gear schedule, engine map, brake/creep dynamics and actuator lag.
        return a

    def step(self, v: float, throttle: float, brake: float, dt: float, grade_rad: float = 0.0) -> float:
        f = lambda x, u: np.array([self.acceleration(float(x[0]), throttle, brake, grade_rad)])
        v_new = float(rk4_step(f, np.array([v]), np.zeros(0), dt)[0])
        return max(0.0, v_new) if v >= 0 else v_new

    def coastdown_decel(self, v: float) -> float:
        return self.resistive_force(v) / self.params.mass
