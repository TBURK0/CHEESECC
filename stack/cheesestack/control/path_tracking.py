"""Geometric / kinematic path-following controllers.

All controllers return a desired path curvature (1/m, +left) for a rear-axle
referenced kinematic bicycle. Convert to steering with `curvature_control`.

- PurePursuit: kappa = 2 sin(alpha) / Ld, speed-adaptive lookahead
  Ld = clamp(k_v * v + Ld0). Same formula, clamps and lookahead floor as the
  app's PurePursuit.swift (alpha clipped to +-pi/2, Ld >= 1 m).
- Stanley: front-axle referenced, delta = -(e_psi + atan(k e_y / (k_soft + v))),
  converted to curvature tan(delta)/L.
- RearWheelFeedback: Lyapunov-based law (Samson / Paden et al. 2016):
  omega = v k_r cos(e_psi)/(1 - k_r e) - k_psi |v| e_psi - k_e v sin(e_psi)/e_psi * e

Limits: pure kinematic models; no tire slip. Large errors at speed should be
handled by the LQR controller plus stability limiting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cheesestack.core import Pose2D, wrap_angle

from .reference_path import ReferencePath


@dataclass
class TrackingResult:
    curvature: float
    cross_track: float
    heading_error: float
    s: float
    target_xy: tuple[float, float] | None = None
    lookahead_m: float | None = None


@dataclass
class PurePursuitConfig:
    lookahead_min_m: float = 4.0
    lookahead_max_m: float = 25.0
    lookahead_gain_s: float = 0.8  # Ld = Ld0 + k*v (seconds)
    lookahead_base_m: float = 3.0


def pure_pursuit_curvature(lookahead_m: float, alpha_rad: float) -> float:
    """Port of PurePursuit.swift, in stack sign (+alpha = target to the LEFT)."""
    lookahead = max(1.0, lookahead_m)
    alpha = max(-math.pi / 2, min(math.pi / 2, alpha_rad))
    return 2.0 * math.sin(alpha) / lookahead


class PurePursuit:
    def __init__(self, config: PurePursuitConfig | None = None) -> None:
        self.config = config or PurePursuitConfig()

    def lookahead(self, v: float) -> float:
        c = self.config
        return min(c.lookahead_max_m, max(c.lookahead_min_m, c.lookahead_base_m + c.lookahead_gain_s * abs(v)))

    def update(self, pose: Pose2D, v: float, path: ReferencePath) -> TrackingResult:
        proj = path.project(pose)
        ld = self.lookahead(v)
        # Pick the path point whose arc length is ~ld ahead of the projection (TODO: circle intersection).
        tx, ty, _ = path.point_at(proj.s + ld)
        dx, dy = tx - pose.x, ty - pose.y
        alpha = wrap_angle(math.atan2(dy, dx) - pose.yaw)
        dist = max(math.hypot(dx, dy), 1e-3)
        # Like VehicleCommandSolver.swift: use the larger of nominal lookahead and actual distance.
        k = pure_pursuit_curvature(max(ld, dist), alpha)
        return TrackingResult(k, proj.cross_track, proj.heading_error, proj.s, (tx, ty), max(ld, dist))


@dataclass
class StanleyConfig:
    k_gain: float = 1.5
    k_soft_mps: float = 1.0
    wheelbase_m: float = 2.83
    max_steer_rad: float = 0.6


class Stanley:
    def __init__(self, config: StanleyConfig | None = None) -> None:
        self.config = config or StanleyConfig()

    def update(self, pose: Pose2D, v: float, path: ReferencePath) -> TrackingResult:
        c = self.config
        front = Pose2D(pose.x + c.wheelbase_m * math.cos(pose.yaw), pose.y + c.wheelbase_m * math.sin(pose.yaw), pose.yaw)
        proj = path.project(front)
        delta = -(proj.heading_error + math.atan2(c.k_gain * proj.cross_track, c.k_soft_mps + abs(v)))
        delta = max(-c.max_steer_rad, min(c.max_steer_rad, delta))
        # TODO(control): add yaw-rate damping and steer-delay terms from Stanley's DARPA paper.
        return TrackingResult(math.tan(delta) / c.wheelbase_m, proj.cross_track, proj.heading_error, proj.s)


@dataclass
class RearWheelFeedbackConfig:
    k_e: float = 0.3  # 1/m^2-ish cross-track gain
    k_psi: float = 1.5  # heading gain, 1/m


class RearWheelFeedback:
    def __init__(self, config: RearWheelFeedbackConfig | None = None) -> None:
        self.config = config or RearWheelFeedbackConfig()

    def update(self, pose: Pose2D, v: float, path: ReferencePath) -> TrackingResult:
        c = self.config
        proj = path.project(pose)
        e, th, kr = proj.cross_track, proj.heading_error, proj.curvature
        sinc = 1.0 if abs(th) < 1e-6 else math.sin(th) / th
        denom = 1.0 - kr * e
        if abs(denom) < 1e-3:
            denom = math.copysign(1e-3, denom)
        # Curvature form (omega / v, v > 0):
        #   kappa = kr cos(th)/(1 - kr e) - k_psi sign(v) th - k_e sinc(th) e
        # TODO(control): validate reverse driving (sign(v) < 0) in sim.
        sgn = 1.0 if v >= 0 else -1.0
        k = kr * math.cos(th) / denom - c.k_psi * sgn * th - c.k_e * sinc * e
        return TrackingResult(k, e, th, proj.s)
