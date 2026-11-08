"""Vehicle dynamics compensation.

- Understeer feedforward (Gillespie steady-state cornering):
      delta = L*kappa + K_us * a_y,   a_y = v^2 * kappa
  `understeer_feedforward` returns the extra road-wheel angle K_us*v^2*kappa to
  add to the kinematic angle.
- Steering actuator lag compensation. The steering servo is modelled as
  first-order lag tau plus pure delay T_d. Two tools:
    * `LagCompensator`: discrete lead filter u = r + tau * dr/dt (inverts the
      first-order lag), with a derivative filter and output bound.
    * `predict_pose`: propagate a kinematic bicycle T_d (+tau) forward so the
      tracker plans from where the car will be when the command lands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from cheesestack.core import Pose2D, wrap_angle

from .vehicle_params import VehicleParams


def understeer_feedforward(kappa: float, v: float, params: VehicleParams, k_us: float | None = None) -> float:
    """Extra road-wheel angle (rad) to hold `kappa` at speed `v` given understeer gradient."""
    kus = params.understeer_gradient if k_us is None else k_us
    return kus * v * v * kappa


def steady_state_curvature(delta_rad: float, v: float, params: VehicleParams) -> float:
    """Inverse: curvature achieved by road-wheel angle delta at speed v (small angle)."""
    return delta_rad / (params.wheelbase_m + params.understeer_gradient * v * v)


@dataclass
class LagCompensatorConfig:
    tau_s: float = 0.15  # identified actuator time constant
    deriv_filter_s: float = 0.05
    max_lead: float = float("inf")  # bound on |u - r|


class LagCompensator:
    """Approximate inverse of a first-order lag: u = r + tau * d/dt(r) (filtered).

    Feeding u into a lag 1/(tau s + 1) yields ~r with the high-frequency rolloff
    of the derivative filter. TODO(control): identify tau online from ActuatorMonitor.
    """

    def __init__(self, config: LagCompensatorConfig | None = None) -> None:
        self.config = config or LagCompensatorConfig()
        self._prev: float | None = None
        self._d = 0.0

    def reset(self) -> None:
        self._prev, self._d = None, 0.0

    def update(self, r: float, dt: float) -> float:
        c = self.config
        if self._prev is None:
            self._prev = r
        raw = (r - self._prev) / dt
        self._prev = r
        a = dt / (c.deriv_filter_s + dt) if c.deriv_filter_s > 0 else 1.0
        self._d += a * (raw - self._d)
        lead = max(-c.max_lead, min(c.max_lead, c.tau_s * self._d))
        return r + lead


def predict_pose(pose: Pose2D, v: float, curvature: float, horizon_s: float, steps: int = 10) -> Pose2D:
    """Integrate a kinematic unicycle (constant v, kappa) for `horizon_s` seconds."""
    x, y, th = pose.x, pose.y, pose.yaw
    dt = horizon_s / max(1, steps)
    for _ in range(max(1, steps)):
        th_mid = th + 0.5 * v * curvature * dt
        x += v * math.cos(th_mid) * dt
        y += v * math.sin(th_mid) * dt
        th += v * curvature * dt
    return Pose2D(x, y, wrap_angle(th))
