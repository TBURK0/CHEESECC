"""Kinematic motion models rolled out with EKF-style uncertainty growth.

All models share one 6-D state ``[x, y, yaw, v, yaw_rate, a]`` (map frame, SI).
Each model decides which of ``yaw_rate`` / ``a`` it keeps; the rest are zeroed:

==========  ===========  =========
model       yaw_rate     accel
==========  ===========  =========
CV          0            0
CA          0            kept
CTRV        kept         0
CTRA        kept         kept
==========  ===========  =========

Propagation is the exact constant-turn-rate arc using the step-average speed
(``v + a*dt/2``), which is exact for CV/CTRV and second-order accurate for
CA/CTRA. Speed is clamped at zero (no reversing) unless configured otherwise.

Uncertainty: covariance is propagated as ``P' = F P F^T + Q*dt`` with a
numerical Jacobian ``F`` and a per-model diagonal process-noise density ``Q``.
The output keeps the 2x2 position block per step for downstream consumers.

Limits: no road/map constraints, no interaction between agents, Gaussian
uncertainty only (no multimodality - see ``object_prediction`` for that).
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cheesestack.core import Header, Trajectory, TrajectoryPoint, VehicleState, wrap_angle

STATE_DIM = 6
IX, IY, IYAW, IV, IW, IA = range(STATE_DIM)


class MotionModelType(enum.Enum):
    CV = "cv"
    CA = "ca"
    CTRV = "ctrv"
    CTRA = "ctra"


@dataclass
class MotionModelConfig:
    """Rollout parameters and process-noise spectral densities (per second)."""

    dt: float = 0.1
    horizon: float = 5.0
    allow_reverse: bool = False
    max_speed: float = 60.0
    max_yaw_rate: float = 1.5
    # Process-noise densities, diag over [x, y, yaw, v, yaw_rate, a].
    q_cv: tuple = (0.05, 0.05, 0.002, 1.0, 0.0, 0.0)
    q_ca: tuple = (0.05, 0.05, 0.002, 0.5, 0.0, 1.0)
    q_ctrv: tuple = (0.05, 0.05, 0.002, 1.0, 0.1, 0.0)
    q_ctra: tuple = (0.05, 0.05, 0.002, 0.5, 0.1, 1.0)
    # Default initial std-devs used when no covariance is supplied.
    init_std: tuple = (0.3, 0.3, 0.05, 0.5, 0.05, 0.5)


@dataclass
class PredictedTrajectory:
    """One predicted future for one agent.

    ``states`` is (N, 6) over ``[x, y, yaw, v, yaw_rate, a]``; ``times`` are
    seconds relative to ``stamp``; ``pos_cov`` is (N, 2, 2) position covariance.
    """

    times: np.ndarray
    states: np.ndarray
    pos_cov: np.ndarray
    probability: float = 1.0
    model: str = ""
    label: str = ""
    track_id: int = -1
    stamp: float = 0.0
    frame_id: str = "map"
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.times)

    @property
    def positions(self) -> np.ndarray:
        return self.states[:, :2]

    def state_at(self, t: float) -> np.ndarray:
        """Linearly interpolated state at relative time ``t`` (clamped to range)."""
        t = float(np.clip(t, self.times[0], self.times[-1]))
        out = np.array([np.interp(t, self.times, self.states[:, k]) for k in range(STATE_DIM)])
        yaw = np.unwrap(self.states[:, IYAW])
        out[IYAW] = wrap_angle(float(np.interp(t, self.times, yaw)))
        return out

    def cov_at(self, t: float) -> np.ndarray:
        t = float(np.clip(t, self.times[0], self.times[-1]))
        flat = self.pos_cov.reshape(len(self.times), 4)
        return np.array([np.interp(t, self.times, flat[:, k]) for k in range(4)]).reshape(2, 2)

    def resample(self, times: np.ndarray) -> "PredictedTrajectory":
        times = np.asarray(times, dtype=float)
        states = np.stack([self.state_at(t) for t in times])
        cov = np.stack([self.cov_at(t) for t in times])
        return PredictedTrajectory(times, states, cov, self.probability, self.model, self.label,
                                   self.track_id, self.stamp, self.frame_id, dict(self.meta))

    def to_trajectory(self) -> Trajectory:
        """Convert to the cross-layer ``core.Trajectory`` (drops covariance)."""
        pts = []
        for t, s in zip(self.times, self.states):
            v = float(s[IV])
            curv = float(s[IW] / v) if abs(v) > 0.1 else 0.0
            pts.append(TrajectoryPoint(float(t), float(s[IX]), float(s[IY]), float(s[IYAW]), v, float(s[IA]), curv))
        return Trajectory(Header(self.stamp, self.frame_id, source="prediction"), pts)


class MotionModel:
    """Kinematic rollout with covariance propagation for one model type."""

    def __init__(self, model_type: MotionModelType = MotionModelType.CV, config: Optional[MotionModelConfig] = None):
        self.type = model_type
        self.cfg = config or MotionModelConfig()
        q = {MotionModelType.CV: self.cfg.q_cv, MotionModelType.CA: self.cfg.q_ca,
             MotionModelType.CTRV: self.cfg.q_ctrv, MotionModelType.CTRA: self.cfg.q_ctra}[model_type]
        self.Q = np.diag(np.asarray(q, dtype=float))

    @property
    def keeps_yaw_rate(self) -> bool:
        return self.type in (MotionModelType.CTRV, MotionModelType.CTRA)

    @property
    def keeps_accel(self) -> bool:
        return self.type in (MotionModelType.CA, MotionModelType.CTRA)

    def condition(self, state: np.ndarray) -> np.ndarray:
        """Zero the components this model does not carry, clamp to limits."""
        s = np.asarray(state, dtype=float).copy()
        if not self.keeps_yaw_rate:
            s[IW] = 0.0
        if not self.keeps_accel:
            s[IA] = 0.0
        s[IW] = float(np.clip(s[IW], -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate))
        return s

    def step(self, s: np.ndarray, dt: float) -> np.ndarray:
        x, y, yaw, v, w, a = s
        v_end = v + a * dt
        lo = -self.cfg.max_speed if self.cfg.allow_reverse else 0.0
        if not self.cfg.allow_reverse and v_end < 0.0 <= v and a < 0.0:
            # Stops mid-step: distance = v^2 / (2|a|), travelled over t_stop.
            t_move = v / -a if a != 0 else dt
            v_avg = 0.5 * v
            v_end = 0.0
        else:
            t_move = dt
            v_end = float(np.clip(v_end, lo, self.cfg.max_speed))
            v_avg = 0.5 * (v + v_end)
        dyaw = w * t_move
        if abs(w) > 1e-6:
            r = v_avg / w
            nx = x + r * (math.sin(yaw + dyaw) - math.sin(yaw))
            ny = y + r * (-math.cos(yaw + dyaw) + math.cos(yaw))
        else:
            nx = x + v_avg * t_move * math.cos(yaw)
            ny = y + v_avg * t_move * math.sin(yaw)
        new_a = 0.0 if (v_end == 0.0 and not self.cfg.allow_reverse) else a
        return np.array([nx, ny, yaw + dyaw, v_end, w, new_a])

    def jacobian(self, s: np.ndarray, dt: float, eps: float = 1e-5) -> np.ndarray:
        F = np.zeros((STATE_DIM, STATE_DIM))
        for k in range(STATE_DIM):
            d = np.zeros(STATE_DIM)
            d[k] = eps
            F[:, k] = (self.step(s + d, dt) - self.step(s - d, dt)) / (2 * eps)
        return F

    def rollout(self, state: np.ndarray, cov: Optional[np.ndarray] = None,
                horizon: Optional[float] = None, dt: Optional[float] = None,
                **kw) -> PredictedTrajectory:
        """Roll the model forward from ``state`` (6,) with covariance ``cov`` (6, 6)."""
        dt = dt or self.cfg.dt
        horizon = self.cfg.horizon if horizon is None else horizon
        n = int(round(horizon / dt)) + 1
        s = self.condition(state)
        P = np.diag(np.square(self.cfg.init_std)) if cov is None else np.asarray(cov, dtype=float).copy()
        mask = np.ones(STATE_DIM)
        if not self.keeps_yaw_rate:
            mask[IW] = 0.0
        if not self.keeps_accel:
            mask[IA] = 0.0
        P = P * np.outer(mask, mask)
        times = np.arange(n) * dt
        states = np.zeros((n, STATE_DIM))
        covs = np.zeros((n, 2, 2))
        states[0], covs[0] = s, P[:2, :2]
        for i in range(1, n):
            F = self.jacobian(s, dt)
            s = self.step(s, dt)
            P = F @ P @ F.T + self.Q * dt
            P = 0.5 * (P + P.T)
            states[i], covs[i] = s, P[:2, :2]
        states[:, IYAW] = [wrap_angle(a) for a in states[:, IYAW]]
        return PredictedTrajectory(times, states, covs, model=self.type.value, **kw)


@dataclass
class EgoPredictorConfig:
    model: MotionModelType = MotionModelType.CTRA
    motion: MotionModelConfig = field(default_factory=MotionModelConfig)
    # Use v * curvature when the state's yaw_rate is ~0 but curvature is set.
    use_curvature_fallback: bool = True


class EgoPredictor:
    """Predict the ego's own future from ``core.VehicleState``."""

    def __init__(self, config: Optional[EgoPredictorConfig] = None):
        self.cfg = config or EgoPredictorConfig()
        self.model = MotionModel(self.cfg.model, self.cfg.motion)

    def state_vector(self, vs: VehicleState) -> np.ndarray:
        w = vs.yaw_rate
        if self.cfg.use_curvature_fallback and abs(w) < 1e-6 and abs(vs.curvature) > 0:
            w = vs.v * vs.curvature
        return np.array([vs.pose.x, vs.pose.y, vs.pose.yaw, vs.v, w, vs.a])

    def predict(self, vs: VehicleState, horizon: Optional[float] = None) -> PredictedTrajectory:
        cov = None
        if vs.covariance is not None:
            c = np.asarray(vs.covariance, dtype=float)
            cov = np.diag(np.square(self.cfg.motion.init_std))
            k = min(c.shape[0], STATE_DIM)
            cov[:k, :k] = c[:k, :k]
        # TODO(prediction): blend with the planner's intended trajectory once available.
        return self.model.rollout(self.state_vector(vs), cov, horizon=horizon, label="ego",
                                  stamp=vs.header.stamp, frame_id=vs.header.frame_id)
