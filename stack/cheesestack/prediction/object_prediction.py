"""Multi-hypothesis trajectory prediction for tracked objects.

For each ``core.TrackedObject`` the predictor:

1. builds a 6-D state ``[x, y, yaw, v, yaw_rate, a]``. Heading comes from the
   velocity vector when moving (else ``pose.yaw``); yaw rate and acceleration
   are finite-differenced from a short per-track history (``observe``);
2. converts the track's (x, y, vx, vy) covariance to (x, y, yaw, v) through the
   polar-transform Jacobian;
3. looks up a per-class list of ``HypothesisSpec`` (e.g. vehicles: keep going,
   turn left, turn right, brake; pedestrians: walk on, stop);
4. rolls each hypothesis out with its motion model and scores it:
   ``p_h ∝ prior_h * N(yaw_rate_obs; w_h, σ_w) * N(a_obs; a_h, σ_a)``.

Limits: hypotheses are kinematic, not map-aware; the likelihood uses only the
current yaw-rate / accel estimates.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cheesestack.core import ObjectClass, TrackedObject

from .motion_models import (IA, IW, MotionModel, MotionModelConfig, MotionModelType,
                            PredictedTrajectory)


@dataclass
class HypothesisSpec:
    """A motion hypothesis. ``yaw_rate`` / ``accel`` of None mean "use the estimate"."""

    label: str
    model: MotionModelType
    prior: float
    yaw_rate: Optional[float] = None
    accel: Optional[float] = None


def _vehicle_hyps() -> list:
    return [
        HypothesisSpec("keep", MotionModelType.CTRA, 0.6),
        HypothesisSpec("turn_left", MotionModelType.CTRV, 0.12, yaw_rate=0.3),
        HypothesisSpec("turn_right", MotionModelType.CTRV, 0.12, yaw_rate=-0.3),
        HypothesisSpec("brake", MotionModelType.CA, 0.16, yaw_rate=0.0, accel=-3.0),
    ]


def _vru_hyps() -> list:
    return [
        HypothesisSpec("keep", MotionModelType.CV, 0.7),
        HypothesisSpec("stop", MotionModelType.CA, 0.3, yaw_rate=0.0, accel=-1.5),
    ]


def _default_class_hyps() -> dict:
    veh = _vehicle_hyps()
    return {
        ObjectClass.CAR: veh, ObjectClass.TRUCK: veh, ObjectClass.BUS: veh,
        ObjectClass.MOTORCYCLE: veh,
        ObjectClass.BICYCLE: [
            HypothesisSpec("keep", MotionModelType.CTRV, 0.6),
            HypothesisSpec("turn_left", MotionModelType.CTRV, 0.15, yaw_rate=0.4),
            HypothesisSpec("turn_right", MotionModelType.CTRV, 0.15, yaw_rate=-0.4),
            HypothesisSpec("stop", MotionModelType.CA, 0.1, yaw_rate=0.0, accel=-2.0),
        ],
        ObjectClass.PEDESTRIAN: _vru_hyps(),
        ObjectClass.ANIMAL: _vru_hyps(),
        ObjectClass.UNKNOWN: [HypothesisSpec("keep", MotionModelType.CV, 1.0)],
    }


@dataclass
class ObjectPredictorConfig:
    horizon: float = 5.0
    dt: float = 0.1
    stationary_speed: float = 0.3  # below this: single stationary hypothesis
    history_len: int = 10
    sigma_yaw_rate: float = 0.12  # likelihood width for hypothesis scoring
    sigma_accel: float = 1.5
    min_probability: float = 0.02  # prune hypotheses below this (then renormalize)
    max_hypotheses: int = 4
    static_classes: tuple = (ObjectClass.STATIC_OBSTACLE, ObjectClass.TRAFFIC_SIGN, ObjectClass.TRAFFIC_LIGHT)
    class_hypotheses: dict = field(default_factory=_default_class_hyps)
    motion: MotionModelConfig = field(default_factory=MotionModelConfig)
    # Process noise is scaled up for VRUs (more erratic).
    vru_noise_scale: float = 2.0


@dataclass
class ObjectPrediction:
    track_id: int
    cls: ObjectClass
    length: float
    width: float
    hypotheses: list  # list[PredictedTrajectory], probabilities sum to 1

    @property
    def most_likely(self) -> PredictedTrajectory:
        return max(self.hypotheses, key=lambda h: h.probability)


def tracked_object_state(obj: TrackedObject, stationary_speed: float = 0.3):
    """Return (state6, cov6) for a tracked object (yaw_rate, a set to 0)."""
    vx, vy = float(obj.velocity[0]), float(obj.velocity[1])
    v = math.hypot(vx, vy)
    yaw = math.atan2(vy, vx) if v > stationary_speed else obj.pose.yaw
    s = np.array([obj.pose.x, obj.pose.y, yaw, v, 0.0, 0.0])
    cov = np.diag([0.3 ** 2, 0.3 ** 2, 0.1 ** 2, 0.5 ** 2, 0.1 ** 2, 1.0 ** 2])
    if obj.covariance is not None:
        C = np.asarray(obj.covariance, dtype=float)
        # (x, y, vx, vy) -> (x, y, yaw, v) via polar Jacobian
        J = np.zeros((4, 4))
        J[0, 0] = J[1, 1] = 1.0
        if v > 1e-3:
            J[2, 2], J[2, 3] = -vy / v ** 2, vx / v ** 2
            J[3, 2], J[3, 3] = vx / v, vy / v
        else:
            J[3, 2] = J[3, 3] = 1.0 / math.sqrt(2)
        C4 = J @ C @ J.T
        if v <= 1e-3:
            C4[2, 2] = max(C4[2, 2], 0.1 ** 2)
        cov[:4, :4] = C4
    return s, cov


class ObjectPredictor:
    """Per-class, multi-hypothesis predictor for ``core.TrackedObject``."""

    def __init__(self, config: Optional[ObjectPredictorConfig] = None):
        self.cfg = config or ObjectPredictorConfig()
        self._history: dict = {}

    # -- history / derivative estimation -------------------------------------------------
    def observe(self, obj: TrackedObject) -> None:
        """Record a track observation for yaw-rate / acceleration estimation."""
        h = self._history.setdefault(obj.track_id, deque(maxlen=self.cfg.history_len))
        s, _ = tracked_object_state(obj, self.cfg.stationary_speed)
        if h and obj.header.stamp <= h[-1][0]:
            return
        h.append((obj.header.stamp, s[2], s[3]))

    def forget(self, track_id: int) -> None:
        self._history.pop(track_id, None)

    def estimate_derivatives(self, track_id: int):
        """Least-squares slope of unwrapped yaw and speed over the history -> (yaw_rate, accel)."""
        h = self._history.get(track_id)
        if not h or len(h) < 2:
            return 0.0, 0.0
        arr = np.array(h)
        t = arr[:, 0] - arr[-1, 0]
        if np.ptp(t) < 1e-6:
            return 0.0, 0.0
        yaw = np.unwrap(arr[:, 1])
        w = float(np.polyfit(t, yaw, 1)[0])
        a = float(np.polyfit(t, arr[:, 2], 1)[0])
        return w, a

    # -- prediction ----------------------------------------------------------------------
    def _motion_cfg(self, cls: ObjectClass) -> MotionModelConfig:
        m = self.cfg.motion
        mc = MotionModelConfig(**{**m.__dict__, "dt": self.cfg.dt, "horizon": self.cfg.horizon})
        if cls in (ObjectClass.PEDESTRIAN, ObjectClass.ANIMAL, ObjectClass.BICYCLE):
            k = self.cfg.vru_noise_scale
            for name in ("q_cv", "q_ca", "q_ctrv", "q_ctra"):
                setattr(mc, name, tuple(k * q for q in getattr(m, name)))
            mc.max_yaw_rate = 3.0
        return mc

    def predict(self, obj: TrackedObject, yaw_rate: Optional[float] = None,
                accel: Optional[float] = None) -> ObjectPrediction:
        """Predict all hypotheses for ``obj``. Explicit yaw_rate/accel override history estimates."""
        s0, P0 = tracked_object_state(obj, self.cfg.stationary_speed)
        w_est, a_est = self.estimate_derivatives(obj.track_id)
        w_obs = w_est if yaw_rate is None else yaw_rate
        a_obs = a_est if accel is None else accel
        mcfg = self._motion_cfg(obj.cls)
        common = dict(track_id=obj.track_id, stamp=obj.header.stamp, frame_id=obj.header.frame_id)

        if obj.cls in self.cfg.static_classes or s0[3] < self.cfg.stationary_speed:
            s = s0.copy()
            s[3] = 0.0
            P = P0.copy()
            P[3, 3] = min(P[3, 3], 0.1 ** 2)
            traj = MotionModel(MotionModelType.CV, mcfg).rollout(s, P, label="stationary", **common)
            traj.probability = 1.0
            return ObjectPrediction(obj.track_id, obj.cls, obj.length, obj.width, [traj])

        specs = self.cfg.class_hypotheses.get(obj.cls, self.cfg.class_hypotheses[ObjectClass.UNKNOWN])
        hyps, weights = [], []
        for spec in specs:
            model = MotionModel(spec.model, mcfg)
            s = s0.copy()
            s[IW] = w_obs if spec.yaw_rate is None else spec.yaw_rate
            s[IA] = a_obs if spec.accel is None else spec.accel
            # Likelihood of the observed derivatives under the hypothesis (only for fixed values).
            lw = 0.0 if spec.yaw_rate is None else -0.5 * ((w_obs - spec.yaw_rate) / self.cfg.sigma_yaw_rate) ** 2
            la = 0.0 if spec.accel is None else -0.5 * ((a_obs - spec.accel) / self.cfg.sigma_accel) ** 2
            if spec.yaw_rate is None:
                # Free hypothesis: penalise in proportion to how far the evidence is from "nominal".
                lw = -0.5 * (w_obs / (3 * self.cfg.sigma_yaw_rate)) ** 2
            if spec.accel is None:
                la = -0.5 * (min(a_obs, 0.0) / (2 * self.cfg.sigma_accel)) ** 2
            weights.append(math.log(max(spec.prior, 1e-12)) + lw + la)
            hyps.append(model.rollout(s, P0, label=spec.label, **common))

        lw = np.array(weights)
        p = np.exp(lw - lw.max())
        p /= p.sum()
        order = np.argsort(-p)[: self.cfg.max_hypotheses]
        keep = [i for i in order if p[i] >= self.cfg.min_probability] or [int(order[0])]
        total = sum(p[i] for i in keep)
        out = []
        for i in keep:
            hyps[i].probability = float(p[i] / total)
            out.append(hyps[i])
        # TODO(prediction): map-aware hypotheses (lane-following, turn lanes) and a learned
        # multimodal predictor (e.g. MTR / Wayformer) behind the same ObjectPrediction output.
        return ObjectPrediction(obj.track_id, obj.cls, obj.length, obj.width, out)

    def predict_all(self, objs: list) -> list:
        for o in objs:
            self.observe(o)
        return [self.predict(o) for o in objs]
