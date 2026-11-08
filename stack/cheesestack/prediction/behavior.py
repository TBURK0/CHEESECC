"""Behavioral (intent) forecasting with a recursive Bayesian / HMM-forward baseline.

Intents: KEEP_LANE, TURN_LEFT, TURN_RIGHT, STOP, YIELD, CUT_IN.

Per observation we build ``BehaviorFeatures`` (speed, longitudinal accel, yaw
rate, and - if an ego lane centerline is given - lateral offset to and lateral
velocity toward the ego lane). Each intent has a diagonal-Gaussian likelihood
over (accel, yaw_rate, lat_vel_toward_ego), plus soft gates (STOP favoured at
low speed, CUT_IN requires being outside but near the ego lane). The belief is
updated per frame with a sticky transition matrix:

    b_t ∝ L(f_t | intent) * (T^T b_{t-1})

``IntentModel`` is the interface a learned classifier would implement;
``GaussianIntentModel`` is the hand-tuned baseline.

Limits: hand-set parameters, no map semantics (stop lines, crosswalks, turn
lanes), no interaction with ego beyond lateral approach.
"""

from __future__ import annotations

import enum
import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from cheesestack.core import TrackedObject

from .lane_change import Lane, project_to_lane


class Intent(enum.Enum):
    KEEP_LANE = "keep_lane"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    STOP = "stop"
    YIELD = "yield"
    CUT_IN = "cut_in"


INTENTS = list(Intent)


@dataclass
class BehaviorFeatures:
    speed: float
    accel: float
    yaw_rate: float
    lat_offset: Optional[float] = None  # signed offset from ego lane center (+ left)
    lat_vel_toward_ego: float = 0.0  # >0 means approaching ego lane center


def features_from_history(history: Sequence[TrackedObject], ego_lane: Optional[Lane] = None) -> BehaviorFeatures:
    """Finite-difference features from >= 1 chronologically ordered observations of one track."""
    last = history[-1]
    v = float(np.hypot(*last.velocity[:2]))
    accel = yaw_rate = 0.0
    if len(history) >= 2:
        t = np.array([h.header.stamp for h in history])
        t = t - t[-1]
        if np.ptp(t) > 1e-6:
            speeds = np.array([np.hypot(*h.velocity[:2]) for h in history])
            yaws = np.unwrap([math.atan2(h.velocity[1], h.velocity[0]) if np.hypot(*h.velocity[:2]) > 0.3
                              else h.pose.yaw for h in history])
            accel = float(np.polyfit(t, speeds, 1)[0])
            yaw_rate = float(np.polyfit(t, yaws, 1)[0])
    lat, lat_vel = None, 0.0
    if ego_lane is not None:
        pr = project_to_lane(ego_lane, (last.pose.x, last.pose.y))
        lat = pr.d
        n = np.array([-math.sin(pr.heading), math.cos(pr.heading)])  # left normal
        d_dot = float(n @ np.asarray(last.velocity[:2], float))
        lat_vel = -math.copysign(1.0, lat) * d_dot if abs(lat) > 1e-6 else 0.0
    return BehaviorFeatures(v, accel, yaw_rate, lat, lat_vel)


class IntentModel(ABC):
    """Returns per-intent log-likelihoods (len(INTENTS),) for one feature frame."""

    @abstractmethod
    def log_likelihood(self, f: BehaviorFeatures) -> np.ndarray:
        ...


def _default_params() -> dict:
    # (mean, std) over accel [m/s^2], yaw_rate [rad/s], lat_vel_toward_ego [m/s]
    return {
        Intent.KEEP_LANE: ((0.0, 0.8), (0.0, 0.06), (0.0, 0.25)),
        Intent.TURN_LEFT: ((-0.5, 1.2), (0.3, 0.12), (0.0, 0.8)),
        Intent.TURN_RIGHT: ((-0.5, 1.2), (-0.3, 0.12), (0.0, 0.8)),
        Intent.STOP: ((-2.5, 1.0), (0.0, 0.08), (0.0, 0.3)),
        Intent.YIELD: ((-1.0, 0.5), (0.0, 0.08), (0.0, 0.3)),
        Intent.CUT_IN: ((0.0, 1.0), (0.0, 0.15), (0.8, 0.35)),
    }


@dataclass
class GaussianIntentConfig:
    params: dict = field(default_factory=_default_params)
    stop_speed: float = 1.0  # below this speed, STOP likelihood is boosted
    cut_in_band: tuple = (1.0, 5.5)  # |lateral offset| range where cut-in is possible [m]
    gate_penalty: float = -6.0  # log-likelihood penalty when a gate fails


class GaussianIntentModel(IntentModel):
    def __init__(self, config: Optional[GaussianIntentConfig] = None):
        self.cfg = config or GaussianIntentConfig()

    def log_likelihood(self, f: BehaviorFeatures) -> np.ndarray:
        x = np.array([f.accel, f.yaw_rate, f.lat_vel_toward_ego])
        out = np.zeros(len(INTENTS))
        for k, intent in enumerate(INTENTS):
            (ma, sa), (mw, sw), (ml, sl) = self.cfg.params[intent]
            mu, sd = np.array([ma, mw, ml]), np.array([sa, sw, sl])
            if f.lat_offset is None:  # no ego-lane info: drop lateral dimension
                mu, sd, xx = mu[:2], sd[:2], x[:2]
            else:
                xx = x
            out[k] = float(np.sum(-0.5 * ((xx - mu) / sd) ** 2 - np.log(sd)))
        # Speed gate: a (nearly) stopped agent is most likely STOP; turning needs motion.
        i_stop = INTENTS.index(Intent.STOP)
        if f.speed < self.cfg.stop_speed:
            out[i_stop] += 3.0
            out[INTENTS.index(Intent.TURN_LEFT)] += self.cfg.gate_penalty * 0.5
            out[INTENTS.index(Intent.TURN_RIGHT)] += self.cfg.gate_penalty * 0.5
        # Cut-in gate: must be beside (not in) the ego lane.
        lo, hi = self.cfg.cut_in_band
        if f.lat_offset is None or not (lo <= abs(f.lat_offset) <= hi):
            out[INTENTS.index(Intent.CUT_IN)] += self.cfg.gate_penalty
        return out


@dataclass
class BehaviorForecasterConfig:
    stickiness: float = 0.9  # P(intent_t == intent_{t-1})
    prior: Optional[dict] = None  # Intent -> prior prob; default favours KEEP_LANE
    history_len: int = 8
    likelihood_weight: float = 1.0  # tempering exponent on likelihoods (< 1 = smoother)


@dataclass
class IntentForecast:
    track_id: int
    probabilities: dict  # Intent -> prob

    @property
    def most_likely(self) -> Intent:
        return max(self.probabilities, key=self.probabilities.get)


class BehaviorForecaster:
    """Per-track recursive intent filter."""

    def __init__(self, config: Optional[BehaviorForecasterConfig] = None, model: Optional[IntentModel] = None):
        self.cfg = config or BehaviorForecasterConfig()
        self.model = model or GaussianIntentModel()
        K = len(INTENTS)
        s = self.cfg.stickiness
        self.T = np.full((K, K), (1.0 - s) / (K - 1))
        np.fill_diagonal(self.T, s)
        pr = self.cfg.prior or {Intent.KEEP_LANE: 0.5, Intent.TURN_LEFT: 0.1, Intent.TURN_RIGHT: 0.1,
                                Intent.STOP: 0.1, Intent.YIELD: 0.1, Intent.CUT_IN: 0.1}
        p0 = np.array([pr.get(i, 1e-3) for i in INTENTS], dtype=float)
        self.prior = p0 / p0.sum()
        self._belief: dict = {}
        self._hist: dict = {}

    def reset(self, track_id: Optional[int] = None) -> None:
        if track_id is None:
            self._belief.clear()
            self._hist.clear()
        else:
            self._belief.pop(track_id, None)
            self._hist.pop(track_id, None)

    def update_features(self, track_id: int, f: BehaviorFeatures) -> IntentForecast:
        b = self._belief.get(track_id, self.prior)
        pred = self.T.T @ b
        ll = self.cfg.likelihood_weight * self.model.log_likelihood(f)
        post = pred * np.exp(ll - ll.max())
        post = post / post.sum() if post.sum() > 0 else self.prior.copy()
        self._belief[track_id] = post
        return IntentForecast(track_id, {i: float(p) for i, p in zip(INTENTS, post)})

    def update(self, obj: TrackedObject, ego_lane: Optional[Lane] = None) -> IntentForecast:
        """Feed one observation of a track; returns the posterior intent distribution."""
        h = self._hist.setdefault(obj.track_id, deque(maxlen=self.cfg.history_len))
        if h and obj.header.stamp <= h[-1].header.stamp:
            b = self._belief.get(obj.track_id, self.prior)
            return IntentForecast(obj.track_id, {i: float(p) for i, p in zip(INTENTS, b)})
        h.append(obj)
        # TODO(prediction): feed intents back into ObjectPredictor hypothesis priors; add map
        # features (distance to stop line / crosswalk / junction) and a learned IntentModel.
        return self.update_features(obj.track_id, features_from_history(list(h), ego_lane))
