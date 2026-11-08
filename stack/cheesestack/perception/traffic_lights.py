"""Traffic-light recognition: per-frame HSV state classifier + temporal Bayes filter.

:class:`HSVTrafficLightClassifier` looks at a crop of a light housing
(the box usually comes from a detector or the map). It counts bright,
saturated pixels in red/yellow/green hue bands, weights them by the
vertical position prior of a standard vertical housing (red top, yellow
middle, green bottom; disable for horizontal lights), and returns a
probability per :class:`LightState`.

:class:`TrafficLightFilter` is a discrete Bayes (HMM forward) filter over
states with a transition model that encodes the usual cycle
(green -> yellow -> red -> green) and reports a state only when its
posterior exceeds a threshold, which suppresses flicker (LED PWM,
occlusion, single-frame misclassifications).

Limits: no arrow/turn semantics, no flashing detection, crop must be
given; colour bands assume roughly white-balanced images.
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cheesestack.core import BoundingBox2D, Header

from .detection import hue_mask, rgb_to_hsv


class LightState(enum.Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    UNKNOWN = "unknown"


STATES = (LightState.RED, LightState.YELLOW, LightState.GREEN, LightState.UNKNOWN)


@dataclass
class TrafficLightObservation:
    header: Header
    state: LightState
    confidence: float
    probs: dict = field(default_factory=dict)  # LightState -> prob
    box2d: Optional[BoundingBox2D] = None


class TrafficLightClassifier(abc.ABC):
    @abc.abstractmethod
    def classify(self, image: np.ndarray, header: Header,
                 box: Optional[BoundingBox2D] = None) -> TrafficLightObservation:
        ...


@dataclass
class HSVLightConfig:
    red_hue: tuple[float, float] = (0.93, 0.04)
    yellow_hue: tuple[float, float] = (0.06, 0.17)
    green_hue: tuple[float, float] = (0.25, 0.55)
    s_min: float = 0.4
    v_min: float = 0.55
    min_lit_frac: float = 0.01  # lit pixels / crop pixels below this -> UNKNOWN
    vertical_prior: bool = True
    prior_strength: float = 0.7  # 0 = ignore position, 1 = only count pixels in the expected third


class HSVTrafficLightClassifier(TrafficLightClassifier):
    def __init__(self, config: Optional[HSVLightConfig] = None) -> None:
        self.config = config or HSVLightConfig()

    def classify(self, image: np.ndarray, header: Header,
                 box: Optional[BoundingBox2D] = None) -> TrafficLightObservation:
        cfg = self.config
        crop = image
        if box is not None:
            h, w = image.shape[:2]
            u0, v0 = int(max(0, box.u)), int(max(0, box.v))
            crop = image[v0:min(h, int(box.v + box.h)), u0:min(w, int(box.u + box.w))]
        if crop.ndim != 3 or crop.size == 0:
            return TrafficLightObservation(header, LightState.UNKNOWN, 0.0, {LightState.UNKNOWN: 1.0}, box)
        hsv = rgb_to_hsv(crop)
        ch = crop.shape[0]
        rows = np.arange(ch)[:, None]
        thirds = {LightState.RED: rows < ch / 3, LightState.YELLOW: (rows >= ch / 3) & (rows < 2 * ch / 3),
                  LightState.GREEN: rows >= 2 * ch / 3}
        bands = {LightState.RED: cfg.red_hue, LightState.YELLOW: cfg.yellow_hue, LightState.GREEN: cfg.green_hue}
        counts = {}
        for st, (lo, hi) in bands.items():
            m = hue_mask(hsv, lo, hi, cfg.s_min, cfg.v_min)
            wgt = 1.0
            if cfg.vertical_prior:
                wgt = np.where(thirds[st], 1.0, 1.0 - cfg.prior_strength)
            counts[st] = float((m * wgt).sum())
        total = sum(counts.values())
        lit_frac = total / crop.shape[0] / crop.shape[1]
        if lit_frac < cfg.min_lit_frac:
            probs = {LightState.RED: 0.0, LightState.YELLOW: 0.0, LightState.GREEN: 0.0, LightState.UNKNOWN: 1.0}
            return TrafficLightObservation(header, LightState.UNKNOWN, 1.0, probs, box)
        lit_conf = float(np.clip(lit_frac / (5 * cfg.min_lit_frac), 0.0, 1.0))
        probs = {st: lit_conf * c / total for st, c in counts.items()}
        probs[LightState.UNKNOWN] = 1.0 - lit_conf
        best = max(probs, key=probs.get)
        return TrafficLightObservation(header, best, probs[best], probs, box)
        # TODO(perception): learned classifier (arrows, flashing, pedestrian heads), exposure-robust color model.


@dataclass
class LightFilterConfig:
    stay_prob: float = 0.9  # P(state unchanged per frame)
    report_threshold: float = 0.7
    obs_confusion: float = 0.15  # P(observing a wrong colour)
    unknown_obs_weight: float = 0.5  # how much an UNKNOWN observation flattens toward uniform


class TrafficLightFilter:
    """HMM forward filter over (RED, YELLOW, GREEN, UNKNOWN)."""

    def __init__(self, config: Optional[LightFilterConfig] = None) -> None:
        self.config = config or LightFilterConfig()
        self.T = self._transition()
        self.belief = np.full(4, 0.25)

    def _transition(self) -> np.ndarray:
        s = self.config.stay_prob
        o = 1.0 - s
        # rows: from, cols: to; order R, Y, G, U. Cycle G->Y->R->G favoured.
        T = np.array([
            [s, 0.05 * o, 0.80 * o, 0.15 * o],  # red -> green mostly
            [0.80 * o, s, 0.05 * o, 0.15 * o],  # yellow -> red
            [0.05 * o, 0.80 * o, s, 0.15 * o],  # green -> yellow
            [o / 3, o / 3, o / 3, s],
        ])
        return T / T.sum(axis=1, keepdims=True)

    def reset(self) -> None:
        self.belief = np.full(4, 0.25)

    def likelihood(self, obs: TrafficLightObservation) -> np.ndarray:
        c = self.config.obs_confusion
        p = np.array([obs.probs.get(st, 0.0) for st in STATES])
        if obs.state == LightState.UNKNOWN:
            w = self.config.unknown_obs_weight
            return (1 - w) * np.ones(4) + w * np.array([0.2, 0.2, 0.2, 1.0])
        return c + (1 - c) * p

    def update(self, obs: TrafficLightObservation) -> TrafficLightObservation:
        pred = self.belief @ self.T
        post = pred * self.likelihood(obs)
        post /= post.sum()
        self.belief = post
        i = int(np.argmax(post))
        st = STATES[i] if post[i] >= self.config.report_threshold else LightState.UNKNOWN
        return TrafficLightObservation(obs.header, st, float(post[i]),
                                       {s: float(p) for s, p in zip(STATES, post)}, obs.box2d)
