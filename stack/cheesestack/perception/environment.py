"""Environmental condition estimation from a single camera frame (heuristics).

Measures (all on luma, float [0,1]):
- brightness: mean luma; contrast: RMS contrast (std); dynamic range: p99 - p1.
- sharpness: variance of the 4-neighbour Laplacian (low => blur/fog/defocus).
- glare: fraction of near-saturated pixels, weighted by how clustered they are
  in the upper image (sun / headlights).
- night: low brightness with sparse bright point sources.
- rain: heuristics on high-frequency streak energy (vertical-vs-horizontal
  gradient anisotropy) combined with low sharpness/contrast (droplets on lens).
- fog: low contrast + low dynamic range at normal brightness.

Each yields a score in [0, 1]; `EnvironmentEstimator` smooths scores with an EMA
and maps them to a primary `Condition` plus a `visibility` in [0,1] that
downstream confidence scoring consumes.

Limits: hand-tuned thresholds for a forward road camera; these are proxies, not
weather classifiers.
# TODO(perception): learned scene-condition classifier behind `ConditionModel`,
#   use exposure/ISO/gain metadata, wiper/rain-sensor CAN inputs, sun-position prior.
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .preprocessing import grayscale


class Condition(enum.Enum):
    NOMINAL = "nominal"
    NIGHT = "night"
    GLARE = "glare"
    LOW_LIGHT = "low_light"
    OVEREXPOSED = "overexposed"
    BLUR = "blur"
    FOG = "fog"
    RAIN = "rain"
    BLOCKED = "blocked"  # lens covered / uniform frame


@dataclass
class ImageStats:
    brightness: float
    contrast: float
    dynamic_range: float
    sharpness: float
    saturated_frac: float
    dark_frac: float
    upper_saturated_frac: float
    grad_anisotropy: float  # (|gy| - |gx|) / (|gx| + |gy|) high-pass energy, in [-1, 1]


def image_stats(img: np.ndarray, saturation_level: float = 0.97, dark_level: float = 0.06) -> ImageStats:
    g = grayscale(img).astype(np.float64)
    H, W = g.shape
    p1, p99 = np.percentile(g, [1, 99])
    lap = np.zeros_like(g)
    lap[1:-1, 1:-1] = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1])
    sharp = float(lap[1:-1, 1:-1].var()) if H > 2 and W > 2 else 0.0
    sat = g >= saturation_level
    gx = np.abs(np.diff(g, axis=1)).mean() if W > 1 else 0.0
    gy = np.abs(np.diff(g, axis=0)).mean() if H > 1 else 0.0
    aniso = float((gx - gy) / (gx + gy)) if gx + gy > 1e-9 else 0.0  # >0: vertical structures (streaks)
    return ImageStats(
        brightness=float(g.mean()), contrast=float(g.std()), dynamic_range=float(p99 - p1),
        sharpness=sharp, saturated_frac=float(sat.mean()), dark_frac=float((g <= dark_level).mean()),
        upper_saturated_frac=float(sat[: max(1, H // 2)].mean()), grad_anisotropy=aniso,
    )


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 at lo, 1 at hi (works for lo > hi as a decreasing ramp)."""
    if hi == lo:
        return float(x >= hi)
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


@dataclass
class EnvironmentConfig:
    ema_alpha: float = 0.3  # 1.0 = no smoothing
    night_brightness: tuple[float, float] = (0.25, 0.08)  # ramp: brighter -> darker
    low_light_brightness: tuple[float, float] = (0.35, 0.15)
    overexposed_brightness: tuple[float, float] = (0.75, 0.92)
    glare_saturation: tuple[float, float] = (0.01, 0.08)
    blur_sharpness: tuple[float, float] = (2e-3, 2e-4)  # Laplacian variance ramp
    fog_contrast: tuple[float, float] = (0.12, 0.04)
    fog_range: tuple[float, float] = (0.45, 0.15)
    rain_anisotropy: tuple[float, float] = (0.15, 0.5)
    blocked_contrast: tuple[float, float] = (0.02, 0.005)
    condition_threshold: float = 0.5  # minimum score to declare a non-nominal condition


@dataclass
class EnvironmentEstimate:
    stamp: float
    condition: Condition
    scores: dict[str, float]
    visibility: float  # [0,1], 1 = clear
    stats: Optional[ImageStats] = None

    def score(self, c: Condition) -> float:
        return self.scores.get(c.value, 0.0)


class ConditionModel(abc.ABC):
    """Interface for a learned condition model returning per-condition scores."""

    @abc.abstractmethod
    def predict(self, img: np.ndarray) -> dict[str, float]: ...


class HeuristicConditionModel(ConditionModel):
    def __init__(self, config: Optional[EnvironmentConfig] = None) -> None:
        self.config = config or EnvironmentConfig()
        self.last_stats: Optional[ImageStats] = None

    def predict(self, img: np.ndarray) -> dict[str, float]:
        c = self.config
        s = image_stats(img)
        self.last_stats = s
        blocked = _ramp(s.contrast, *c.blocked_contrast)
        dark = _ramp(s.brightness, *c.night_brightness)
        # Night: dark scene, optionally with point lights (some saturated pixels raise it).
        night = float(np.clip(dark * (0.8 + 0.2 * _ramp(s.saturated_frac, 0.0, 0.01)), 0, 1)) * (1 - blocked)
        low_light = _ramp(s.brightness, *c.low_light_brightness) * (1 - night)
        over = _ramp(s.brightness, *c.overexposed_brightness)
        glare = _ramp(s.upper_saturated_frac, *c.glare_saturation) * (1 - dark) * (1 - over * 0.5)
        blur = _ramp(s.sharpness, *c.blur_sharpness) * (1 - blocked)
        normal_light = 1 - max(dark, over)
        fog = min(_ramp(s.contrast, *c.fog_contrast), _ramp(s.dynamic_range, *c.fog_range)) * normal_light * (1 - blocked)
        rain = _ramp(s.grad_anisotropy, *c.rain_anisotropy) * (1 - blocked) * (1 - dark)
        return {
            Condition.NIGHT.value: night, Condition.LOW_LIGHT.value: low_light,
            Condition.OVEREXPOSED.value: over, Condition.GLARE.value: glare,
            Condition.BLUR.value: blur, Condition.FOG.value: fog, Condition.RAIN.value: rain,
            Condition.BLOCKED.value: blocked,
        }


# How much each condition degrades visibility at score 1.
VISIBILITY_PENALTY = {
    Condition.NIGHT.value: 0.4, Condition.LOW_LIGHT.value: 0.25, Condition.OVEREXPOSED.value: 0.5,
    Condition.GLARE.value: 0.5, Condition.BLUR.value: 0.5, Condition.FOG.value: 0.6,
    Condition.RAIN.value: 0.4, Condition.BLOCKED.value: 1.0,
}


class EnvironmentEstimator:
    """Per-frame condition scores -> EMA -> primary condition + visibility."""

    def __init__(self, config: Optional[EnvironmentConfig] = None, model: Optional[ConditionModel] = None) -> None:
        self.config = config or EnvironmentConfig()
        self.model = model or HeuristicConditionModel(self.config)
        self._ema: dict[str, float] = {}

    def reset(self) -> None:
        self._ema.clear()

    def update(self, img: np.ndarray, stamp: float = 0.0) -> EnvironmentEstimate:
        raw = self.model.predict(img)
        a = self.config.ema_alpha
        for k, v in raw.items():
            self._ema[k] = v if k not in self._ema else (1 - a) * self._ema[k] + a * v
        scores = dict(self._ema)
        key = max(scores, key=scores.get) if scores else None
        cond = Condition(key) if key and scores[key] >= self.config.condition_threshold else Condition.NOMINAL
        vis = 1.0
        for k, v in scores.items():
            vis *= 1.0 - VISIBILITY_PENALTY.get(k, 0.3) * v
        stats = getattr(self.model, "last_stats", None)
        return EnvironmentEstimate(stamp, cond, scores, float(np.clip(vis, 0, 1)), stats)
