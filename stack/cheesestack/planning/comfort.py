"""Comfort modelling: jerk / lateral-accel costs and an ISO 2631-1-style score.

ISO 2631-1 rates whole-body vibration by the frequency-weighted RMS acceleration
a_w, combining axes as a_v = sqrt((k_x a_wx)^2 + (k_y a_wy)^2 + (k_z a_wz)^2) with
k_x = k_y = 1.4 for horizontal axes of a seated occupant. We approximate:
  * no frequency weighting (W_d) — plan-level signals are low frequency anyway;
  * longitudinal = trajectory accel, lateral = v^2 * kappa, vertical ignored.
The resulting a_v is mapped to the standard's comfort bands and to a [0, 1]
score. # TODO(planning): apply W_d/W_k filters on resampled signals, add the
vertical axis from road roughness, motion-sickness dose value (W_f, 0.1-0.5 Hz).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cheesestack.core import Trajectory

# (upper bound m/s^2, label) — ISO 2631-1 Annex C (bands overlap in the standard;
# we use the lower edge of the next band as the split).
ISO_BANDS = (
    (0.315, "not uncomfortable"),
    (0.5, "a little uncomfortable"),
    (0.8, "fairly uncomfortable"),
    (1.25, "uncomfortable"),
    (2.0, "very uncomfortable"),
    (float("inf"), "extremely uncomfortable"),
)


@dataclass
class ComfortConfig:
    k_horizontal: float = 1.4
    score_scale: float = 2.0  # a_v at which score reaches 0
    w_jerk_lon: float = 1.0
    w_jerk_lat: float = 1.0
    w_lat_accel: float = 0.5
    w_lon_accel: float = 0.5


@dataclass
class ComfortReport:
    a_v: float  # combined weighted RMS accel, m/s^2
    rms_lon: float
    rms_lat: float
    peak_jerk_lon: float
    peak_jerk_lat: float
    band: str
    score: float  # 1 = perfectly smooth, 0 = extremely uncomfortable
    cost: float


def iso_band(a_v: float) -> str:
    for ub, label in ISO_BANDS:
        if a_v < ub:
            return label
    return ISO_BANDS[-1][1]


class ComfortModel:
    def __init__(self, config: ComfortConfig = ComfortConfig()) -> None:
        self.config = config

    def evaluate_signals(self, t, a_lon, a_lat) -> ComfortReport:
        c = self.config
        t, a_lon, a_lat = (np.asarray(z, float) for z in (t, a_lon, a_lat))
        rms_lon = float(np.sqrt(np.mean(a_lon**2))) if len(a_lon) else 0.0
        rms_lat = float(np.sqrt(np.mean(a_lat**2))) if len(a_lat) else 0.0
        a_v = float(np.hypot(c.k_horizontal * rms_lon, c.k_horizontal * rms_lat))
        if len(t) >= 2:
            dt = np.maximum(np.diff(t), 1e-9)
            j_lon, j_lat = np.diff(a_lon) / dt, np.diff(a_lat) / dt
            pj_lon, pj_lat = float(np.abs(j_lon).max()), float(np.abs(j_lat).max())
            T = float(t[-1] - t[0]) or 1.0
            int_j = (c.w_jerk_lon * float(np.sum(j_lon**2 * dt)) + c.w_jerk_lat * float(np.sum(j_lat**2 * dt))) / T
        else:
            pj_lon = pj_lat = int_j = 0.0
        cost = int_j + c.w_lon_accel * rms_lon**2 + c.w_lat_accel * rms_lat**2
        score = float(np.clip(1.0 - a_v / c.score_scale, 0.0, 1.0))
        return ComfortReport(a_v, rms_lon, rms_lat, pj_lon, pj_lat, iso_band(a_v), score, cost)

    def evaluate_array(self, arr: np.ndarray) -> ComfortReport:
        t, v, a, k = arr[:, 0], arr[:, 4], arr[:, 5], arr[:, 6]
        return self.evaluate_signals(t, a, v**2 * k)

    def evaluate(self, traj: Trajectory) -> ComfortReport:
        return self.evaluate_array(traj.as_array())
