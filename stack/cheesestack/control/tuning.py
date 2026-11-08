"""Controller tuning infrastructure.

- `GainSchedule`: named gains tabulated vs speed, linear interpolation with
  end clamping (np.interp).
- `step_response_metrics`: rise time (10-90 %), overshoot %, peak time,
  settling time (+-band), steady-state error, from sampled (t, y).
- `simulate_first_order_step`: tiny helper to sanity-check metrics/tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


class GainSchedule:
    def __init__(self, speeds_mps: Sequence[float], gains: Mapping[str, Sequence[float]]) -> None:
        sp = np.asarray(speeds_mps, dtype=float)
        if sp.ndim != 1 or len(sp) == 0 or np.any(np.diff(sp) <= 0):
            raise ValueError("speeds must be strictly increasing")
        self.speeds = sp
        self.gains = {}
        for k, vals in gains.items():
            arr = np.asarray(vals, dtype=float)
            if arr.shape != sp.shape:
                raise ValueError(f"gain {k!r} length {arr.shape} != speeds {sp.shape}")
            self.gains[k] = arr

    def __call__(self, v: float) -> dict[str, float]:
        return {k: float(np.interp(v, self.speeds, arr)) for k, arr in self.gains.items()}

    def get(self, name: str, v: float) -> float:
        return float(np.interp(v, self.speeds, self.gains[name]))


@dataclass
class StepMetrics:
    rise_time: float
    overshoot_pct: float
    peak_time: float
    settling_time: float
    steady_state_error: float
    final_value: float


def step_response_metrics(t, y, setpoint: float, y0: float = 0.0, settle_band: float = 0.02) -> StepMetrics:
    """Classical step metrics. NaN is returned for metrics that are never reached."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    span = setpoint - y0
    if span == 0:
        raise ValueError("setpoint must differ from y0")
    yn = (y - y0) / span  # normalised 0 -> 1
    def first_cross(level):
        idx = np.nonzero(yn >= level)[0]
        if len(idx) == 0:
            return float("nan")
        i = idx[0]
        if i == 0:
            return float(t[0])
        # linear interpolation
        f = (level - yn[i - 1]) / (yn[i] - yn[i - 1])
        return float(t[i - 1] + f * (t[i] - t[i - 1]))
    rise = first_cross(0.9) - first_cross(0.1)
    ip = int(np.argmax(yn))
    overshoot = max(0.0, (yn[ip] - 1.0) * 100.0)
    outside = np.nonzero(np.abs(yn - 1.0) > settle_band)[0]
    if len(outside) == 0:
        settling = float(t[0])
    elif outside[-1] == len(t) - 1:
        settling = float("nan")
    else:
        settling = float(t[outside[-1] + 1])
    final = float(y[-1])
    return StepMetrics(rise, overshoot, float(t[ip]), settling, setpoint - final, final)


def simulate_second_order_step(wn: float, zeta: float, t_end: float = 10.0, dt: float = 0.001):
    """Unit step response of wn^2/(s^2 + 2 zeta wn s + wn^2) by semi-implicit Euler."""
    n = int(t_end / dt) + 1
    t = np.arange(n) * dt
    y = np.zeros(n)
    yd = 0.0
    for i in range(1, n):
        ydd = wn * wn * (1.0 - y[i - 1]) - 2 * zeta * wn * yd
        yd += ydd * dt
        y[i] = y[i - 1] + yd * dt
    return t, y

# TODO(control): automated tuning (relay autotune / Ziegler-Nichols, or sweep over a sim
# closed loop and pick gains minimising IAE subject to overshoot), schedule export/import.
