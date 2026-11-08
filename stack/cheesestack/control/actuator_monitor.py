"""Actuator monitoring: commanded vs measured.

Over a sliding window it computes
- RMS and max tracking error,
- lag estimate: the shift (s) maximising the cross-correlation of measured
  against delayed commanded signals (lag search 0..max_lag_s),
- stuck detection: measured range below `stuck_range` while commanded range
  exceeds `stuck_cmd_range` (actuator not following a moving command), or
  |error| > `stuck_error` continuously for `stuck_time_s`.
Emits core.Fault objects (code ACTUATOR_STUCK / ACTUATOR_TRACKING / ACTUATOR_LAG).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from cheesestack.core import Fault, Severity


@dataclass
class ActuatorMonitorConfig:
    name: str = "steer"
    sample_rate_hz: float = 20.0
    window_s: float = 3.0
    max_lag_s: float = 0.6
    rms_error_warn: float = 10.0  # units (deg handwheel)
    max_lag_warn_s: float = 0.4
    stuck_range: float = 0.5
    stuck_cmd_range: float = 10.0
    stuck_error: float = 30.0
    stuck_time_s: float = 0.5


@dataclass
class ActuatorStatus:
    rms_error: float = 0.0
    max_error: float = 0.0
    lag_s: float = 0.0
    stuck: bool = False
    faults: list[Fault] = field(default_factory=list)


def estimate_lag(cmd: np.ndarray, meas: np.ndarray, fs: float, max_lag_s: float) -> float:
    """Lag (s) of meas behind cmd via normalised cross-correlation over integer shifts."""
    cmd = np.asarray(cmd, float) - np.mean(cmd)
    meas = np.asarray(meas, float) - np.mean(meas)
    n = len(cmd)
    max_k = min(int(round(max_lag_s * fs)), n - 4)
    if max_k < 0 or np.std(cmd) < 1e-9 or np.std(meas) < 1e-9:
        return 0.0
    best_k, best_c = 0, -np.inf
    for k in range(max_k + 1):
        a, b = cmd[: n - k], meas[k:]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        c = float(a @ b / denom) if denom > 0 else -np.inf
        if c > best_c:
            best_k, best_c = k, c
    return best_k / fs


class ActuatorMonitor:
    def __init__(self, config: ActuatorMonitorConfig | None = None) -> None:
        self.config = config or ActuatorMonitorConfig()
        n = max(8, int(round(self.config.window_s * self.config.sample_rate_hz)))
        self.cmd: deque[float] = deque(maxlen=n)
        self.meas: deque[float] = deque(maxlen=n)
        self._big_err_since: float | None = None

    def reset(self) -> None:
        self.cmd.clear()
        self.meas.clear()
        self._big_err_since = None

    def update(self, t: float, commanded: float, measured: float) -> ActuatorStatus:
        c = self.config
        self.cmd.append(float(commanded))
        self.meas.append(float(measured))
        err_now = abs(commanded - measured)
        if err_now > c.stuck_error:
            if self._big_err_since is None:
                self._big_err_since = t
        else:
            self._big_err_since = None
        cmd = np.asarray(self.cmd)
        meas = np.asarray(self.meas)
        err = cmd - meas
        st = ActuatorStatus(float(np.sqrt(np.mean(err ** 2))), float(np.max(np.abs(err))))
        full = len(cmd) == self.cmd.maxlen
        if full:
            st.lag_s = estimate_lag(cmd, meas, c.sample_rate_hz, c.max_lag_s)
        frozen = full and np.ptp(meas) < c.stuck_range and np.ptp(cmd) > c.stuck_cmd_range
        persistent = self._big_err_since is not None and t - self._big_err_since >= c.stuck_time_s
        st.stuck = bool(frozen or persistent)
        src = f"control.actuator_monitor.{c.name}"
        if st.stuck:
            st.faults.append(Fault("ACTUATOR_STUCK", Severity.CRITICAL, src, t,
                                   f"{c.name} not following command", {"rms": st.rms_error}))
        elif full and st.rms_error > c.rms_error_warn:
            st.faults.append(Fault("ACTUATOR_TRACKING", Severity.WARNING, src, t,
                                   f"{c.name} rms error {st.rms_error:.2f}", {"rms": st.rms_error}))
        if full and st.lag_s > c.max_lag_warn_s:
            st.faults.append(Fault("ACTUATOR_LAG", Severity.WARNING, src, t,
                                   f"{c.name} lag {st.lag_s:.2f}s", {"lag_s": st.lag_s}))
        return st

# TODO(control): first-order + delay system identification (least squares) feeding
# LagCompensator; separate thresholds by speed; saturation-aware error (cmd beyond limits).
