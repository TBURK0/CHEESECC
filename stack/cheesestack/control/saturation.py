"""Saturation handling: magnitude clamps, rate limiters, and PID with anti-windup.

Anti-windup modes:
- "clamp" (conditional integration): integrator is frozen while the output is
  saturated and the error would push it further into saturation.
- "back_calculation": integrator is bled by (u_sat - u_unsat) / Tt (Astrom).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class RateLimiterConfig:
    max_rate: float = 1.0  # units/s, rising
    max_fall_rate: Optional[float] = None  # units/s, falling; defaults to max_rate
    lo: float = float("-inf")
    hi: float = float("inf")


class RateLimiter:
    """Magnitude + slew-rate limiter. Stateful: remembers the last output."""

    def __init__(self, config: RateLimiterConfig | None = None, initial: float = 0.0) -> None:
        self.config = config or RateLimiterConfig()
        self.value = clamp(initial, self.config.lo, self.config.hi)
        self.saturated = False

    def reset(self, value: float = 0.0) -> None:
        self.value = clamp(value, self.config.lo, self.config.hi)

    def update(self, target: float, dt: float) -> float:
        c = self.config
        fall = c.max_rate if c.max_fall_rate is None else c.max_fall_rate
        tgt = clamp(target, c.lo, c.hi)
        delta = clamp(tgt - self.value, -fall * dt, c.max_rate * dt)
        self.value += delta
        self.saturated = abs(self.value - target) > 1e-12
        return self.value


@dataclass
class PIDConfig:
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    out_min: float = float("-inf")
    out_max: float = float("inf")
    anti_windup: str = "clamp"  # "clamp" | "back_calculation" | "none"
    tracking_time: float = 0.5  # Tt for back-calculation, s
    d_filter_tau: float = 0.0  # first-order filter on derivative, s
    i_limit: float = float("inf")  # absolute bound on integrator contribution


class PID:
    """Discrete PID, derivative on measurement, with anti-windup.

    `update(error, dt, measurement=None, feedforward=0.0)`; if measurement is
    given the D term uses -d(measurement)/dt (no derivative kick).
    """

    def __init__(self, config: PIDConfig | None = None) -> None:
        self.config = config or PIDConfig()
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0  # stored as the integrator *contribution* (ki already applied)
        self._prev_err: Optional[float] = None
        self._prev_meas: Optional[float] = None
        self._d_state = 0.0
        self.saturated = False
        self.last_terms = (0.0, 0.0, 0.0)

    def update(self, error: float, dt: float, measurement: Optional[float] = None, feedforward: float = 0.0) -> float:
        c = self.config
        if dt <= 0:
            raise ValueError("dt must be > 0")
        p = c.kp * error
        # Derivative
        if measurement is not None:
            raw_d = 0.0 if self._prev_meas is None else -(measurement - self._prev_meas) / dt
            self._prev_meas = measurement
        else:
            raw_d = 0.0 if self._prev_err is None else (error - self._prev_err) / dt
        self._prev_err = error
        if c.d_filter_tau > 0:
            alpha = dt / (c.d_filter_tau + dt)
            self._d_state += alpha * (raw_d - self._d_state)
        else:
            self._d_state = raw_d
        d = c.kd * self._d_state

        candidate_i = clamp(self.integral + c.ki * error * dt, -c.i_limit, c.i_limit)
        u_unsat = feedforward + p + candidate_i + d
        u = clamp(u_unsat, c.out_min, c.out_max)
        self.saturated = u != u_unsat

        if c.anti_windup == "clamp":
            pushing_further = self.saturated and (u_unsat - u) * error > 0
            if not pushing_further:
                self.integral = candidate_i
        elif c.anti_windup == "back_calculation":
            self.integral = candidate_i + (u - u_unsat) * dt / max(c.tracking_time, 1e-6)
            self.integral = clamp(self.integral, -c.i_limit, c.i_limit)
        else:
            self.integral = candidate_i
        self.last_terms = (p, self.integral, d)
        return u
