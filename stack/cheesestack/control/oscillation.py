"""Oscillation detection on a scalar signal (e.g. steering command, lateral error).

Sliding window of the last N samples. Flags oscillation when
    sign-change rate of (x - mean) >= min_crossing_rate_hz * 2 (two crossings per cycle)
    and half peak-to-peak amplitude >= min_amplitude.
Also reports the dominant FFT frequency (Hann-windowed, mean removed) so callers
can tell limit cycles (~1-3 Hz steering hunting) from slow weaving.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class OscillationConfig:
    sample_rate_hz: float = 20.0
    window_s: float = 4.0
    min_amplitude: float = 1.0  # signal units (e.g. handwheel deg)
    min_freq_hz: float = 0.3  # crossing-rate frequency to count as oscillation
    max_freq_hz: float = 8.0
    deadband: float = 0.0  # ignore crossings smaller than this around the mean


@dataclass
class OscillationReport:
    oscillating: bool
    crossing_freq_hz: float
    dominant_freq_hz: float
    amplitude: float
    n_samples: int


def dominant_frequency(x: np.ndarray, fs: float) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 4:
        return 0.0
    x = (x - x.mean()) * np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    spec[0] = 0.0
    return float(freqs[int(np.argmax(spec))]) if spec.max() > 0 else 0.0


class OscillationDetector:
    def __init__(self, config: OscillationConfig | None = None) -> None:
        self.config = config or OscillationConfig()
        n = max(4, int(round(self.config.window_s * self.config.sample_rate_hz)))
        self.buf: deque[float] = deque(maxlen=n)

    def reset(self) -> None:
        self.buf.clear()

    def update(self, x: float) -> OscillationReport:
        self.buf.append(float(x))
        return self.analyze()

    def analyze(self) -> OscillationReport:
        c = self.config
        x = np.asarray(self.buf, dtype=float)
        n = len(x)
        if n < self.buf.maxlen // 2:
            return OscillationReport(False, 0.0, 0.0, 0.0, n)
        d = x - x.mean()
        # hysteresis sign: keep previous sign within deadband
        s = np.sign(d)
        if c.deadband > 0:
            s[np.abs(d) < c.deadband] = 0
        nz = s[s != 0]
        crossings = int(np.sum(nz[1:] != nz[:-1])) if len(nz) > 1 else 0
        duration = n / c.sample_rate_hz
        f_cross = crossings / (2.0 * duration)
        amp = 0.5 * float(x.max() - x.min())
        f_dom = dominant_frequency(x, c.sample_rate_hz)
        osc = amp >= c.min_amplitude and c.min_freq_hz <= f_cross <= c.max_freq_hz
        return OscillationReport(osc, f_cross, f_dom, amp, n)

# TODO(control): per-band energy ratio (reject broadband noise), coupling to gain
# scheduler to back off gains when sustained oscillation is detected.
