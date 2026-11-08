import math

import numpy as np

from cheesestack.control import (
    ActuatorMonitor, ActuatorMonitorConfig, OscillationConfig, OscillationDetector, dominant_frequency, estimate_lag,
)


def test_dominant_frequency():
    fs = 50.0
    t = np.arange(200) / fs
    assert abs(dominant_frequency(np.sin(2 * np.pi * 2.0 * t), fs) - 2.0) < 0.3


def test_oscillation_detected_for_hunting_not_for_steady_turn():
    cfg = OscillationConfig(sample_rate_hz=20.0, window_s=4.0, min_amplitude=2.0)
    det = OscillationDetector(cfg)
    for i in range(100):
        rep = det.update(5.0 * math.sin(2 * math.pi * 1.5 * i / 20.0))
    assert rep.oscillating and abs(rep.crossing_freq_hz - 1.5) < 0.3 and abs(rep.dominant_freq_hz - 1.5) < 0.3
    det.reset()
    for i in range(100):
        rep = det.update(30.0 + 0.01 * i)  # slow ramp, no oscillation
    assert not rep.oscillating
    det.reset()
    rng = np.random.default_rng(0)
    for _ in range(100):
        rep = det.update(0.1 * rng.standard_normal())  # small noise below amplitude
    assert not rep.oscillating


def test_lag_estimate():
    fs = 20.0
    t = np.arange(120) / fs
    cmd = np.sin(2 * np.pi * 0.4 * t)
    meas = np.sin(2 * np.pi * 0.4 * (t - 0.25))
    assert abs(estimate_lag(cmd, meas, fs, 0.6) - 0.25) <= 1 / fs


def test_actuator_monitor_healthy_and_stuck():
    cfg = ActuatorMonitorConfig(sample_rate_hz=20.0, window_s=3.0)
    mon = ActuatorMonitor(cfg)
    meas = 0.0
    for i in range(80):
        t = i / 20.0
        cmd = 40 * math.sin(2 * math.pi * 0.3 * t)
        meas += (cmd - meas) * 0.05 / 0.1
        st = mon.update(t, cmd, meas)
    assert not st.stuck and not any(f.code == "ACTUATOR_STUCK" for f in st.faults)
    mon.reset()
    for i in range(80):
        t = i / 20.0
        st = mon.update(t, 40 * math.sin(2 * math.pi * 0.3 * t), 3.0)
    assert st.stuck and st.faults[0].code == "ACTUATOR_STUCK"
