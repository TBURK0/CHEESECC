import numpy as np

from cheesestack.core import MessageBus, SimClock
from cheesestack.ml.monitoring import MonitorConfig, MonitorNode, OnlineMonitor, PredictionEvent


def feed(mon, n, conf, correct_rate, rng, latency=5.0):
    for i in range(n):
        ok = rng.random() < correct_rate
        mon.record(PredictionEvent(float(i), "car", conf, "car" if ok else "truck", latency))


def test_healthy_window_no_alerts():
    rng = np.random.default_rng(0)
    mon = OnlineMonitor(MonitorConfig(window=200))
    feed(mon, 300, 0.9, 0.95, rng)
    s = mon.snapshot()
    assert s.count == 200 and s.labeled == 200
    assert s.confidence_histogram.sum() == 200 and s.confidence_histogram[-1] == 200
    assert s.ece < 0.1 and s.latency_p99_ms == 5.0
    assert mon.check(1.0) == []


def test_degradation_alerts():
    rng = np.random.default_rng(1)
    mon = OnlineMonitor(MonitorConfig(window=100))
    feed(mon, 100, 0.35, 0.5, rng, latency=80.0)
    codes = {f.code for f in mon.check(2.0)}
    assert codes == {"ML_ACCURACY_LOW", "ML_CONFIDENCE_LOW", "ML_LOW_CONF_RATE", "ML_LATENCY_HIGH"}


def test_min_samples_gate_and_empty():
    mon = OnlineMonitor()
    assert mon.snapshot().count == 0
    mon.record(PredictionEvent(0.0, "car", 0.1))
    assert mon.check(0.0) == []


def test_monitor_node_publishes_faults():
    bus, clock = MessageBus(), SimClock()
    node = MonitorNode(OnlineMonitor(MonitorConfig(min_samples=5)), bus, clock)
    got = []
    bus.subscribe("safety/faults", lambda t, m: got.append(m))
    for i in range(10):
        bus.publish("ml/predictions", PredictionEvent(float(i), "car", 0.1))
    node.tick()
    assert {f.code for f in got} >= {"ML_CONFIDENCE_LOW"}
