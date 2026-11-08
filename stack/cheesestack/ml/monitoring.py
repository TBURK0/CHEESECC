"""Online performance monitoring of deployed models.

``OnlineMonitor`` keeps a rolling window of live predictions (class, confidence,
optional delayed ground truth / proxy label, latency) and computes rolling
accuracy, mean confidence, low-confidence rate, class mix, latency percentiles
and a confidence histogram. ``check()`` turns threshold breaches into
``core.Fault`` alerts for the safety layer. Pure ``record()`` / ``check()``
methods; ``MonitorNode`` wraps it as a ``core.Node`` reading a bus topic.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cheesestack.core import Fault, MessageBus, Node, Severity, Clock


@dataclass
class MonitorConfig:
    window: int = 500
    min_samples: int = 50  # no alerts until the window has this many predictions
    min_labeled: int = 20
    accuracy_alert: float = 0.8  # rolling accuracy below -> DEGRADED
    mean_confidence_alert: float = 0.5
    low_confidence: float = 0.4  # a prediction below this counts as "low confidence"
    low_confidence_rate_alert: float = 0.3
    latency_p99_alert_ms: float = 50.0
    histogram_bins: int = 10
    source: str = "ml.monitor"


@dataclass
class PredictionEvent:
    stamp: float
    cls: str
    confidence: float
    label: Optional[str] = None  # ground truth or proxy (e.g. tracker-confirmed class), if known
    latency_ms: Optional[float] = None


@dataclass
class MonitorSnapshot:
    count: int
    labeled: int
    accuracy: Optional[float]
    mean_confidence: float
    low_confidence_rate: float
    class_distribution: dict = field(default_factory=dict)
    confidence_histogram: np.ndarray = field(default_factory=lambda: np.zeros(0))
    latency_p50_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    ece: Optional[float] = None  # expected calibration error on labelled events


class OnlineMonitor:
    """# TODO(ml): per-class/per-condition (night, rain) slices, label-delay joins
    by sample id, persistence of windows across restarts, auto-trigger data collection."""

    def __init__(self, config: Optional[MonitorConfig] = None) -> None:
        self.config = config or MonitorConfig()
        self.events: deque[PredictionEvent] = deque(maxlen=self.config.window)

    def record(self, event: PredictionEvent) -> None:
        self.events.append(event)

    def snapshot(self) -> MonitorSnapshot:
        cfg = self.config
        ev = list(self.events)
        if not ev:
            return MonitorSnapshot(0, 0, None, 0.0, 0.0, {}, np.zeros(cfg.histogram_bins, int))
        conf = np.array([e.confidence for e in ev], float)
        hist, _ = np.histogram(conf, bins=cfg.histogram_bins, range=(0.0, 1.0))
        lab = [e for e in ev if e.label is not None]
        acc = ece = None
        if lab:
            correct = np.array([e.cls == e.label for e in lab], float)
            lc = np.array([e.confidence for e in lab])
            acc = float(correct.mean())
            idx = np.minimum((lc * cfg.histogram_bins).astype(int), cfg.histogram_bins - 1)
            ece = 0.0
            for b in range(cfg.histogram_bins):
                m = idx == b
                if m.any():
                    ece += m.mean() * abs(correct[m].mean() - lc[m].mean())
            ece = float(ece)
        classes: dict[str, float] = {}
        for e in ev:
            classes[e.cls] = classes.get(e.cls, 0) + 1
        classes = {k: v / len(ev) for k, v in classes.items()}
        lats = np.array([e.latency_ms for e in ev if e.latency_ms is not None], float)
        return MonitorSnapshot(
            len(ev), len(lab), acc, float(conf.mean()), float(np.mean(conf < cfg.low_confidence)), classes, hist,
            float(np.percentile(lats, 50)) if lats.size else None,
            float(np.percentile(lats, 99)) if lats.size else None, ece,
        )

    def check(self, stamp: float) -> list[Fault]:
        cfg = self.config
        s = self.snapshot()
        faults: list[Fault] = []
        if s.count < cfg.min_samples:
            return faults

        def f(code, sev, msg, **data):
            faults.append(Fault(code, sev, cfg.source, stamp, msg, data))

        if s.accuracy is not None and s.labeled >= cfg.min_labeled and s.accuracy < cfg.accuracy_alert:
            f("ML_ACCURACY_LOW", Severity.DEGRADED, f"rolling accuracy {s.accuracy:.3f}", accuracy=s.accuracy)
        if s.mean_confidence < cfg.mean_confidence_alert:
            f("ML_CONFIDENCE_LOW", Severity.WARNING, f"mean confidence {s.mean_confidence:.3f}",
              mean_confidence=s.mean_confidence)
        if s.low_confidence_rate > cfg.low_confidence_rate_alert:
            f("ML_LOW_CONF_RATE", Severity.WARNING, f"low-confidence rate {s.low_confidence_rate:.3f}",
              rate=s.low_confidence_rate)
        if s.latency_p99_ms is not None and s.latency_p99_ms > cfg.latency_p99_alert_ms:
            f("ML_LATENCY_HIGH", Severity.WARNING, f"p99 latency {s.latency_p99_ms:.1f} ms",
              p99_ms=s.latency_p99_ms)
        return faults


class MonitorNode(Node):
    """Subscribes to ``input_topic`` (PredictionEvent messages) and publishes faults
    to ``fault_topic`` on each step."""

    def __init__(self, monitor: OnlineMonitor, bus: Optional[MessageBus] = None, clock: Optional[Clock] = None,
                 input_topic: str = "ml/predictions", fault_topic: str = "safety/faults") -> None:
        super().__init__("ml_monitor", bus, clock)
        self.monitor = monitor
        self.fault_topic = fault_topic
        self.bus.subscribe(input_topic, self._on_event)

    def _on_event(self, topic: str, msg: PredictionEvent) -> None:
        self.monitor.record(msg)

    def step(self, t: float) -> None:
        for fault in self.monitor.check(t):
            self.bus.publish(self.fault_topic, fault)
