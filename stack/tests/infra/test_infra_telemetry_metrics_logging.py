import io
import json
import logging

import numpy as np

from cheesestack.core import MessageBus, SimClock
from cheesestack.infra import (ChannelSpec, LoggingConfig, MetricsRegistry, TelemetryConfig,
                               TelemetrySink, UplinkPacket, get_logger, setup_logging)


def test_telemetry_decimation_and_rate():
    clock = SimClock()
    sink = TelemetrySink(clock=clock)
    sink.add_channel("speed", ChannelSpec(decimate=5))
    sink.add_channel("pose", ChannelSpec(max_rate_hz=2.0))
    for i in range(20):
        sink.record("speed", float(i))
        sink.record("pose", [i, i])
        clock.advance(0.1)
    # speed: 20/5 = 4 kept; pose: 2 s at 2 Hz -> t=0,0.5,1.0,1.5 = 4 kept
    assert sink.accepted == 8 and sink.decimated == 32


def test_telemetry_packets_bounded_and_roundtrip():
    sink = TelemetrySink(TelemetryConfig(max_packet_bytes=200))
    sink.add_channel("fault", ChannelSpec(priority=10))
    for i in range(50):
        sink.record("speed", np.float64(i * 0.5), t=i * 0.1)
    sink.record("fault", "E42", t=4.0)
    pkts = sink.flush()
    assert len(pkts) > 1 and sink.pending() == 0
    assert all(len(p.payload) <= 200 for p in pkts)
    first = UplinkPacket.decode(pkts[0].payload)
    assert first[0].channel == "fault"  # priority first
    decoded = [s for p in pkts for s in UplinkPacket.decode(p.payload)]
    assert len(decoded) == 51
    assert [p.seq for p in pkts] == list(range(1, len(pkts) + 1))


def test_telemetry_compressed_and_bus_and_buffer_limit():
    sink = TelemetrySink(TelemetryConfig(compress=True, max_buffer=3, max_packet_bytes=10_000))
    bus = MessageBus()
    sink.attach(bus, {"/v": lambda m: m["v"]})
    for i in range(5):
        bus.publish("/v", {"v": i})
    bus.publish("/ignored", 1)
    assert sink.pending() == 3 and sink.dropped == 2
    (p,) = sink.flush()
    assert [s.value for s in UplinkPacket.decode(p.payload, compressed=True)] == [2, 3, 4]


def test_metrics_and_prometheus():
    t = [0.0]
    reg = MetricsRegistry(time_fn=lambda: t[0])
    reg.counter("msgs_total", "messages", topic="/a").inc()
    reg.counter("msgs_total", topic="/a").inc(2)
    reg.counter("msgs_total", topic="/b").inc()
    reg.gauge("speed_mps").set(3.5)
    h = reg.histogram("lat_seconds", buckets=(0.1, 1.0))
    for v in np.linspace(0, 2, 101):
        h.observe(v)
    assert abs(h.percentile(50) - 1.0) < 1e-9
    assert abs(h.percentile(90) - 1.8) < 1e-9
    with reg.timer("lat_seconds") as tm:
        t[0] = 0.25
    assert tm.elapsed == 0.25 and h.count == 102
    txt = reg.render_prometheus()
    assert '# TYPE cheese_msgs_total counter' in txt
    assert 'cheese_msgs_total{topic="/a"} 3' in txt
    assert 'cheese_speed_mps 3.5' in txt
    assert 'cheese_lat_seconds_bucket{le="+Inf"} 102' in txt
    assert 'cheese_lat_seconds_count 102' in txt
    snap = reg.snapshot()
    assert snap["cheese_speed_mps"] == 3.5


def test_metrics_reservoir_bounded():
    from cheesestack.infra import MetricsConfig
    reg = MetricsRegistry(MetricsConfig(reservoir_size=100))
    h = reg.histogram("x")
    for v in range(10_000):
        h.observe(v)
    assert len(h._res) == 100 and h.count == 10_000
    assert 2000 < h.percentile(50) < 8000


def test_json_logging(tmp_path):
    buf = io.StringIO()
    clock = SimClock(12.5)
    cfg = LoggingConfig(module_levels={"cheesestack.noisy": "ERROR"},
                        file_path=str(tmp_path / "log.jsonl"), max_bytes=500, backup_count=2)
    setup_logging(cfg, clock=clock, stream=buf)
    setup_logging(cfg, clock=clock, stream=buf)  # idempotent
    log = get_logger("planning")
    log.info("hello %s", "world", extra={"speed": 3.0})
    get_logger("noisy").warning("suppressed")
    lines = buf.getvalue().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["msg"] == "hello world" and rec["speed"] == 3.0 and rec["t"] == 12.5
    assert rec["logger"] == "cheesestack.planning" and rec["level"] == "INFO"
    for i in range(50):
        log.info("fill %d", i)
    assert (tmp_path / "log.jsonl.1").exists()
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        log.exception("failed")
    assert "RuntimeError" in json.loads(buf.getvalue().strip().splitlines()[-1])["exc"]
    setup_logging(LoggingConfig(stream=False))  # detach file handler
    logging.getLogger("cheesestack.noisy").setLevel(logging.NOTSET)
