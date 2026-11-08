import json
import urllib.request

import pytest

from cheesestack.core import Fault, Node, NodeStatus, Severity, SimClock
from cheesestack.infra import (ExperimentConfig, ExperimentTracker, HealthAggregator, MetricsRegistry,
                               ModelRegistry, RegistryError, StatusServer, compare_runs)


def test_registry_versions_promotion_rollback(tmp_path):
    clock = SimClock(100.0)
    reg = ModelRegistry(tmp_path / "reg", clock=clock)
    src = tmp_path / "lane.onnx"
    src.write_bytes(b"weights-v1")
    v1 = reg.register("lane", src, metadata={"arch": "tiny"}, metrics={"iou": 0.7})
    v2 = reg.register("lane", b"weights-v2")
    v3 = reg.register("lane", b"weights-v3")
    assert [v.version for v in reg.versions("lane")] == [1, 2, 3]
    assert v1.artifact.endswith(".onnx") and len(v1.sha256) == 64
    assert reg.get("lane", 1).metadata == {"arch": "tiny"}
    with pytest.raises(RegistryError):
        reg.promote("lane", 1, "prod")  # never staged
    for v in (v1, v2, v3):
        reg.promote("lane", v.version, "staging")
        reg.promote("lane", v.version, "prod")
    assert reg.stage_version("lane", "prod") == 3
    assert reg.load_path("lane", "prod").read_bytes() == b"weights-v3"
    assert reg.rollback("lane", "prod") == 2
    assert reg.rollback("lane", "prod") == 1
    assert reg.rollback("lane", "prod") is None
    assert reg.stage_version("lane", "prod") is None
    # tamper detection
    (tmp_path / "reg" / v2.artifact).write_bytes(b"evil")
    assert not reg.verify("lane", 2)
    with pytest.raises(RegistryError):
        reg.load_path("lane", 2)
    with pytest.raises(RegistryError):
        reg.promote("lane", 2, "dev")
    assert reg.models() == ["lane"]


class Dummy(Node):
    def step(self, t):
        pass


def test_health_aggregator_levels():
    clock = SimClock()
    agg = HealthAggregator(clock=clock)
    n = Dummy("planner", clock=clock)
    n.tick()
    agg.add_node(n)
    agg.add_check("disk", lambda: (True, {"free_gb": 10}))
    agg.metrics = MetricsRegistry()
    agg.metrics.gauge("x").set(1)
    doc = agg.snapshot()
    assert doc["level"] == "ok" and doc["components"]["node:planner"]["heartbeat_age_s"] == 0.0
    assert doc["metrics"]["cheese_x"] == 1.0
    clock.advance(2.0)  # stale heartbeat
    assert agg.snapshot()["level"] == "degraded"
    n.tick()
    agg.report_fault(Fault("E1", Severity.CRITICAL, "gps", clock.now(), "no fix"))
    doc = json.loads(agg.to_json())
    assert doc["level"] == "failed" and doc["faults"][0]["fault"]["severity"] == 3
    clock.advance(0.5)
    n.tick()
    clock.advance(5.0)
    n.tick()
    assert agg.snapshot()["level"] == "ok"  # fault expired
    agg.add_check("broken", lambda: 1 / 0)
    assert agg.snapshot()["components"]["check:broken"]["level"] == "failed"
    n.status = NodeStatus.FAILED
    assert agg.snapshot()["components"]["node:planner"]["level"] == "failed"


def test_status_server_opt_in():
    agg = HealthAggregator()
    srv = StatusServer(agg, metrics_text=lambda: "cheese_up 1\n")
    assert srv._httpd is None  # nothing bound until start()
    with srv:
        base = f"http://127.0.0.1:{srv.port}"
        body = json.loads(urllib.request.urlopen(base + "/status", timeout=5).read())
        assert body["level"] == "ok"
        assert urllib.request.urlopen(base + "/metrics", timeout=5).read() == b"cheese_up 1\n"
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/nope", timeout=5)


def test_experiment_tracking(tmp_path):
    tr = ExperimentTracker(ExperimentConfig(root=str(tmp_path)), clock=SimClock())
    with tr.start_run("lka", "baseline", params={"kp": 0.5, "horizon": 50}) as r1:
        for e in [0.3, 0.2, 0.15]:
            r1.log_metric("lat_err", e)
        r1.log_artifact(b"{}", "cfg.json")
    with tr.start_run("lka", "tuned kp", params={"kp": 0.8, "horizon": 50}) as r2:
        r2.log_metrics({"lat_err": 0.12, "jerk": 1.0}, step=0)
    with pytest.raises(RuntimeError):
        with tr.start_run("lka", "crash") as r3:
            raise RuntimeError("x")
    runs = tr.list_runs("lka")
    assert [r.run_id for r in runs] == ["0000-baseline", "0001-tuned_kp", "0002-crash"]
    assert runs[2].info.status == "failed" and runs[0].info.status == "finished"
    assert runs[0].artifacts() == ["cfg.json"]
    assert runs[0].metric_history("lat_err") == [(0, 0.3), (1, 0.2), (2, 0.15)]
    cmp = compare_runs(runs[:2])
    assert cmp["differing_params"] == ["kp"]
    assert cmp["metrics"]["lat_err"] == [0.15, 0.12] and cmp["metrics"]["jerk"] == [None, 1.0]
    assert compare_runs(runs[:2], ["lat_err"], agg="max")["metrics"]["lat_err"] == [0.3, 0.12]
    assert tr.best_run("lka", "lat_err").run_id == "0001-tuned_kp"
    assert tr.experiments() == ["lka"]
