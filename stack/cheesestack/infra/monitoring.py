"""Remote monitoring: health snapshot aggregation and an opt-in status endpoint.

``HealthAggregator`` collects health from registered sources and produces one
JSON-serializable status document:

- ``Node`` objects: status (``NodeStatus``), heartbeat age vs. a staleness
  threshold, step count;
- ``Fault`` objects reported via ``report_fault`` (latest per code, expiring
  after ``fault_ttl_s``);
- arbitrary ``check`` callables returning ``(ok: bool, detail)``;
- an optional ``MetricsRegistry``-like object with ``snapshot()``.

The overall level is the worst of all components: ``ok`` < ``degraded`` <
``failed``. ``StatusServer`` is a tiny ``http.server`` that serves
``/status`` (JSON) and ``/metrics`` (Prometheus text). It binds only when
``start()`` is called - nothing listens on import.

TODO(infra): auth on the endpoint, push to fleet backend over comms, history
ring buffer of status documents, alert rules / dedup.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional

from cheesestack.core import Clock, Fault, Node, NodeStatus, SimClock

LEVELS = ("ok", "degraded", "failed")


def _worst(a: str, b: str) -> str:
    return a if LEVELS.index(a) >= LEVELS.index(b) else b


@dataclass
class MonitoringConfig:
    stale_heartbeat_s: float = 1.0
    fault_ttl_s: float = 5.0
    vehicle_id: str = "cheese-0"


def _severity_level(sev: Any) -> str:
    name = getattr(sev, "name", str(sev)).upper()
    if name in ("FATAL", "CRITICAL", "ERROR", "EMERGENCY"):
        return "failed"
    if name in ("WARN", "WARNING", "DEGRADED"):
        return "degraded"
    return "ok"


class HealthAggregator:
    def __init__(self, config: Optional[MonitoringConfig] = None, clock: Optional[Clock] = None) -> None:
        self.config = config or MonitoringConfig()
        self.clock = clock or SimClock()
        self._nodes: dict[str, Node] = {}
        self._checks: dict[str, Callable[[], tuple[bool, Any]]] = {}
        self._faults: dict[str, tuple[float, Any]] = {}
        self.metrics: Any = None
        self._lock = threading.Lock()

    def add_node(self, node: Node) -> None:
        self._nodes[node.name] = node

    def add_check(self, name: str, fn: Callable[[], tuple[bool, Any]]) -> None:
        self._checks[name] = fn

    def report_fault(self, fault: Fault, t: Optional[float] = None) -> None:
        code = f"{getattr(fault, 'source', '')}:{getattr(fault, 'code', 'fault')}"
        with self._lock:
            self._faults[str(code)] = (self.clock.now() if t is None else t, fault)

    def snapshot(self) -> dict:
        now = self.clock.now()
        overall = "ok"
        comps: dict[str, dict] = {}
        for name, node in self._nodes.items():
            age = now - node.heartbeat
            if node.status == NodeStatus.FAILED:
                lvl = "failed"
            elif node.status == NodeStatus.DEGRADED or age > self.config.stale_heartbeat_s:
                lvl = "degraded"
            else:
                lvl = "ok"
            comps[f"node:{name}"] = {"level": lvl, "status": node.status.value,
                                     "heartbeat_age_s": None if age == float("inf") else round(age, 6),
                                     "steps": node.step_count}
            overall = _worst(overall, lvl)
        for name, fn in self._checks.items():
            try:
                ok, detail = fn()
                lvl = "ok" if ok else "degraded"
            except Exception as e:  # noqa: BLE001 - a broken check is itself a failure
                lvl, detail = "failed", repr(e)
            comps[f"check:{name}"] = {"level": lvl, "detail": _safe(detail)}
            overall = _worst(overall, lvl)
        faults = []
        with self._lock:
            for code, (t, f) in list(self._faults.items()):
                if now - t > self.config.fault_ttl_s:
                    del self._faults[code]
                    continue
                lvl = _severity_level(getattr(f, "severity", None))
                faults.append({"code": code, "t": t, "level": lvl, "fault": _safe(f)})
                overall = _worst(overall, lvl)
        doc = {"vehicle_id": self.config.vehicle_id, "t": now, "level": overall,
               "components": comps, "faults": faults}
        if self.metrics is not None:
            doc["metrics"] = _safe(self.metrics.snapshot())
        return doc

    def to_json(self) -> str:
        return json.dumps(self.snapshot(), sort_keys=True)


def _safe(x: Any) -> Any:
    """Make a value JSON-safe (dataclasses, enums, numpy, NaN -> None)."""
    import dataclasses
    import math
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {k: _safe(v) for k, v in vars(x).items()}
    if isinstance(x, dict):
        return {str(k): _safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_safe(v) for v in x]
    if isinstance(x, float):
        return None if math.isnan(x) or math.isinf(x) else x
    if hasattr(x, "tolist"):
        return _safe(x.tolist())
    if hasattr(x, "value") and hasattr(x, "name"):
        return x.value if isinstance(x.value, (str, int)) else x.name
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    return repr(x)


class StatusServer:
    """Opt-in HTTP endpoint. ``port=0`` picks a free port (see ``.port``)."""

    def __init__(self, aggregator: HealthAggregator, host: str = "127.0.0.1", port: int = 0,
                 metrics_text: Optional[Callable[[], str]] = None) -> None:
        self.aggregator = aggregator
        self.host = host
        self.requested_port = port
        self.metrics_text = metrics_text
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        return self._httpd.server_address[1] if self._httpd else self.requested_port

    def start(self) -> "StatusServer":
        agg, mtext = self.aggregator, self.metrics_text

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib naming
                if self.path.startswith("/status"):
                    body, ctype, code = agg.to_json().encode(), "application/json", 200
                elif self.path.startswith("/metrics") and mtext is not None:
                    body, ctype, code = mtext().encode(), "text/plain; version=0.0.4", 200
                else:
                    body, ctype, code = b'{"error":"not found"}', "application/json", 404
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:  # silence stderr access log
                pass

        self._httpd = ThreadingHTTPServer((self.host, self.requested_port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True,
                                        name="cheesestack-status")
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    def __enter__(self) -> "StatusServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
