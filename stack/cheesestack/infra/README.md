# infra

| Module | Capability | Works now | Key TODOs |
|---|---|---|---|
| `config.py` | Configuration management | Layered dataclass config: defaults <- JSON/TOML <- `CHEESE_*` env (`__` nesting) <- dotted overrides; type coercion, strict unknown keys, `validate()` hooks, JSON round-trip | TOML writer, per-key provenance, schema export |
| `params.py` | Parameter server | Thread-safe `/ns/param` store, typed + ranged, prefix change callbacks, namespace views, tree/list, snapshot/restore | Network sync, persistence, read-only params |
| `telemetry.py` | Telemetry | Channel decimation + rate limit, bounded buffer, priority-ordered MTU-bounded JSON packets (optional zlib), MessageBus tap | Binary encoding, store-and-forward, ack/retransmit |
| `metrics.py` | Metrics | Labelled counters/gauges/histograms (buckets + reservoir percentiles), timer context manager, Prometheus text, JSON snapshot | OTLP/push export, t-digest |
| `logging_setup.py` | Logging | Idempotent JSON logging setup, extras, sim-time stamp, per-module levels, rotating file | Uplink shipping, rate limiting, redaction |
| `storage.py` | Data storage | Segmented append-only JSONL record store with time index + orphan recovery; npz `ArrayStore` for dense streams | MCAP export, compression, retention, file locks |
| `compression.py` | Data compression | zlib/lzma with codec header, zigzag+varint, quantized delta(0/1/2) timeseries codec (error <= step/2), uint8/16 quantization | Gorilla floats, zstd |
| `distributed.py` | Distributed processing | Thread/process task queue with retries + exponential backoff, deterministic map-reduce over shards | Ray/K8s backend, priorities, cancellation |
| `ci.py` (+ `.github/workflows/stack-ci.yml`) | CI/CD | Import every module, run pytest; `python -m cheesestack.infra.ci [--imports]` | Coverage gate, lint/mypy, JUnit |
| `registry.py` | Model deployment | Versioned artifacts + sha256 verify, dev/staging/prod promotion (staging-before-prod guard), history, multi-step rollback | Signing, remote backend, staged OTA rollout |
| `monitoring.py` | Remote monitoring | Health aggregation (nodes/heartbeats, checks, expiring faults, metrics) -> JSON doc; opt-in `StatusServer` (`/status`, `/metrics`) | Auth, fleet push, alert rules |
| `experiments.py` | Experiment tracking | On-disk runs: params, tags, metric series, artifacts, status; `compare_runs`, `best_run` | MLflow export, git sha capture, sweeps |
