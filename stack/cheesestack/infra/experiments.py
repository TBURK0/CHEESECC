"""Experiment tracking: runs with params, metric series and artifacts on disk.

Layout (``root/<experiment>/<run_id>/``)::

    run.json        RunInfo: id, name, status, params, tags, start/end
    metrics.jsonl   {"key", "value", "step", "t"} per logged point
    artifacts/      copied files / written bytes

Run ids are deterministic ``<seq:04d>-<name>`` so tests and reports are
stable. ``compare_runs`` builds a table of params and final (or best) metric
values across runs and marks which params differ - the core of "what changed
between these two sim evaluations?".

TODO(infra): MLflow/W&B export, git sha + dirty flag capture, system metrics,
nested runs / sweeps, a web view.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from cheesestack.core import Clock, WallClock


@dataclass
class ExperimentConfig:
    root: str = "experiments"


@dataclass
class RunInfo:
    run_id: str
    experiment: str
    name: str
    status: str = "running"  # running | finished | failed
    params: dict = field(default_factory=dict)
    tags: dict = field(default_factory=dict)
    start: float = 0.0
    end: Optional[float] = None


def _jsonable(v: Any) -> Any:
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return repr(v)


class Run:
    def __init__(self, path: Path, info: RunInfo, clock: Clock) -> None:
        self.path = path
        self.info = info
        self.clock = clock

    @property
    def run_id(self) -> str:
        return self.info.run_id

    def _save(self) -> None:
        tmp = self.path / "run.json.tmp"
        tmp.write_text(json.dumps(asdict(self.info), indent=2, sort_keys=True))
        os.replace(tmp, self.path / "run.json")

    def log_params(self, **params) -> None:
        self.info.params.update({k: _jsonable(v) for k, v in params.items()})
        self._save()

    def set_tags(self, **tags) -> None:
        self.info.tags.update({k: str(v) for k, v in tags.items()})
        self._save()

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        if step is None:
            step = len(self.metric_history(key))
        rec = {"key": key, "value": float(value), "step": int(step), "t": float(self.clock.now())}
        with (self.path / "metrics.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        for k, v in metrics.items():
            self.log_metric(k, v, step)

    def metrics(self) -> dict[str, list[tuple[int, float]]]:
        out: dict[str, list[tuple[int, float]]] = {}
        p = self.path / "metrics.jsonl"
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    out.setdefault(r["key"], []).append((r["step"], r["value"]))
        for v in out.values():
            v.sort(key=lambda sv: sv[0])
        return out

    def metric_history(self, key: str) -> list[tuple[int, float]]:
        return self.metrics().get(key, [])

    def log_artifact(self, src: str | os.PathLike | bytes, name: Optional[str] = None) -> Path:
        d = self.path / "artifacts"
        d.mkdir(exist_ok=True)
        if isinstance(src, (bytes, bytearray)):
            if not name:
                raise ValueError("name required for bytes artifacts")
            dst = d / name
            dst.write_bytes(bytes(src))
        else:
            dst = d / (name or Path(src).name)
            shutil.copyfile(src, dst)
        return dst

    def artifacts(self) -> list[str]:
        d = self.path / "artifacts"
        return sorted(p.name for p in d.iterdir()) if d.exists() else []

    def finish(self, status: str = "finished") -> None:
        self.info.status = status
        self.info.end = float(self.clock.now())
        self._save()

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, *exc) -> None:
        self.finish("failed" if exc_type else "finished")


class ExperimentTracker:
    def __init__(self, config: Optional[ExperimentConfig] = None, clock: Optional[Clock] = None) -> None:
        self.config = config or ExperimentConfig()
        self.root = Path(self.config.root)
        self.clock = clock or WallClock()

    def start_run(self, experiment: str, name: str = "run", params: Optional[dict] = None,
                  tags: Optional[dict] = None) -> Run:
        ed = self.root / experiment
        ed.mkdir(parents=True, exist_ok=True)
        seq = len([p for p in ed.iterdir() if p.is_dir()])
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        run_id = f"{seq:04d}-{safe}"
        path = ed / run_id
        path.mkdir()
        info = RunInfo(run_id, experiment, name, start=float(self.clock.now()))
        run = Run(path, info, self.clock)
        run._save()
        if params:
            run.log_params(**params)
        if tags:
            run.set_tags(**tags)
        return run

    def get_run(self, experiment: str, run_id: str) -> Run:
        path = self.root / experiment / run_id
        info = RunInfo(**json.loads((path / "run.json").read_text()))
        return Run(path, info, self.clock)

    def list_runs(self, experiment: str) -> list[Run]:
        ed = self.root / experiment
        if not ed.exists():
            return []
        return [self.get_run(experiment, p.name) for p in sorted(ed.iterdir()) if (p / "run.json").exists()]

    def experiments(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir()) if self.root.exists() else []

    def best_run(self, experiment: str, metric: str, mode: str = "min") -> Optional[Run]:
        scored = []
        for r in self.list_runs(experiment):
            h = r.metric_history(metric)
            if h:
                scored.append((h[-1][1], r))
        if not scored:
            return None
        pick = min if mode == "min" else max
        return pick(scored, key=lambda s: s[0])[1]


def compare_runs(runs: list[Run], metrics: Optional[list[str]] = None, agg: str = "last") -> dict:
    """Table of params + aggregated metrics per run.

    agg: last | min | max | mean. Returns
    ``{"runs": [ids], "params": {k: [..]}, "metrics": {k: [..]}, "differing_params": [..]}``.
    """
    import numpy as np
    fns = {"last": lambda v: v[-1], "min": np.min, "max": np.max, "mean": np.mean}
    if agg not in fns:
        raise ValueError(f"agg must be one of {sorted(fns)}")
    pkeys = sorted({k for r in runs for k in r.info.params})
    all_metrics = [r.metrics() for r in runs]
    mkeys = metrics or sorted({k for m in all_metrics for k in m})
    params = {k: [r.info.params.get(k) for r in runs] for k in pkeys}
    mtable = {}
    for k in mkeys:
        col = []
        for m in all_metrics:
            vals = [v for _, v in m.get(k, [])]
            col.append(float(fns[agg](vals)) if vals else None)
        mtable[k] = col
    differing = [k for k, col in params.items()
                 if len({json.dumps(v, sort_keys=True) for v in col}) > 1]
    return {"runs": [r.run_id for r in runs], "params": params, "metrics": mtable,
            "differing_params": differing}
