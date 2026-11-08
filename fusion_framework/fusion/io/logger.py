"""Step 0: run logger.

One gzipped JSON-lines file per run. Line 1 is a header (run metadata and a
snapshot of the config used), every following line is one message. JSON is
slower than a binary format but you can open it with anything, and at a few
hundred Hz it is plenty fast. Switch to a binary format only if profiling on
the car says you must.

Usage from your existing hardware loop:

    with RunLogger("logs", notes="tap test, motor off", config=cfg) as log:
        while running:
            log.write(ImuSample(t=..., gyro=..., accel=...))
"""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

from ..messages import to_dict

LOG_VERSION = 1


class RunLogger:
    def __init__(self, directory, notes: str = "", config: dict | None = None,
                 name: str | None = None):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = self.dir / f"{name or 'run'}_{stamp}.jsonl.gz"
        self._header = {"_type": "Header", "version": LOG_VERSION,
                        "created": stamp, "notes": notes, "config": config or {}}
        self._f = None
        self.count = 0

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    def open(self):
        self._f = gzip.open(self.path, "wt", encoding="utf-8")
        self._f.write(json.dumps(self._header) + "\n")

    def write(self, msg):
        self._f.write(json.dumps(to_dict(msg), separators=(",", ":")) + "\n")
        self.count += 1

    def close(self):
        if self._f:
            self._f.close()
            self._f = None
