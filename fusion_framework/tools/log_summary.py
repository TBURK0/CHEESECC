"""Step 0/2 sanity check for a recorded run.

For each stream: message count, actual rate, the largest gap between
samples, and timestamps that go backwards. Run it after every test; a
dropped-sample or clock problem found here saves hours of filter tuning.

    python -m tools.log_summary logs/run_20261108_120000.jsonl.gz
"""
import sys
from collections import defaultdict

import numpy as np

from fusion.io import ReplaySource


def summarize(path):
    times = defaultdict(list)
    arrival_backwards = defaultdict(int)
    last = {}
    # deliver in file (arrival) order to catch clock problems at the source
    src = ReplaySource(path, reorder_window_s=None)
    for msg in src:
        key = type(msg).__name__
        sub = getattr(msg, "sensor", None) or getattr(msg, "unit", None)
        if sub:
            key = f"{key}[{sub}]"
        if key in last and msg.t < last[key]:
            arrival_backwards[key] += 1
        last[key] = msg.t
        times[key].append(msg.t)

    print(f"Run: {src.header.get('created')}  notes: {src.header.get('notes')!r}")
    print(f"{'stream':32s} {'count':>7s} {'rate Hz':>8s} {'max gap s':>10s} {'backwards':>9s}")
    rows = {}
    for key in sorted(times):
        t = np.sort(np.array(times[key]))
        dur = t[-1] - t[0] if len(t) > 1 else 0.0
        rate = (len(t) - 1) / dur if dur > 0 else float("nan")
        gap = float(np.diff(t).max()) if len(t) > 1 else float("nan")
        rows[key] = dict(count=len(t), rate=rate, max_gap=gap, backwards=arrival_backwards[key])
        print(f"{key:32s} {len(t):7d} {rate:8.1f} {gap:10.3f} {arrival_backwards[key]:9d}")
    return rows


if __name__ == "__main__":
    summarize(sys.argv[1])
