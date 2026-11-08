"""Run fusion layers over a recorded log and save their outputs as CSV.
This is the offline tuning loop: change a gain, re-run, compare CSVs.

    python -m tools.replay_fusion logs/run_X.jsonl.gz --layers attitude pose -o out.csv
    python -m tools.replay_fusion LOG --layers pose --late-vo   # VO delivered late, as live
"""
import argparse
import csv

from fusion.config import NotMeasuredError, load_config
from fusion.io import ReplaySource
from fusion.layers import AccelArrayLayer, AttitudeLayer, ObstacleLayer, PoseEKF
from fusion.node import FusionNode

LAYERS = {"attitude": AttitudeLayer, "accel_array": AccelArrayLayer,
          "pose": PoseEKF, "obstacles": ObstacleLayer}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--layers", nargs="+", required=True, choices=sorted(LAYERS))
    ap.add_argument("--config")
    ap.add_argument("--late-vo", action="store_true")
    ap.add_argument("--every", type=int, default=5, help="write every Nth state")
    ap.add_argument("-o", "--out", default="fusion_out.csv")
    a = ap.parse_args()

    cfg = load_config(a.config)
    try:
        node = FusionNode(cfg, [LAYERS[n](cfg) for n in a.layers])
    except NotMeasuredError as e:
        raise SystemExit(str(e))
    rows, k = [], 0

    def keep(msg, st):
        nonlocal k
        k += 1
        if k % a.every == 0 and st:
            rows.append({key: v for key, v in st.items() if isinstance(v, (int, float))})

    node.run(ReplaySource(a.log, deliver_at_arrival=a.late_vo), keep)
    if not rows:
        raise SystemExit("No state produced; are the layers' handlers implemented?")
    keys = sorted({key for r in rows for key in r}, key=lambda s: (s != "t", s))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
