"""Step 0 pass check: every stream on one shared time axis.

    python -m tools.plot_log logs/run_X.jsonl.gz [-o out.png] [--t0 5 --t1 20]

Event markers (fusion.messages.Event) are drawn as vertical lines on every
panel, so a 'tap' should line up with spikes in the IMU and camera streams.
"""
import argparse
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fusion.io import ReplaySource

# categorical slots 1-3 (validated as a set) for x/y/z; more series cycle
# through the remaining slots in fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(path, t0=None, t1=None):
    data = defaultdict(lambda: defaultdict(list))
    events = []
    src = ReplaySource(path)
    for m in src:
        if (t0 is not None and m.t < t0) or (t1 is not None and m.t > t1):
            continue
        name = type(m).__name__
        if name == "ImuSample":
            data["imu"]["t"].append(m.t)
            data["imu"]["gyro"].append(m.gyro)
            data["imu"]["accel"].append(m.accel)
        elif name == "AccelSample":
            data[f"acc:{m.unit}"]["t"].append(m.t)
            data[f"acc:{m.unit}"]["accel"].append(m.accel)
        elif name == "UltrasonicReading":
            data[f"us:{m.sensor}"]["t"].append(m.t)
            data[f"us:{m.sensor}"]["r"].append(np.nan if m.range_m is None else m.range_m)
        elif name == "WheelSpeed":
            data["wheel"]["t"].append(m.t)
            data["wheel"]["ticks"].append(m.ticks)
        elif name == "VoPose":
            data["vo"]["t"].append(m.t)
            data["vo"]["xy"].append((m.x, m.y))
        elif name == "Command":
            data["cmd"]["t"].append(m.t)
            data["cmd"]["steer"].append(m.steer_rad)
            data["cmd"]["throttle"].append(m.throttle)
        elif name == "Event":
            events.append((m.t, m.text))
    return data, events, src.header


def style(ax, ylabel):
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot(path, out, t0=None, t1=None):
    data, events, header = load(path, t0, t1)
    panels = []
    if "imu" in data:
        panels += ["gyro", "accel"]
    acc_units = sorted(k for k in data if k.startswith("acc:"))
    if acc_units:
        panels.append("extra")
    us = sorted(k for k in data if k.startswith("us:"))
    if us:
        panels.append("us")
    if "wheel" in data:
        panels.append("wheel")
    if "vo" in data:
        panels.append("vo")
    if "cmd" in data:
        panels.append("cmd")
    if not panels:
        raise SystemExit("No plottable messages in log.")

    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.1 * len(panels)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    lw = 1.0
    for ax, p in zip(axes, panels):
        if p in ("gyro", "accel"):
            t = np.array(data["imu"]["t"])
            v = np.array(data["imu"][p])
            for i, lab in enumerate("xyz"):
                ax.plot(t, v[:, i], color=SERIES[i], lw=lw, label=lab)
            style(ax, "gyro rad/s" if p == "gyro" else "MPU accel m/s²")
        elif p == "extra":
            for i, k in enumerate(acc_units):
                v = np.array(data[k]["accel"])
                ax.plot(data[k]["t"], np.linalg.norm(v, axis=1), color=SERIES[i % 8],
                        lw=lw, label=k[4:])
            style(ax, "|accel| m/s²")
        elif p == "us":
            for i, k in enumerate(us):
                ax.plot(data[k]["t"], data[k]["r"], ".", ms=3, color=SERIES[i % 8], label=k[3:])
            style(ax, "range m (gap = no echo)")
        elif p == "wheel":
            ax.plot(data["wheel"]["t"], data["wheel"]["ticks"], color=SERIES[0], lw=lw)
            style(ax, "encoder ticks")
        elif p == "vo":
            xy = np.array(data["vo"]["xy"])
            ax.plot(data["vo"]["t"], xy[:, 0], color=SERIES[0], lw=lw, label="x")
            ax.plot(data["vo"]["t"], xy[:, 1], color=SERIES[1], lw=lw, label="y")
            style(ax, "VO position m")
        elif p == "cmd":
            ax.plot(data["cmd"]["t"], data["cmd"]["steer"], color=SERIES[0], lw=lw, label="steer rad")
            ax.plot(data["cmd"]["t"], data["cmd"]["throttle"], color=SERIES[1], lw=lw, label="throttle")
            style(ax, "commands")
        if len(ax.get_lines()) > 1:
            ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.98), fontsize=8, frameon=False, ncol=4, labelcolor=INK)
        for te, _ in events:
            ax.axvline(te, color=MUTED, lw=0.8, ls="--")
    for te, text in events:
        axes[0].annotate(text, (te, 1.0), xycoords=("data", "axes fraction"),
                         xytext=(3, 2), textcoords="offset points", fontsize=8, color=INK)
    axes[-1].set_xlabel("time s (capture clock)", color=MUTED, fontsize=9)
    fig.suptitle(f"{header.get('created', '')}  {header.get('notes', '')}", color=INK,
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor="#fcfcfb")
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("-o", "--out", default="log_plot.png")
    ap.add_argument("--t0", type=float)
    ap.add_argument("--t1", type=float)
    a = ap.parse_args()
    plot(a.log, a.out, a.t0, a.t1)
