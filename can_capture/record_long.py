#!/usr/bin/env python3
"""Long-duration CAN logger. Robust for multi-hour / full-drive captures.

    pip install pyserial
    python record_long.py COM5 drive_lkas_on

Writes rotating files   drive_lkas_on_0001.csv, _0002.csv, ...   (default 100 MB each)
next to a session log that records gaps, reconnects and per-10s frame rate.

Why rotate + reconnect: a single giant file is painful and risks losing everything
on a crash; a long drive will have USB hiccups you want to survive, not abort on.

Flags:
    --baud 1000000      serial baud (match the Arduino sketch)
    --rotate-mb 100     new file every N megabytes
    --split-min 0       also start a new file every N minutes (0 = off)
    --quiet             don't print the live rate line

Ctrl-C stops cleanly and prints totals. Row format is passed through from the
Arduino unchanged, with a host wall-clock timestamp prepended:
    host_epoch, <whatever the sketch prints>
"""
import argparse
import sys
import time
from datetime import datetime

try:
    import serial
except ImportError:
    sys.exit("pyserial not installed:  pip install pyserial")


def open_port(port, baud):
    """Open the port, retrying forever until it succeeds (survives unplug/replug)."""
    while True:
        try:
            s = serial.Serial(port, baud, timeout=1)
            s.reset_input_buffer()
            return s
        except Exception as e:  # noqa: BLE001
            print(f"\r[{datetime.now():%H:%M:%S}] waiting for {port}: {e}        ", end="", flush=True)
            time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port")
    ap.add_argument("prefix", help="output name prefix, e.g. drive_lkas_on")
    ap.add_argument("--baud", type=int, default=1000000)
    ap.add_argument("--rotate-mb", type=float, default=100.0)
    ap.add_argument("--split-min", type=float, default=0.0)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    rotate_bytes = int(a.rotate_mb * 1024 * 1024)
    split_sec = a.split_min * 60
    session_log = f"{a.prefix}_session.log"

    def log(msg):
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print("\n" + line)
        with open(session_log, "a") as lf:
            lf.write(line + "\n")

    ser = open_port(a.port, a.baud)
    log(f"recording {a.port} @ {a.baud} -> {a.prefix}_NNNN.csv (rotate {a.rotate_mb} MB"
        + (f", split {a.split_min} min)" if split_sec else ")"))

    file_idx = 0
    f = None
    file_bytes = 0
    file_started = 0.0

    total = 0            # frames this session
    window = 0           # frames in the current 10 s rate window
    window_start = time.monotonic()
    last_rx = time.monotonic()

    def new_file():
        nonlocal f, file_idx, file_bytes, file_started
        if f:
            f.flush()
            f.close()
        file_idx += 1
        name = f"{a.prefix}_{file_idx:04d}.csv"
        f = open(name, "w", newline="")
        f.write("host_epoch,arduino_line\n")
        file_bytes = 0
        file_started = time.monotonic()
        log(f"-> {name}")

    new_file()
    try:
        while True:
            try:
                raw = ser.readline()
            except Exception as e:  # noqa: BLE001 - USB dropped mid-drive
                log(f"serial error: {e} — reconnecting")
                try:
                    ser.close()
                except Exception:  # noqa: BLE001
                    pass
                ser = open_port(a.port, a.baud)
                log("reconnected")
                continue

            now = time.monotonic()
            if not raw:
                # no data for a while during an active session is worth noting
                if now - last_rx > 3.0:
                    log(f"no frames for {now - last_rx:.0f}s (engine off? cable? bus asleep?)")
                    last_rx = now
                continue
            last_rx = now

            line = raw.decode("ascii", "replace").strip()
            if not line or line.startswith("#"):
                continue

            row = f"{time.time():.4f},{line}\n"
            f.write(row)
            file_bytes += len(row)
            total += 1
            window += 1

            # rotate by size or time
            if file_bytes >= rotate_bytes or (split_sec and now - file_started >= split_sec):
                new_file()

            # live rate every 10 s — the number to watch for dropped frames
            if now - window_start >= 10.0:
                rate = window / (now - window_start)
                f.flush()
                if not a.quiet:
                    print(f"\r{datetime.now():%H:%M:%S}  {total:>9,} frames  "
                          f"{rate:>7.0f} f/s now  (file {file_idx})      ", end="", flush=True)
                window = 0
                window_start = now
    except KeyboardInterrupt:
        pass
    finally:
        if f:
            f.flush()
            f.close()
        log(f"stopped. {total:,} frames across {file_idx} file(s).")


if __name__ == "__main__":
    main()
