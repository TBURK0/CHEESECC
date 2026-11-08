#!/usr/bin/env python3
"""Save CAN logger serial output to a CSV file."""
import sys
import time
import serial

if len(sys.argv) != 3:
    sys.exit(f"Usage: python {sys.argv[0]} <COM_PORT> <OUTPUT_FILE.csv>")

port, out = sys.argv[1], sys.argv[2]

# Ensure baud rate matches Arduino (1,000,000)
try:
    ser = serial.Serial(port, 1000000, timeout=1)
except Exception as e:
    sys.exit(
        f"Could not open {port}. Make sure Arduino Serial Monitor is CLOSED! Error: {e}"
    )

n = 0
print(f"Logging {port} -> {out} (Press Ctrl+C to stop)...")

with open(out, "w", newline="") as f:
    # Write CSV Header
    f.write("timestamp,id,dlc,data\n")
    try:
        while True:
            raw_line = ser.readline().decode("ascii", "replace").strip()
            if not raw_line:
                continue

            # Record timestamp in seconds with microsecond precision
            ts = f"{time.time():.4f}"
            f.write(f"{ts},{raw_line}\n")
            n += 1

            if n % 100 == 0:
                f.flush()
                print(f"\rCaptured {n} frames...", end="", flush=True)

    except KeyboardInterrupt:
        f.flush()
        print(f"\nStopped! Saved {n} frames to {out}")