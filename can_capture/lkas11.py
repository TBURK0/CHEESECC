#!/usr/bin/env python3
"""LKAS11 (0x340) frame builder + validator — ANALYSIS / BENCH-LOOPBACK ONLY.

This shows how a Hyundai lateral-steering command frame is ASSEMBLED and proves the
construction against your real captures. It is for understanding and for loopback
testing (generate a frame, feed it back into a logger, confirm it decodes).

!!  DO NOT TRANSMIT THESE FRAMES ONTO A VEHICLE'S STEERING BUS.  !!
On a 2019 Elantra the stock LKAS is speed-gated (~32 mph) and the stock camera also
sends 0x340; injecting from here would fight the camera and/or require driving at
speed with no safety layer. The supported, safe way to send lateral commands on this
car is openpilot on a comma 3X + the hyundai_g harness, on-road, as designed.

    python lkas11.py bench_baseline_02.csv      # validate construction vs real frames

Verified from 1548 captured 0x340 frames:
  * checksum  = sum of the other 7 bytes & 0xFF   (byte 6)
  * counter   = high nibble of byte 4, 0..15 rolling (+1 per message)
The TORQUE field (bytes 0-1) read 0x0000 in every capture (all taken in park, where
LKAS never steers), so its exact scale/sign/bit-width is NOT confirmed here — it must
be derived from an on-steering capture and cross-checked against opendbc. The builder
below refuses to fabricate a nonzero torque it can't verify.
"""
import sys

LKAS11_ID = 0x340
# Constant bytes observed across all captures (idle state). Byte4 low nibble 0x08 is a
# flag; its high nibble is the counter. Byte6 is the checksum (filled in by build()).
TEMPLATE = [0x00, 0x00, 0x00, 0x04, 0x08, 0x20, 0x00, 0x11]


def checksum(d):
    """Byte 6 = sum of the other 7 bytes, mod 256. Verified on 1548 frames."""
    return sum(d[i] for i in range(8) if i != 6) & 0xFF


def build(counter, torque=0):
    """Assemble an LKAS11 frame with a given rolling counter and (verified-only) torque.

    torque must be 0 — the nonzero encoding is UNCONFIRMED from our data, and this tool
    will not emit a steering value it can't validate. Confirm the field from an
    on-steering capture + opendbc before ever encoding a real torque.
    """
    if torque != 0:
        raise ValueError("torque encoding UNCONFIRMED (all captures were 0); refusing to fabricate it")
    d = list(TEMPLATE)
    d[4] = ((counter & 0x0F) << 4) | (TEMPLATE[4] & 0x0F)
    d[6] = checksum(d)
    return d


def parse_capture(path):
    """Yield the 8 data bytes of every 0x340 frame in one of our CSV captures."""
    with open(path) as f:
        for line in f:
            if ",0x340," not in line:
                continue
            tail = line.split(",0x340,", 1)[1]          # e.g. "0,0,8,00 00 00 04 98 20 CD 11"
            hexbytes = tail.split(",")[-1].split()
            if len(hexbytes) == 8:
                yield [int(b, 16) for b in hexbytes]


def validate(path):
    total = cs_ok = rt_ok = 0
    counters = set()
    for d in parse_capture(path):
        total += 1
        if checksum(d) == d[6]:
            cs_ok += 1
        counter = d[4] >> 4
        counters.add(counter)
        # round-trip: rebuild from the counter alone and compare (only valid for torque==0 frames)
        if d[0] == 0 and d[1] == 0 and build(counter) == d:
            rt_ok += 1
    print(f"frames checked:         {total}")
    print(f"checksum matches:       {cs_ok}/{total}")
    print(f"round-trip rebuilt:     {rt_ok}/{total}  (byte-identical from counter alone)")
    print(f"counter values seen:    {len(counters)}/16")
    assert total and cs_ok == total and rt_ok == total, "construction does NOT match the captures"
    print("\nOK — frame construction (counter + checksum) reproduces the real frames exactly.")
    print("Demo: 4 consecutive idle frames we would generate (counter 0..3):")
    for c in range(4):
        print("  " + " ".join(f"{b:02X}" for b in build(c)))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "bench_baseline_02.csv"
    validate(path)
