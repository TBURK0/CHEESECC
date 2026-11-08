# CAN steering project — handover

Self-contained recap for a fresh Claude/engineer taking over. Read it top to bottom.

## Goal
Command the **steering** of a **2019 Hyundai Elantra** over CAN — turn the front wheels
from camera feedback — with the **car suspended on a lift (wheels off the ground), not
moving**. Gas/brake are out of scope (done by hand). This is a *steering test bench*, not
a car that drives.

## Hardware on hand
- **Arduino UNO Q + Inland (MCP2515) CAN shield** — the only CAN interface. CS pin 9 (try 10 if init fails), 500 kbps, crystal auto-detected (16 or 8 MHz).
- A "CANLink" device (unspecified) — treated as just another CAN interface.
- **No comma device, no panda.** This matters (see Blocker 6).

## What works and is DONE (all safe, read-only)
- **Passive logging.** `can_logger/can_logger.ino` runs the MCP2515 in **listen-only** (physically can't transmit) and drains every frame per loop (no `delay()`). `capture_v2.py` saves serial → CSV with host + device-µs timestamps.
  - Early captures were throttled to ~20 frames/s by a leftover `delay()` in the desk-test sketch (fixed).
  - **Capture rate is still inconsistent and the "peak" numbers lie.** A big spike in the FIRST second of a file (e.g. 7356 f/s) is just the serial/MCP2515 **backlog dumping at startup**, all stamped in one host-time bucket — not real sustained throughput. Check the *steady-state* rate after the first second: `bench_baseline_01.csv` settles to **~70 f/s** (starved — dropping most frames), but `bench_baseline_02.csv` sustained **~1300 f/s** (good). A live Hyundai C-CAN is thousands/s, so a starved run is missing most frames.
  - **Root issue:** Uno + MCP2515 (2 RX buffers) + per-frame serial print is marginal for sustained full-bus logging. For reliable dense capture, a USB-CAN adapter (CANable/slcan), a native-CAN MCU (Teensy 4 + transceiver), or a panda is the right tool. With the Uno, verify each capture's steady-state rate before trusting it, and prefer the densest run.
- **Bus identified.** The IDs match the standard Hyundai C-CAN (opendbc `hyundai`). Steering-relevant:
  | ID | opendbc name | role |
  |----|--------------|------|
  | **0x340** | LKAS11 | lane-keep **command** (what steers) — openpilot sends this |
  | **0x251** | MDPS12 | steering torque feedback |
  | **0x2B0** | SAS11 | steering angle sensor (true wheel angle) |
  | 0x386 / 0x4F1 | WHL_SPD11 / CLU11 | wheel speeds / cluster speed |

## Current data (`can_capture/`)
- `bench_baseline_02.csv` — **best capture**, sustained ~1300 f/s, 1548 LKAS11 frames. Use this one.
- `bench_baseline_01.csv`, `bench_ground_fixed_01.csv` — spike on boot then starve to ~70 f/s. Partial; don't rely on them. (Note: the leading per-file number is a startup-backlog artifact, not the real rate — see the logging note above.)
- `test.csv` — old throttled capture (~8 f/s), engine cycled on/off. Not usable.
- `lkas_off.csv` — **not a capture**; it's leftover loopback desk-test text. Ignore/delete.
- **Key finding:** across all captures, **LKAS11 (0x340) torque bytes are `00 00` in every frame.** The stock lane-keep was never actively steering, so we have **no reference of what a commanded turn looks like**. Observed 0x340 structure (matches opendbc LKAS11, bit offsets TO CONFIRM against opendbc): torque in the low bytes (always 0 here), byte5=`0x20` const, a 4-bit **rolling counter** in byte 4's high nibble, a **checksum** in byte 6, byte3=`0x04`/byte7=`0x11` flags.

## LKAS11 (0x340) decode — verified on 1548 frames (bench_baseline_02.csv)

8 data bytes. In this capture only byte4 and byte6 varied; the rest were constant
because the system was idle (torque 0). Example: `00 00 00 04 98 20 CD 11`.

| Byte | Value here | Meaning |
|------|-----------|---------|
| 0–1 | `00 00` | **steering torque request** (CR_Lkas_StrToqReq). **0 in ALL captures — never saw a real steer.** Scale/sign/bit-width UNCONFIRMED; needs an on-steering capture + opendbc. |
| 2 | `00` | flags (const here) |
| 3 | `04` | LKAS state flags (const here) |
| 4 | `_8` | **rolling counter** = high nibble (0–15, +1 per msg). Low nibble `0x8` = flag, const here. `byte4 = (counter<<4) | 0x08` |
| 5 | `20` | flags (const here) |
| 6 | varies | **checksum** (see below) |
| 7 | `11` | LKAS config flags (const here) |

**Checksum (CONFIRMED, 1548/1548 frames):** `byte6 = (sum of the other 7 bytes) & 0xFF`.

```c
// LKAS11 0x340 — verified against 1548 captured frames
uint8_t lkas11_checksum(const uint8_t d[8]) {   // result goes in d[6]
    uint16_t s = 0;
    for (int i = 0; i < 8; i++) if (i != 6) s += d[i];
    return s & 0xFF;
}
// counter: d[4] = (counter << 4) | 0x08;  counter = (counter + 1) & 0x0F each message
```

Caveat: this is the **idle** state. The checksum sum holds regardless of byte values, but
the torque field encoding and the byte 2/3/5/7 flag bits must be re-derived from a capture
where the stock system is **actively steering** (Blocker 1), and cross-checked against the
opendbc `hyundai` LKAS11 definition before trusting any of it.

## Blockers to "send turns to the wheels" — read before writing any TX code
1. **No on-steering reference.** Torque is 0 in all logs — and all captures so far were **in PARK**. You must capture LKAS11 while the stock system is *actively applying torque*, which (see #2) means **driving above ~32 mph with lane-keep engaged and correcting**. You cannot get this parked or on a lift.
2. **CONFIRMED: LKAS is speed-gated — min steering speed ~32 mph on this car** (opendbc `HYUNDAI_ELANTRA`; older READMEs say 34; min *enable* ~19 mph). The MDPS will not accept lane-keep torque below that, so it **never steers at a standstill / on a lift**. This largely breaks the "turn the suspended wheels via LKAS" plan: the only ways around it are spoofing vehicle speed to the MDPS (deep, fragile, multi-ECU, fault-prone — not a weekend job) or doing it on the road above 32 mph as designed. Flag this to whoever continues; it changes the whole approach.
3. **You must REPLACE the camera's 0x340, not add a second one.** The stock lane-keep camera broadcasts 0x340 continuously. Two senders on one ID = bus errors / MDPS ignores it / fault. openpilot solves this with a **harness that intercepts the camera connector** and blocks the stock camera. A plain Y-tap from the Arduino can't do this.
4. **Checksum + rolling counter must be reproduced exactly** or the MDPS rejects the frame. Decode both from a real on-steering capture against opendbc's LKAS11 checksum (it's a documented 6-bit/byte sum variant — verify).
5. **Physical safety layer is specified but NOT built/verified.** Required and must fail safe: E-stop → **normally-open contactor (energize-to-close)** that removes MDPS power (any break = unpowered); **double-pole CAN kill** opening CAN-H and CAN-L; ARMED indicator light; everyone's hands clear of wheels/tie rods while armed. Verify each with a meter BEFORE any TX reaches the car.
6. **The Arduino has no safety firmware.** On openpilot, the actuation safety (torque limit, rate limit, driver-override cutout) lives in the **panda**, not in the message-builder. Sending raw 0x340 from an Arduino is injection with none of that enforcement.

## What's been built toward the safe version
- `steer_bench/steer_bench.ino` — a bench controller with a **software safety envelope** (e-stop input, command timeout → release, rate limiter, tiny torque clamp, arm/disarm, ARMED LED). It is **inert by design**: MCP2515 stays listen-only (`ENABLE_STEER_OUTPUT 0`) and `build_steer_frame()` is an empty stub. It cannot move anything until someone decodes the real frame and enables output.
- The physical safety spec above.

## Position I (the handing-over assistant) held — carry it forward
- Helped fully with **logging, decoding, and a safety-enveloped suspended-wheels bench**.
- **Declined to write safety-stripped steering injection** for a real car (no limits / car could move). Urgency/closed-facility don't change that; the hazard is unchanged.
- Recommended the real actuation path: **openpilot on a comma 3X + the `hyundai_g` harness**. The 2019 Elantra **is supported** (`HYUNDAI_ELANTRA`, SCC + LKAS, stock longitudinal, full steering torque, min steer ~32 mph, min enable ~19 mph). This inherits the camera interception (Blocker 3) and the safety firmware (Blocker 6) instead of rebuilding them wrong — but it only steers **on the road above ~32 mph**, not on a bench.

## Capture tooling
- `record_long.py` — long-duration logger: rotating files, auto-reconnect on USB drops, per-10s rate display, session log. For full drives. Use: `python record_long.py COM5 drive_lkas_on`.
- Reminder: a longer capture does NOT fix the torque=0 problem — that's about driving state (speed > ~32 mph, lane-keep engaged), not duration. Watch the live f/s; if it starves below the bus rate, the Uno+MCP2515 is dropping frames (use a USB-CAN adapter / panda for reliable dense logging).

## Concrete next steps (in order)
1. Capture **LKAS11 actively steering** (stock lane-keep engaged and tugging) → the missing reference.
2. Decode 0x340 against opendbc: torque scale/sign, rolling counter, checksum. Validate the checksum by recomputing it over the captured frames.
3. Decide the camera-interception approach (Blocker 3) — you cannot command steering without it.
4. Build + **meter-verify** the physical safety layer (Blocker 5) before any transmit.
5. Only then, on the lift, output enabled, smallest torque first, hand on the E-stop.
