# Jetson ↔ Arduino UNO Q serial protocol

Status: **v1, implemented in firmware and `vehicle_bridge`, verified only against the host simulator.**
Nothing here has been verified on the physical link yet; see `docs/hardware.md` → *Hardware verification checklist*.

The STM32 MCU on the UNO Q is the **safety authority**. The Jetson only *requests*; the MCU
clamps, times out, and stops on its own. Every rule in [Safety rules](#safety-rules) is
enforced on the MCU, regardless of what the Jetson sends.

## Physical link

| Item | Value | Status |
|---|---|---|
| Preferred path | Jetson USB → 3.3 V USB-UART adapter → UNO Q **MCU** UART header pins | UNVERIFIED: which header pins / `Serial*` object map to an MCU UART must be confirmed for the UNO Q core |
| Why not the UNO Q USB-C port | Believed to belong to the Qualcomm MPU (Linux), not the STM32. Going through it puts Linux + the Arduino Bridge in the safety path. | UNVERIFIED |
| Logic level | 3.3 V TTL. **Do not use a 5 V adapter.** | UNVERIFIED for UNO Q header tolerance; assume 3.3 V only |
| Format | 460800 baud, 8N1, no flow control | Default; reduce to 230400 if HIL shows errors |
| Firmware port | `LINK_SERIAL` macro in `firmware/arduino_uno_q/uno_q_vehicle/config.h` | Set during bring-up |

Bandwidth budget at defaults: 2 IMUs × 100 Hz × ~70 B + 4 × ~4 Hz × ~30 B + 10 Hz × ~60 B
≈ 15 kB/s, about one-third of 460800 baud (~46 kB/s).

## Framing

```
$<body>*<CRC>\n
```

- `body`: printable ASCII (0x20–0x7E) excluding `$` and `*`; comma-separated fields; **≤ 120 chars**.
- `CRC`: CRC-16/CCITT-FALSE of `body` bytes only (poly `0x1021`, init `0xFFFF`, no reflection,
  xorout `0x0000`), 4 **uppercase** hex digits. Check value: `CRC("123456789") = 29B1`.
- Terminator `\n`. A `\r` before it is ignored. Bytes before `$` are discarded.
- A line longer than 128 bytes is discarded up to the next `\n` and counts as a **bad frame**.
- All numeric fields are **decimal integers** (no floats on the wire). `-` means "not available".
- Example: `$ARM,1*` + CRC of `ARM,1` + `\n`.

A frame is **bad** if: CRC mismatch, missing `*`, overlength, non-integer where an integer is required,
wrong field count, or unknown type. Bad-frame handling: see Safety rules.

## Units and conventions

| Quantity | Wire unit | Range |
|---|---|---|
| Throttle | per-mille of full forward, 0…1000 | reverse not supported in v1 |
| Brake | per-mille of full brake, 0…1000 | |
| Steering | per-mille, −1000…1000, **positive = left** (REP-103, +yaw) | |
| Distance | millimetres | |
| Acceleration | mm/s² | |
| Angular rate | µrad/s | |
| Temperature | centi-°C | |
| MCU time `t_us` | `micros()` as uint32, **wraps every ~71.6 min** | host must unwrap |

IMU axes on the wire are the **chip's own axes** (as printed on the datasheet). Mapping into the
vehicle frame happens in `sensor_bridge` (see `docs/hardware.md` → *Axis conventions*).

## Host → MCU messages

Every message carries a host-chosen `seq` (uint32, should increase). The MCU answers every
non-`CMD` message with `ACK`. `CMD` is answered **only on error** (to save bandwidth); success
is visible in `STA.last_cmd_seq`.

| Body | Meaning |
|---|---|
| `CMD,<seq>,<throttle>,<brake>,<steer>` | Actuator request. Only accepted while `ARMED`. |
| `ARM,<seq>` | Request `DISARMED → ARMED`. Never moves anything by itself; a fresh `CMD` is required. |
| `DISARM,<seq>` | Request safe stop and `→ DISARMED`. Always OK (no-op if not armed; `ESTOP` stays `ESTOP`). |
| `ESTOP,<seq>` | Software emergency stop; latches `ESTOP`. |
| `RESET,<seq>` | Clear a latched `ESTOP → DISARMED`. Refused while the physical e-stop input is active. |
| `SCAN,<seq>` | I²C bus scan + chip identification; answered with `I2C` lines then `ACK`. |

## MCU → host messages

| Body | When |
|---|---|
| `BOOT,<fw_version>,<OUTPUTS_ENABLED\|OUTPUTS_DISABLED>,<HWWDT_ON\|HWWDT_UNVERIFIED>` | Once at startup |
| `ACK,<seq>,OK,-` / `ACK,<seq>,ERR,<reason>` | Reply to a host message |
| `STA,<t_us>,<state>,<faults>,<last_cmd_seq>,<cmd_age_ms>,<thr>,<brk>,<str>,<bad_frames>` | 10 Hz, and immediately on every state change |
| `US,<t_us>,<id>,<dist_mm>,<status>` | After each ultrasonic slot |
| `IMU,<t_us>,<src>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>,<temp_cC>,<status>` | 100 Hz per IMU |
| `I2C,<addr>,<ident>` … `I2C,END,<count>` | Reply to `SCAN` |
| `LOG,<text>` | Free-form diagnostics (text sanitised: no `$`, `*`, `,`) |

### `ACK` reasons

`ESTOP_ACTIVE`, `ESTOP_LATCHED`, `ESTOP_PIN_UNASSIGNED`, `NOT_ARMED`, `ALREADY_ARMED`,
`BAD_ARGS`, `UNKNOWN_TYPE`.

- Unknown type → bad frame **and** `ACK,<seq>,ERR,UNKNOWN_TYPE`.
- Known type with wrong field count / non-integer field → bad frame **and** `ACK,<seq>,ERR,BAD_ARGS`.
- In-format `CMD` with an out-of-range value → `ACK,<seq>,ERR,BAD_ARGS` (+ `INVALID_CMD` safe stop if armed); **not** counted in `bad_frames`.
- `<seq>` is `0` when it can't be parsed.
- `ARM` checks run in this order: `ALREADY_ARMED`, `ESTOP_LATCHED`, `ESTOP_ACTIVE`, `ESTOP_PIN_UNASSIGNED`.
  The e-stop input is sampled before messages are handled, so `ARM` with the button held returns `ESTOP_LATCHED`.
- `DISARM` while armed records no sticky fault (it's a requested stop, not a fault). `RESET` outside `ESTOP` is a no-op `OK`.

### `STA` fields

- `state`: `DISARMED` | `ARMED` | `ESTOP`.
- `faults`: 4-digit uppercase hex bitmask.
- `last_cmd_seq`: seq of the last **accepted** `CMD`, `-` if none since boot.
- `cmd_age_ms`: ms since last accepted `CMD`, capped at 65535 (65535 also means "never").
- `thr`,`brk`,`str`: the values **currently driven** to the actuator backend (after limits/safe-stop), per-mille.
- `bad_frames`: running count since boot.

### Fault bits

"Sticky" bits record *why* the last safe stop happened; they are cleared by the next successful `ARM`.
"Live" bits reflect a current condition.

| Bit | Name | Kind |
|---|---|---|
| `0x0001` | `ESTOP_INPUT` | live: physical e-stop input active (or wire broken) |
| `0x0002` | `ESTOP_LATCHED` | live: in `ESTOP` state |
| `0x0004` | `CMD_TIMEOUT` | sticky |
| `0x0008` | `BAD_FRAME` | sticky |
| `0x0010` | `INVALID_CMD` | sticky |
| `0x0020` | `ESTOP_PIN_UNASSIGNED` | live: firmware has no e-stop pin configured → arming impossible |
| `0x0040` | `OUTPUTS_DISABLED_BUILD` | live: built with `ENABLE_ACTUATOR_OUTPUT 0` (the default) |
| `0x0080` | `HW_WATCHDOG_UNVERIFIED` | live: hardware watchdog not confirmed running |
| `0x0100` | `IMU_FAULT` | live: an expected IMU is `NOT_FOUND` or `READ_FAIL` (always set until both IMUs are wired) |
| `0x0200` | `US_FAULT` | live: an ultrasonic's last status was `NO_ECHO`, `STUCK_HIGH` or `UNASSIGNED` (always set while pins are unassigned) |

`ESTOP_INPUT` is only reported when an e-stop pin is assigned. With no pin, `ESTOP_PIN_UNASSIGNED` is set instead and arming is refused.

### `US` fields

- `id`: `0`=FL, `1`=FR, `2`=RL, `3`=RR (front-left, front-right, rear-left, rear-right).
- `t_us`: MCU time of the trigger pulse.
- `dist_mm`: measured distance; `-` when no echo pulse was timed.
- `status`:

| Status | Meaning | `valid` | ROS `Range.range` |
|---|---|---|---|
| `OK` | `MIN ≤ d ≤ MAX` | yes | `d` |
| `TOO_CLOSE` | `d < US_MIN_MM` | no | `-inf` |
| `TOO_FAR` | `d > US_MAX_MM` | no | `+inf` |
| `TIMEOUT` | echo rose but stayed high past `US_ECHO_MAX_US` (HC-SR04 does this with nothing in range) | no | `+inf` |
| `NO_ECHO` | echo never rose within `US_ECHO_START_TIMEOUT_US` (disconnected / dead sensor) | no | `NaN` |
| `STUCK_HIGH` | echo already high at trigger time | no | `NaN` |
| `UNASSIGNED` | no pins configured for this sensor | no | `NaN` |

Distance: `d_mm = echo_us × 343 / 2000` (speed of sound fixed at 343 m/s, ≈20 °C; no
temperature compensation in v1).

### `IMU` fields

- `src`: `MPU6050` | `ITG3200`.
- accel fields are `-` for `ITG3200` (gyro-only part).
- `status`: `OK` | `READ_FAIL` | `NOT_FOUND`. When not `OK`, all data fields are `-`.
  Both IMUs report every `IMU_PERIOD_MS` even when `NOT_FOUND`, so a missing chip is visible rather than silent.
  After `READ_FAIL` the firmware keeps retrying reads; it does not re-initialise the chip (v1).
- `t_us`: MCU time at the start of the I²C burst read.

### `I2C` idents

`MPU6050`, `MPU6500`, `MPU9250`, `ITG3200`, `ADXL345`, `HMC5883L`, `QMC5883L`, `BMP180`, `UNKNOWN`.
`addr` is two uppercase hex digits (e.g. `68`).

## Safety rules

Enforced by `SafetySupervisor` in the firmware. The host bridge mirrors some of them but is never relied on.

1. **Boot state is `DISARMED`**; outputs are *safe*.
2. **Safe output** = throttle 0, brake `SAFE_STOP_BRAKE_PM`, steering 0.
3. `ARM` is accepted only if: state is `DISARMED`, e-stop input inactive, e-stop pin assigned
   (or host-sim build), not latched. Otherwise `ACK ERR` with the reason. Accepting `ARM`
   clears sticky faults and starts the command timer **as if a command had just arrived**.
4. While `ARMED`, a `CMD` with every field inside its wire range is **clamped** to
   `MAX_THROTTLE_PM`, `MAX_BRAKE_PM`, `±MAX_STEER_PM`. If brake > 0, throttle is forced to 0.
5. A `CMD` with any field outside its wire range (e.g. throttle 1001, steer −1001) while
   `ARMED` → safe stop, `DISARMED`, `INVALID_CMD`. While not armed → `ACK ERR NOT_ARMED`, ignored.
6. **Command expiry**: while `ARMED`, if no accepted `CMD` for `CMD_TIMEOUT_MS` or longer (≥ 200 ms trips, 199 ms does not) →
   safe stop, `DISARMED`, `CMD_TIMEOUT`. A disconnected Jetson is indistinguishable from this and is handled the same way.
7. **Bad frame** while `ARMED` → safe stop, `DISARMED`, `BAD_FRAME`. (A corrupted frame might have been a stop request.)
8. **E-stop input** is active-low with pull-up, so an unplugged/broken wire reads as *active*.
   Any active sample, in any state → `ESTOP` (latched), safe output. No debounce on entry.
9. `ESTOP` is left only via `RESET`, and only while the input is inactive → `DISARMED` (never straight to `ARMED`).
10. The MCU main loop kicks the hardware watchdog once per complete iteration. On watchdog
    reset the firmware boots `DISARMED` (rule 1). Whether the hardware watchdog is actually
    running on the UNO Q core is **UNVERIFIED** and reported via `HW_WATCHDOG_UNVERIFIED` and `BOOT`.
11. With `ENABLE_ACTUATOR_OUTPUT 0` (default), PWM pins are never attached; actuators receive
    no signal at all regardless of state.

## Defaults (`config.h`)

| Name | Default | Notes |
|---|---|---|
| `ENABLE_ACTUATOR_OUTPUT` | `0` | Must be changed by hand to drive motors |
| `CMD_TIMEOUT_MS` | 200 | Host sends `CMD` at 50 Hz → 10 missed frames |
| `STATUS_PERIOD_MS` | 100 | |
| `MAX_THROTTLE_PM` | 250 | Conservative bring-up limit, not a tuned value |
| `MAX_BRAKE_PM` | 1000 | |
| `MAX_STEER_PM` | 1000 | |
| `SAFE_STOP_BRAKE_PM` | 1000 | Brake applied on safe stop; tune on hardware |
| `US_SLOT_MS` | 60 | One sensor per slot → each sensor ≈ 4.2 Hz |
| `US_ORDER` | FL, RR, FR, RL | Diagonal alternation to reduce cross-talk |
| `US_ECHO_START_TIMEOUT_US` | 5000 | |
| `US_ECHO_MAX_US` | 30000 | ≈ 5.1 m |
| `US_MIN_MM` / `US_MAX_MM` | 20 / 4000 | HC-SR04 nominal, UNVERIFIED for the actual sensor |
| `IMU_PERIOD_MS` | 10 | 100 Hz |

## Host-simulator-only messages

Accepted only by the host simulator build (`uno_q_sim`), rejected as `UNKNOWN_TYPE` by real firmware:

| Body | Effect |
|---|---|
| `SIM,US,<id>,<mm>` | Obstacle distance for sensor `id`. `0` = nothing in range (echo times out); `-1` = sensor disconnected (no echo). |
| `SIM,ESTOP,<0\|1>` | Physical e-stop input released / pressed |
| `SIM,IMUFAIL,<MPU6050\|ITG3200>,<0\|1>` | Make an IMU's I²C reads fail |

`SIM` messages are answered with `ACK` like any other host message (the second field is the
literal `0` as seq: `ACK,0,OK,-`).
