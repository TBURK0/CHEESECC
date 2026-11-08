# Hardware integration: sensors, actuators, safety

Owner: hardware subsystem (`claude/hardware-sensors`).
Out of scope here (owned by the autonomy team): Intel RealSense on the Jetson, iPhone GPS.

> **Read this first.** Almost nothing about the physical car has been measured yet. Every
> part model, pin, voltage and calibration value below is either **VERIFIED** (checked on the
> car, with date and who) or **UNVERIFIED / UNKNOWN**. As of this branch, **nothing is VERIFIED**.
> The code is built and tested only against the software simulator.
> Use the [hardware verification checklist](#hardware-verification-checklist) to change that.

## System overview

```
                    ┌──────────────────────────── NVIDIA Jetson (variant UNKNOWN) ─────────────────────────────┐
 RealSense ──USB3──►│ autonomy stack (not this branch)                                                          │
 iPhone GPS ───────►│      │ vehicle/cmd (ActuatorCommand)           ▲ ultrasonic/*, imu/*, vehicle/status     │
                    │      ▼                                          │                                         │
                    │  vehicle_bridge_node ──uno_q/frames──► sensor_bridge_node                                 │
                    └──────┬────────────────────────────────────────────────────────────────────────────────────┘
                           │ USB
                    ┌──────▼──────┐  3.3 V TTL UART, 460800 8N1, docs/serial_protocol.md
                    │ USB-UART    │──────────────────────────────┐
                    │ adapter     │  (common GND)                 │
                    └─────────────┘                               ▼
 ┌──────────────────────── Arduino UNO Q ─────────────────────────────────────────────────────────────┐
 │  Qualcomm MPU (Linux) — not used in the control path in v1                                          │
 │  STM32 MCU  ── firmware/arduino_uno_q/uno_q_vehicle  (SAFETY AUTHORITY: limits, expiry, e-stop)     │
 │     │ GPIO trig/echo ×4      │ I²C (SDA/SCL)            │ PWM ×3                 │ GPIO in (pull-up)  │
 └─────┼────────────────────────┼──────────────────────────┼────────────────────────┼────────────────────┘
       │ (level shift echo      │                          │                        │
       │  if sensor is 5 V)     │                          │                        │
   4 × ultrasonic          MPU-6050, ITG-3200,        throttle (ESC),          E-STOP switch
   FL FR RL RR             other 3-axis sensors       brake, steering          (normally closed to GND)
```

Why the MCU, not the Jetson or the UNO Q's Linux side, owns safety: it's the smallest piece of
software in the loop, it has no OS to hang, and it is the last thing before the motors. If the
Jetson, the USB link, or the bridge crashes, the MCU stops the car on its own within
`CMD_TIMEOUT_MS` (200 ms).

## Sensor and device inventory

| Device | Model | Interface | Supply / logic | Pins needed | Status |
|---|---|---|---|---|---|
| Controller | Arduino UNO Q: Qualcomm QRB2210 MPU + STM32U585 MCU | — | MCU I/O believed **3.3 V** | — | Model per project brief; I/O voltage and 5 V tolerance **UNVERIFIED** |
| Main computer | NVIDIA Jetson, **variant UNKNOWN** | USB to the UART adapter | — | 1 USB port | UNKNOWN. Run `cat /etc/nv_tegra_release`. JetPack 6 = Ubuntu 22.04 → ROS 2 **Humble**; Ubuntu 24.04-based JetPack → ROS 2 **Jazzy**. Code targets both. |
| Jetson ↔ MCU link | 3.3 V USB-UART adapter, **model UNKNOWN** (e.g. CP2102 / FT232 / CH340 class) | UART TX/RX/GND | 3.3 V logic | 2 MCU pins + GND | UNVERIFIED (see `serial_protocol.md` → Physical link) |
| Ultrasonic ×4 | **Assumed HC-SR04-style** trigger/echo | GPIO trigger out, echo in (interrupt) | HC-SR04: **5 V supply, 5 V echo output**. HC-SR04P / RCWL-1601 variants run at 3.3 V. | 8 GPIO (4 echo pins must support interrupts) | Model **UNVERIFIED**. If 5 V parts: echo **must** be level-shifted to 3.3 V. |
| IMU (accel + gyro) | **Assumed MPU-6050** (e.g. GY-521 board) | I²C, addr `0x68` (AD0 low) / `0x69` | Chip 2.375–3.46 V; GY-521 boards have an LDO and accept 3.3–5 V on VCC | shared I²C | Model **UNVERIFIED**; confirm with `SCAN` |
| Gyro | **Assumed ITG-3200 / ITG-3205** | I²C, addr `0x69` (AD0 high, common on breakouts) / `0x68` | Chip 2.1–3.6 V: **3.3 V only** unless the board has a regulator | shared I²C | Model **UNVERIFIED**; confirm with `SCAN` |
| "Additional 3-axis sensors" | **UNKNOWN** | I²C (assumed) | ? | shared I²C | Identify with `SCAN` (below). If the gyro is on a **GY-85** board, expect ADXL345 (accel, `0x53`) + HMC5883L/QMC5883L (magnetometer, `0x1E`/`0x0D`). On a GY-87, expect MPU-6050 + HMC5883L + BMP180 (`0x77`). v1 identifies these but doesn't drive them. |
| Throttle (gas) | **UNKNOWN**, assumed RC ESC, 50 Hz PWM | PWM 1000–2000 µs | ESC BEC may output 5–6 V on the red wire: **do not connect it to the UNO Q** | 1 PWM pin + GND | "Already wired": the signal type must be confirmed |
| Brake | **UNKNOWN**: separate servo, or the ESC's own brake on the throttle channel | PWM (assumed) | — | 1 PWM pin + GND | If braking is done by the ESC on the throttle channel, the 3-channel model needs changing: see *Remaining integration requirements* |
| Steering | **UNKNOWN**, assumed RC servo | PWM 1000–2000 µs | servo power from BEC/battery, not the UNO Q | 1 PWM pin + GND | UNVERIFIED |
| Emergency stop | **Not known to exist yet**: needs a normally-closed switch to GND | GPIO input, internal pull-up, active-low | 3.3 V | 1 GPIO | **Required before arming is possible** (firmware refuses `ARM` without it) |
| RealSense | Intel RealSense (model per autonomy team) | USB3 to Jetson | — | — | Not this subsystem |
| GPS | iPhone (autonomy team) | — | — | — | Not this subsystem |

### Duplicate sensing

- **Two gyros**: the MPU-6050 and the ITG-3200 both measure 3-axis angular rate. Both are
  published (`imu/mpu6050`, `imu/itg3200`). Recommendation: choose one as primary for the
  autonomy filter and use the other as a cross-check (disagreement beyond a threshold → fault).
  The choice is the autonomy team's; neither has been characterised on this car.
- **Possibly two accelerometers** (MPU-6050 + an ADXL345 if a GY-85 is present).
- **I²C address conflict risk**: both the MPU-6050 and the ITG-3200 can sit at `0x68`. They must be
  strapped to different addresses (usually MPU `0x68`, ITG `0x69`). `SCAN` shows what is
  actually on the bus; `config.h` holds the expected addresses.

### Identifying the unknown sensors

With the firmware running (outputs disabled is fine):

```bash
python3 tools/hil/hil_check.py --port /dev/ttyUSB0 scan
```

This prints every responding I²C address and the chip identified from its ID register
(`MPU6050`, `MPU6500`, `MPU9250`, `ITG3200`, `ADXL345`, `HMC5883L`, `QMC5883L`, `BMP180`, or `UNKNOWN`).
Record the result in the checklist. Cross-check against the markings on the boards.

## Pin assignment

**No pin numbers are assigned yet**: the real wiring hasn't been documented, and guessing would be
dangerous. Every pin in `firmware/arduino_uno_q/uno_q_vehicle/config.h` defaults to
`PIN_UNASSIGNED`. With a pin unassigned:

- that ultrasonic reports `UNASSIGNED` (invalid);
- that PWM channel is never attached (no signal);
- if it's the **e-stop** pin, `ARM` is refused (`ESTOP_PIN_UNASSIGNED`).

Fill this table from the real harness, then copy the numbers into `config.h`:

| Signal | Direction (MCU) | Needs | UNO Q pin | Verified |
|---|---|---|---|---|
| `US_TRIG_FL/FR/RL/RR` | out | any GPIO | TBD | ☐ |
| `US_ECHO_FL/FR/RL/RR` | in | interrupt-capable GPIO; **3.3 V level** | TBD | ☐ |
| `I2C SDA / SCL` | bidir | the MCU's I²C header pins (pull-ups to **3.3 V**) | TBD | ☐ |
| `PWM_THROTTLE` | out | PWM-capable (Servo library) | TBD | ☐ |
| `PWM_BRAKE` | out | PWM-capable | TBD | ☐ |
| `PWM_STEERING` | out | PWM-capable | TBD | ☐ |
| `ESTOP_IN` | in, pull-up | any GPIO; switch NC to GND | TBD | ☐ |
| `LINK TX / RX` | out / in | the MCU UART behind `LINK_SERIAL` | TBD | ☐ |
| `GND` | — | common between MCU, sensors, ESC/servo signal ground, USB-UART adapter | — | ☐ |

Wiring pattern (per signal type, not actual pins):

```
HC-SR04 (5 V part)                     UNO Q (3.3 V MCU)
  VCC ── 5 V supply
  TRIG ◄──────────────────────────────  US_TRIG_x   (3.3 V high is normally enough for HC-SR04 — VERIFY)
  ECHO ──[ level shifter or divider ]─► US_ECHO_x   (e.g. 1 kΩ series / 2 kΩ to GND ≈ 3.3 V — VERIFY)
  GND  ── common GND

E-STOP (normally-closed)               UNO Q
  one side ─────────────────────────── ESTOP_IN  (INPUT_PULLUP)
  other side ───────────────────────── GND
  closed = run, open/broken wire = STOP

ESC / servo                            UNO Q
  signal ◄──────────────────────────── PWM_x
  GND ──────────────────────────────── GND
  +V (BEC)  ── NOT connected to the UNO Q
```

## Axis conventions

- **Vehicle frame `base_link`** (REP-103): **x forward, y left, z up**. Positive yaw rate is a left turn.
  Positive `steering` in `ActuatorCommand` is **left**.
- **On the wire** (`IMU` frames), axes are the **chip's own axes** as printed in its datasheet.
  The MCU does not rotate anything.
- **In ROS**, `sensor_bridge` applies a per-IMU `axis_map`: a 3×3 signed permutation
  (9 ints, row-major) from chip axes to `base_link`. It must be a proper rotation (entries in
  {−1, 0, 1}, one non-zero per row/column, determinant +1); invalid maps are rejected at startup.
  **Default: identity (UNVERIFIED)**, i.e. it assumes the chip's x points forward and z points up.
- **Ultrasonic frames** `us_fl`, `us_fr`, `us_rl`, `us_rr`: the static transforms from `base_link`
  (position + outward angle) are **UNKNOWN** and must be measured on the car and published by the
  autonomy team's URDF / static TF.

How to determine each `axis_map` on the car (2 minutes, no tools):

1. Car level and still: whichever chip axis reads ≈ **+9.8 m/s²** is **up** (`+z`).
   (Accelerometers read +g along the axis pointing *away* from the ground.)
2. Lift the nose: the axis whose reading becomes **negative** in proportion is forward (`+x`).
3. `+y` = `z × x` (right-hand rule), which the determinant check enforces.
4. Gyro-only ITG-3200: rotate the car left (counter-clockwise from above). The gyro axis showing a
   **positive** rate is `+z`. Roll right side down → `+x` positive.

## Timestamps

- Every `US`/`IMU`/`STA` frame carries MCU `micros()` (`t_us`, uint32, wraps every ~71.6 min).
- `vehicle_bridge` stamps each received frame with host receive time and forwards it on `uno_q/frames`.
- `sensor_bridge` unwraps `t_us` and estimates `offset = host_rx_time − mcu_time` as the
  **minimum over a sliding window** (the sample with the least transport delay), then stamps
  messages `mcu_time + offset`. Assumption: clock drift is small over the window. Not yet
  measured on hardware.
- The ultrasonic stamp is the **trigger** time; the IMU stamp is the **start of the I²C burst read**.

## Safety design

Layers, from the motors outward. Each one stops the car without needing the layers above it.

| # | Layer | Where | Triggers safe stop on | Status |
|---|---|---|---|---|
| 0 | Physical power kill switch for the drive battery | wiring | human | **Recommended, not known to exist** |
| 1 | ESC signal-loss failsafe | ESC | PWM signal disappears (MCU dead/reset) | **UNVERIFIED**: many ESCs go neutral on signal loss, some hold last value. Test it. |
| 2 | Hardware watchdog | STM32 | MCU main loop hangs → reset → boots `DISARMED` | **UNVERIFIED**: the hook exists, but the UNO Q core's watchdog API is not confirmed; flagged via `HW_WATCHDOG_UNVERIFIED` |
| 3 | `SafetySupervisor` | MCU firmware | e-stop input (or broken wire), `ESTOP` msg, command older than 200 ms, out-of-range command, corrupted frame | Unit-tested + sim-tested |
| 4 | Limits | MCU firmware | every command is clamped to `MAX_*`; brake > 0 forces throttle 0 | Unit-tested (incl. randomized sweep) |
| 5 | `vehicle_bridge` | Jetson | upstream command stale (>0.2 s) → `DISARM`; MCU telemetry lost (>0.5 s) → `DISARM`, no auto re-arm | Unit-tested + sim-tested |
| 6 | Defaults | build + config | `ENABLE_ACTUATOR_OUTPUT 0` (PWM never attached), `allow_arm: false`, arming needs an explicit service call | — |

Safe stop = throttle 0, brake `SAFE_STOP_BRAKE_PM` (default full), steering centred.

**No autonomous physical movement is possible with this branch as checked in.** To drive motors,
someone must (1) set real pins, (2) set `ENABLE_ACTUATOR_OUTPUT 1` and re-flash, (3) wire an e-stop,
(4) launch with `allow_arm:=true`, and (5) call `vehicle/arm`. Do steps 2–5 only after the
checklist's actuator section passes **with the wheels off the ground**.

## Setup

### Without hardware (any Ubuntu 22.04/24.04 machine or WSL2)

```bash
# 1. Firmware logic: build the host simulator + unit tests
cmake -S firmware/arduino_uno_q/host -B /tmp/fwbuild && cmake --build /tmp/fwbuild -j
/tmp/fwbuild/test_core

# 2. ROS packages
source /opt/ros/jazzy/setup.bash            # or humble
cd ros2_ws && colcon build && source install/setup.bash

# 3. All ROS tests, including end-to-end against the simulated UNO Q
UNO_Q_SIM=/tmp/fwbuild/uno_q_sim colcon test && colcon test-result --verbose

# 4. Run the whole stack against the simulator
ros2 launch vehicle_bridge bringup.launch.py port:=sim:/tmp/fwbuild/uno_q_sim
ros2 topic echo /vehicle/status
```

### Flashing the UNO Q MCU

The sketch is `firmware/arduino_uno_q/uno_q_vehicle/`. Open it in Arduino App Lab / Arduino IDE
with the UNO Q board package installed, or use `arduino-cli`. **It has not been compiled for the
UNO Q target yet** (the board core isn't installed on the dev machine). Expect to adjust
`LINK_SERIAL` and possibly the Servo/interrupt includes for the UNO Q core. Flash it with
`ENABLE_ACTUATOR_OUTPUT 0` first.

### On the Jetson

1. Install ROS 2 matching the JetPack Ubuntu version (Humble on 22.04, Jazzy on 24.04).
2. `sudo apt install python3-serial ros-$ROS_DISTRO-diagnostic-msgs`, add your user to `dialout`.
3. Build `ros2_ws` as above. Find the adapter with `ls /dev/serial/by-id/` and use that path (stable across reboots).
4. `ros2 launch vehicle_bridge bringup.launch.py port:=/dev/serial/by-id/<adapter>`

## Hardware-in-the-loop tests

`tools/hil/hil_check.py` talks to the real MCU over the serial port, with no ROS needed. Each
subcommand prints PASS/FAIL per check and exits non-zero on failure. They also run against the
simulator: `--port sim:/tmp/fwbuild/uno_q_sim`.

| Subcommand | What it checks | Needs |
|---|---|---|
| `link` | BOOT/STA arrive, CRCs valid, bad-frame counter works, reported build flags | MCU only |
| `scan` | I²C scan + chip identification | IMU boards wired |
| `ultrasonic` | each of the 4 sensors reports, rates, status per sensor; prompts you to cover each sensor in turn | sensors wired |
| `imu` | both IMUs report at ~100 Hz, gravity magnitude ≈ 9.8 m/s² at rest, gyro noise at rest | IMU boards wired |
| `estop` | pressing the e-stop latches `ESTOP`; `RESET` refused while pressed; cleared after release | e-stop wired |
| `watchdog` | ARM, send CMDs, stop: `DISARMED` + `CMD_TIMEOUT` within 200 ms (±1 STA period); bad frame while armed → `BAD_FRAME`; out-of-range CMD → `INVALID_CMD` | MCU only; **refuses to run unless `--wheels-off-ground` is passed** when the build reports `OUTPUTS_ENABLED` |

## Hardware verification checklist

Tick each item with date + initials. Anything unticked stays UNVERIFIED in this document.

**Identification**
- [ ] Jetson model + JetPack version recorded: ______
- [ ] Ultrasonic model (marking on the board) recorded: ______; supply 3.3 V or 5 V: ______
- [ ] `hil_check.py scan` output recorded; MPU-6050 and ITG-3200 present at distinct addresses
- [ ] "Additional 3-axis sensors" identified: ______
- [ ] Throttle/brake/steering signal type confirmed (PWM? which channel does braking?): ______
- [ ] USB-UART adapter is 3.3 V logic (measured TX idle voltage with a meter): ______ V

**Electrical (power off unless stated)**
- [ ] Common ground between MCU, sensors, ESC/servo signal, UART adapter
- [ ] No 5 V signal reaches any UNO Q MCU pin (measure each ultrasonic echo at the MCU pin: ≤ 3.3 V)
- [ ] ESC/BEC +V is **not** connected to the UNO Q
- [ ] I²C pull-ups go to 3.3 V, not 5 V
- [ ] E-stop is normally-closed to GND on the configured pin; unplugging it reads as active
- [ ] Pin table above filled in and copied into `config.h`

**Firmware with outputs disabled** (`ENABLE_ACTUATOR_OUTPUT 0`)
- [ ] `hil_check.py link` passes; `BOOT` shows `OUTPUTS_DISABLED`
- [ ] `hil_check.py ultrasonic` passes: each sensor goes invalid/near when covered, and **only that sensor** changes
- [ ] Cross-talk check: all four sensors facing a wall at ~1 m, readings stable (record std dev): ______
- [ ] `hil_check.py imu` passes; gravity ≈ 9.8 m/s² on the expected axis
- [ ] `axis_map` for each IMU determined with the procedure above and set in `sensor_bridge.yaml`
- [ ] `hil_check.py estop` passes, including pulling the e-stop wire out
- [ ] `hil_check.py watchdog` passes
- [ ] Hardware watchdog: implement the UNO Q watchdog hook, then confirm that a deliberately hung loop resets the MCU and it boots `DISARMED`; set `HW_WATCHDOG_VERIFIED 1`

**Actuators: wheels OFF the ground, drive battery on a switch you can reach**
- [ ] PWM endpoints measured with a servo tester / scope and copied into `config.h` (neutral, max, brake release/full, steering centre/left/right)
- [ ] ESC signal-loss failsafe: unplug the throttle signal wire while spinning slowly → wheels stop: ______
- [ ] Re-flash with `ENABLE_ACTUATOR_OUTPUT 1`; `BOOT` shows `OUTPUTS_ENABLED`; car does **not** move at boot
- [ ] `hil_check.py watchdog --wheels-off-ground`: wheels stop within the timeout when commands stop
- [ ] Kill the bridge process (`kill -9`) while the wheels spin → wheels stop
- [ ] Unplug the USB-UART cable while the wheels spin → wheels stop
- [ ] Press the e-stop while the wheels spin → wheels stop; can't restart without `RESET` + `ARM`
- [ ] Throttle never exceeds `MAX_THROTTLE_PM` even when `vehicle/cmd` sends 1.0 (measure the pulse width)
- [ ] Steering direction: positive `steering` turns the wheels **left**
- [ ] `SAFE_STOP_BRAKE_PM` tuned (full brake may skid at speed): ______

**Only after all of the above:** a ground test at low `MAX_THROTTLE_PM` in a closed area, with a person on the e-stop.

## Remaining integration requirements

1. **Real pin numbers and wiring**: fill in the pin table. Nothing can be tested physically until this is done.
2. **E-stop hardware**: a normally-closed switch to an MCU GPIO; ideally also a drive-power kill switch (layer 0).
3. **Confirm the link path**: which UNO Q header UART is the MCU's, and the matching `LINK_SERIAL`; or decide to route via the UNO Q's Linux side instead (not implemented).
4. **Compile and flash for the UNO Q core**: verify that `Servo`, `attachInterrupt`, `Wire` and the `Serial*` naming work, and implement the hardware watchdog hook.
5. **Actuator signal model**: confirm 3 independent PWM channels. If braking goes through the ESC's throttle channel, or the car needs reverse, the actuator mapping and protocol (v1 is forward-only) must change.
6. **Sensor models**: confirm the ultrasonic part (and the 5 V vs 3.3 V level shifting), the IMU boards and the extra 3-axis sensors; add drivers for any that are needed (e.g. a magnetometer).
7. **Mounting geometry**: ultrasonic positions/angles and IMU orientation → `axis_map` + static TF (with the autonomy team).
8. **Jetson variant / ROS distro**: confirm, then build on the Jetson.
9. **Primary-gyro choice and fusion**: autonomy team.
10. **Performance numbers** (ultrasonic rate/noise, IMU noise, link latency, clock-sync accuracy) have not been measured. The defaults are datasheet-nominal or conservative guesses.
