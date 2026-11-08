# RC dashboard signal contract

The phone dashboard reads **one JSON object per frame, newline-terminated**, at ~10 Hz.
Same dashboard interface as before, RC signals. `null` = not available yet — render it
**blank**, not 0 (a null speedo must not look like "stopped").

## Signals

| Field | Unit / range | Source | Status |
|-------|--------------|--------|--------|
| `ts` | epoch seconds | sender clock | live |
| `schema` | int (=1) | — | live |
| `state` | DISARMED \| ARMED \| ESTOP \| UNKNOWN | VehicleStatus | live |
| `armed` | bool | VehicleStatus | live |
| `link_ok` | bool | VehicleStatus | live |
| `throttle` | 0..1 (driven) | VehicleStatus | live |
| `brake` | 0..1 (driven) | VehicleStatus | live |
| `steering` | −1..1, + = left | VehicleStatus | live |
| `faults` | list of strings | VehicleStatus | live |
| `firmware` | string | VehicleStatus | live |
| `outputs_enabled` | bool | VehicleStatus | live |
| `hw_watchdog_verified` | bool | VehicleStatus | live |
| `us_fl` `us_fr` `us_rl` `us_rr` | meters, ≥0 (null if invalid) | sensor_bridge Range | live |
| `yaw_rate` | rad/s | IMU gz | live |
| `heading_deg` | degrees | fusion | **null — TODO (needs odometry/fusion)** |
| `speed_mps` | m/s | wheel encoder | **null — TODO (no encoder yet)** |
| `battery_v` / `battery_pct` | volts / 0..100 | battery ADC | **null — TODO (no ADC wired)** |
| `detections` | int | perception | **null — TODO (no model wired)** |

Live fields carry real data today. The `null` fields are wired into the contract so the
dashboard layout is final, but they stay `null` until that sensor/model exists — nothing is
faked.

## Example frame

```json
{"ts":1794130000.12,"schema":1,"state":"ARMED","armed":true,"link_ok":true,
 "throttle":0.20,"brake":0.0,"steering":-0.15,"faults":[],"firmware":"0.1.0",
 "outputs_enabled":false,"hw_watchdog_verified":false,
 "us_fl":1.23,"us_fr":2.10,"us_rl":null,"us_rr":0.80,
 "yaw_rate":0.02,"heading_deg":null,"speed_mps":null,
 "battery_v":null,"battery_pct":null,"detections":null}
```

## Transport (BLE / serial)

The publisher writes each frame as a line to a transport:

- **BLE-UART module (recommended for the phone):** connect an HM-10 / nRF52 / ESP32 BLE-UART
  module to the compute box's serial port. The phone connects to it over BLE and receives the
  JSON lines unchanged — no GATT server to write. Run with `transport:=serial port:=<dev>`.
- **USB serial:** same thing over a USB cable (Android/desktop).
- **Native BLE GATT peripheral** (no module): run a BlueZ GATT server on the Jetson exposing a
  Nordic-UART-style notify characteristic and push each frame as a notification. Not included
  here — the module path is far less fragile.

## Run

```bash
cd ros2_ws && colcon build && source install/setup.bash
# bench, no phone: prints frames to the console
ros2 run rc_telemetry rc_telemetry
# to the phone over a BLE-UART module / USB serial:
ros2 run rc_telemetry rc_telemetry --ros-args -p transport:=serial -p port:=/dev/ttyACM0
```

Run it alongside `vehicle_bridge` + `sensor_bridge` (with the sim or the real car) and the
frames fill in with live state.
