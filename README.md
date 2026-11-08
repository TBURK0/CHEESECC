# CHEESECC

MAKEUC 26 submission. CHEESECC is an on-vehicle autonomy stack with an iPhone as the rider interface.

The phone picks a destination and rides along with GPS. A vehicle computer turns the forward camera, depth, ultrasonics, and inertial sensors into a path and an actuator request. A DAVE-2 style network maps one camera frame to a steering-wheel angle. A small microcontroller is the last safety authority before the steering, throttle, and brake. If that gateway goes quiet, the vehicle computer cannot keep the car moving.

The phone is Swift and MapKit. The classical autonomy layers are a NumPy stack that also runs without ROS. The vehicle computer is ROS 2. The gateway is an STM32 on a serial link, and it is the only component that drives the actuators.

## A drive, from destination to actuators

1. The rider types a stop on the phone. An empty start means the current GPS fix. MapKit returns a driving polyline.
2. The phone follows that line. About four times a second it solves a short lookahead: where the vehicle should be next, and the wheel angle and speed target that points there. The horizon is about 0.80 s when stopped and about 0.25 s at highway speed, so the step stays longer than GPS noise and shorter than a late correction.
3. The selected car changes the numbers. Wheelbase, track, mass, steering ratio, and tire radius come from the garage catalog. The same curvature is a different handwheel angle on a different car.
4. Each push is a `DriveCommand`: lookahead time, target latitude and longitude, speed, wheel angle, throttle, brake, curvature, and which car it belongs to. The phone shows the frame and stores it on device.
5. The same command is what `DriveBridge` in the Python stack accepts. It checks the numbers, and if they are not usable it replaces them with a braking stop. It then encodes steering, speed, lookahead, and the GPS fields as CAN frames in `stack/cheesestack/comms/data/cheesecc.dbc`.
6. On the vehicle computer, ROS 2 keeps a local occupancy map from depth and ultrasonics, searches a bicycle-lattice path, and runs pure pursuit. The command that leaves this pipeline is still a request.
7. The STM32 on the gateway board is the only thing that drives PWM. It clamps the request, drops it if the link goes stale (200 ms), and stops the vehicle on its own if the computer, the USB link, or the bridge dies. An e-stop input is required before the firmware will arm.

Positive steering is left on the phone, on the CAN database, and on the serial link.

## iPhone

Open `theapp/CHEESECC.xcodeproj` in Xcode 16 or newer. Bundle id is `com.makeuc.cheesecc`. Change it to the id registered on your Apple team before upload.

The app is a SwiftUI tab bar: Drive, Garage, Learn.

**Drive** is the route and the live follow. The top card is titled Real-Time Route Planning & Dynamic Lookahead. From and To search with MapKit. From stays empty to mean the current location. The route is drawn with a green start pin and a red destination pin. Go locks the camera heading-up on the phone. If the fix stays more than 60 m off the polyline, the route is recalculated to the same stop. Under the map, the deck shows speed, wheel angle, the lookahead horizon, the chassis, a handwheel slider, and the command bus. Dragging the wheel during a run logs those samples as `driver`. Planner samples are marked `planner`.

**Garage** is a catalog of published-spec cars in `theapp/CHEESECC/Data/Catalog/cars.json`. The default is a 2019 Hyundai Elantra, shown from `Elantra2019` in the asset catalog. Other cars use a chassis diagram scaled from length, width, and wheelbase. Those dimensions feed the bicycle model: steer gain is wheelbase times steering ratio in the small-angle form, and the speed target depends on mass, drag, and power.

**Learn** is titled On-Device Telemetry & Kinematic Learning. It plots handwheel angle against curvature for the selected car, fits a slope on device, lists drive sessions, and exports JSON. That fit is the measured gain. The catalog gain is the theoretical one. They are supposed to move toward each other as the log grows.

Location is when-in-use. The follow loop runs while the app is in the foreground. A simulated location or a GPX track in Xcode is read the same way as a live fix.

TestFlight is the install path for phones that are not plugged into Xcode. The workflow is `.github/workflows/testflight.yml`. It is manual (`workflow_dispatch`). On a Mac runner it selects Xcode 26, archives `theapp/CHEESECC.xcodeproj`, and uploads. The archive is built unsigned so the export step can sign for the App Store. App Store Connect currently rejects an iOS 18 SDK upload, which is why the runner is macOS 26. Secrets required on the GitHub repo: `APPLE_TEAM_ID`, `ASC_KEY_ID`, `ASC_ISSUER_ID`, and `ASC_PRIVATE_KEY` (the full `.p8`). Add internal testers after processing. External review is too slow for a hackathon.

## Vehicle computer

The vehicle side is ROS 2 under `ros2_ws`. The autonomy node is the one that should run. Older scaffold planners are not part of this loop.

```bash
ros2 launch autonomy local.launch.py
```

Inputs it expects:

| Topic | What it must be |
|---|---|
| `d405/depth/image_rect_raw` | Rectified depth, 16-bit millimetres or 32-bit metres. A bad patch in the center of the frame rejects the scene. |
| `d405/depth/camera_info` | Intrinsics for that same frame. |
| `ultrasonic/{fl,fr,rl,rr}` | Four ranges in a calibrated frame. A too-close reading brakes in every mode. |
| `odom` | Pose of the rear axle, orientation normalized, speed newer than 0.2 s. |
| `goal` | A pose in `base_link`, newer than 0.5 s. |
| `vehicle/status` | Gateway status, healthy and armed, newer than 0.3 s. |

Sensor readings are kept in the pose they were taken in and moved into the current rear-axle frame. If odometry or TF is missing for that stamp, the reading is dropped. Depth marks free space only along the ray from the camera to the hit. Occupied cells are added. Old free space is never treated as permission to move. Unknown cells block motion, including the cells under the vehicle footprint. An empty obstacle list is not a clear road.

The planner searches a bounded bicycle lattice and checks a padded footprint on every primitive. Goals behind the vehicle, unknown space, and a search that runs out of room all produce an empty path. Pure pursuit runs only when a path exists. Throttle out of this node is capped at 0.25. The stopping distance it protects is `0.25 + abs(speed)*0.3 + speed**2/(2*0.8)` metres. A fault, a stale input, a requested brake, a too-close ultrasonic, or a collision with unknown space becomes a full brake. The reason is published on `safety/reason`.

Wheel and IMU odometry is a separate node. It integrates measured wheel speed and yaw rate. It does not invent speed from throttle. A gap in time latches it off until it is restarted. It drifts. It is not a map fix.

Optional semantic perception runs YOLO on color aligned to depth, tracks boxes, and attaches a median depth. Those tracks are published beside the geometric map. They do not by themselves issue a traffic-light or lane command.

Commands can enter as an internal candidate, as JSON, or as an HMAC-signed packet. Authenticated mode wants `AUTONOMY_COMMAND_KEY` in the environment (at least 32 random bytes, hex), an operator name, a session id, a rising sequence, and a timestamp. Packets expire in 0.2 s. The key does not belong in YAML. HMAC checks identity and integrity. It does not encrypt the link, and a process that can publish `vehicle/cmd` directly still bypasses it. A deployed network needs DDS security on top.

Launch flags for mount calibration and vehicle geometry stay off until those lengths are measured: wheelbase, front and rear overhang, half-width, and steering angle at the road wheel, all relative to the rear axle. `config/autonomy.yaml` holds the assumed numbers only.

## Python stack

`stack/cheesestack` is the layer that can run without ROS. NumPy only. Layers do not import each other. Shared messages live in `cheesestack.core`. Integration for a phone command is `cheesestack.pipeline.DriveBridge`.

| Layer | Role |
|---|---|
| `core` | Poses, headers, the message bus, the node base. Planar frames are x forward, y left, yaw counterclockwise from +x. The map frame is a local east-north tangent plane. |
| `perception` | Camera ingest, sync, calibration, undistort, then scene understanding: detection, tracking, lanes, road edge, drivable space, signs, lights, an occupancy grid, and detection fusion. |
| `localization` | GPS into the tangent plane, IMU, wheel odometry, visual odometry, Kalman filters, heading, and frame transforms. |
| `mapping` | Lane graph, a route as a sequence of lanes, map matching, curvature, boundaries. |
| `prediction` | Constant-turn and lane-change models, time to collision, a risk score. |
| `planning` | Mission, behavior, gap selection, merge, overtake, a trajectory in Frenet coordinates. |
| `control` | Pure pursuit and Stanley-style path tracking, lateral LQR, longitudinal PID, rate limits, an oscillation check. |
| `vehicle` | The catalog parameters again, a bicycle model, slip, wheel-speed checks, diagnostics. |
| `safety` | Input checks, a watchdog, heartbeat, redundant voters, end-to-end counters, and a minimal-risk stop that ramps brake with a jerk limit. |
| `comms` | CAN encode and decode, the CHEESECC DBC, a virtual bus, ISO-TP, and a UDS server. |
| `sim` | Roads, a vehicle twin, sensor noise, scripted scenarios, closed-loop pure pursuit. |
| `ml` | Dataset manifests, augmentation, a small trainer, metrics, drift, and checkpoints as `.npz`. |

`DriveBridge` is the seam. Give it the phone's command dictionary and a timestamp. It builds a `ControlCommand`, runs `InputValidator`, and on a non-finite or rejected command steps `MinimalRiskManeuver` so the first published frame already asks for brake. A watchdog named `drive` has to be kicked on that same tick. The accepted command is encoded with `DriveCommandCodec` onto a virtual CAN bus: `SteeringCommand`, `SpeedCommand`, `DriveCommandMeta`, and multiplexed `DriveCommandGeo`. `set_route` takes the phone polyline as latitude and longitude, projects it with the same equirectangular meters the phone uses, and builds a one-lane `Route`. `progress` returns station, lateral offset, and metres left. `run_sim` runs a named scenario (`straight_follow`, `cut_in`, `hard_brake_lead`, `curve`, `pedestrian_crossing`) and drops the other actors into the ego occupancy grid.

```bash
cd stack
python -m pip install -e ".[dev]"
python -m pytest tests -q
```

Conventions, units, and the rule that layers stay independent are in `stack/CONVENTIONS.md`.

## Camera-to-steering network

`stack/models/steering_policy` is a PilotNet / DAVE-2 style network for one forward camera. The crop is 3×66×200 RGB in `[0, 1]`, hood and sky already removed. Five valid convolutions (24, 36, 48, 64, 64 filters) feed fully connected layers 100, 50, 10, and 1. That is 252,219 parameters. The output is steering-wheel angle in degrees.

`infer.py` loads `steering_policy.npz`, subtracts `normalization_stats.json`, and runs one built-in frame:

```bash
cd stack/models/steering_policy
python infer.py
```

It prints `steering_deg:`. The checked-in arrays are a seeded initialization (seed 26), so that number is the forward pass of this file, not a road test. `training_config.yaml`, `dataset_manifest.json`, `training_log.csv`, `metrics.json`, and `eval_report.md` are the training record for the campus set: 12,437 labeled frames kept from 6.2 hours, session split 9950 / 1244 / 1243, 50 epochs, loss from 0.86 to 0.10, steering error from 10.9° to 2.3°. The test writeup is worse on rain left turns (6.8°) and sun glare (5.6°) than on the rest of the test split (1.88°). `model_card.md` is the use and non-use statement. Do not put this file in the actuator path.

## Gateway and sensors

The computer speaks a text serial protocol to the STM32 on an Arduino UNO Q. The Qualcomm side of that board is not in the control path. Framing is `$<body>*<CRC>\n` with CRC-16/CCITT-FALSE. Numbers on the wire are integers. Throttle and brake are per-mille. Steering is per-mille from −1000 to 1000, positive left. Ranges are millimetres. Rates are microradians per second. The host sends requests. The MCU clamps them and is the component that times out.

Four ultrasonics sit at the corners. An IMU (and a second gyro, when it is on the bus) is published separately so the filter can cross-check them. Depth is an Intel RealSense into the Jetson. GPS for the route is the phone. Actuator outputs and arming stay disabled until the e-stop wiring and the measured geometry exist. The protocol, the pin map, and the checklist are in `docs/serial_protocol.md` and `docs/hardware.md`.

`tools/hil/hil_check.py` talks to the gateway on a serial port and can scan I²C for the IMU parts. `tools/vision/label_video.py` is the offline labeling tool for a recorded forward camera.

## Simulation

Two simulators cover the stack before a vehicle is in the loop.

The cheesestack sim is a 2D road, a kinematic twin, noisy GPS, IMU, wheel speed, and object lists, plus closed-loop pure pursuit. Scenarios include a slower lead, a cut-in, a hard brake, a curve, and a pedestrian crossing. `DriveBridge.run_sim` is the short entry.

The ROS sim is `python -m autonomy.simulation` and `python -m autonomy.physics_sim` with `PYTHONPATH=ros2_ws/src/autonomy`. Physics scenarios cover a goal, an obstacle that appears, and a sensor dropout. They use an assumed acceleration and brake model, not a measured one.

Host-side firmware tests live in `firmware/arduino_uno_q/host` and do not need the board.

## Layout

```
theapp/                  iOS app (Drive, Garage, Learn)
stack/cheesestack/       NumPy autonomy layers and DriveBridge
stack/models/steering_policy/   DAVE-2 style net, configs, infer.py
ros2_ws/                 ROS 2 autonomy, vehicle bridge, sensor bridge, telemetry
firmware/arduino_uno_q/  Gateway firmware and host tests
docs/                    Protocol, hardware, autonomy, labeling
tools/                   Hardware-in-the-loop check and video labeling
can_capture/             Bus logs and capture scripts
.github/workflows/       TestFlight upload and stack CI
```

## What is already checked, and what is not

Stack unit tests run on any machine with NumPy. ROS unit tests run without ROS. The ROS smoke test skips until a ROS install exists. Firmware logic tests run on the host.

Still required on a real vehicle: measured mounts and geometry, a confirmed e-stop, IMU axis checks, wheel-speed scale, braking distance, and a ROS run with the live depth camera and the four ultrasonics. Until those are done, actuator outputs stay off and the phone command remains a request. The stack is built so that a failed check becomes a stop, not a guess.
