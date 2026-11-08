# Experimental autonomy

Implemented modules cover calibrated depth geometry, ultrasonic fusion, object detection/tracking, dead reckoning, occupancy history, car-feasible planning, pure pursuit, command safety, authenticated ingress and vehicle simulation. Physical-car validation remains outstanding. Mount/geometry calibration and firmware actuator outputs default disabled; no arming requests are issued.

For the standalone YOLO/OpenCV tool see [video_labeling.md](video_labeling.md).

## Hardware-free verification

From the repo root in PowerShell:

```powershell
$env:PYTHONPATH='ros2_ws/src/autonomy'
python -B -m unittest discover -s ros2_ws/src/autonomy/test -v
python -B -m autonomy.simulation
python -B -m autonomy.physics_sim --output trajectory.json
python -B -m unittest discover -s tools/vision -p test_label_video.py -v
```

The ROS smoke test explicitly skips without ROS. Tests cover transforms, depth buffer layouts, sensor union, ego-motion memory, map expiry, bicycle routes, swept footprint collision, tracking, authentication/replay and dead reckoning. Physics scenarios cover goal tracking, a newly appearing obstacle and sensor dropout using ideal known-map sensing and assumed acceleration/drag/braking. They are not a measured vehicle model.

## ROS setup

In a Linux ROS 2 environment install dependencies with rosdep, build `vehicle_interfaces` and `autonomy` using colcon, and source the workspace. Optional semantic perception also requires Ultralytics in the ROS Python environment.

```bash
ros2 launch autonomy local.launch.py
# Optional separate nodes:
ros2 run autonomy wheel_imu_odometry --ros-args -p imu_axes_verified:=true
ros2 run autonomy semantic_perception --ros-args -p model:=/path/to/yolo11n.pt
```

Run existing vehicle/sensor bridges separately. Do not run old perception/fusion/planning scaffold nodes alongside the pipeline. The legacy planner now publishes an empty route.

`config/autonomy.yaml` contains assumed geometry. Measure wheelbase, front/rear extents, half width and wheel steering angle relative to the rear-axle `base_link` before enabling. Launch arguments: `config`, `mount_calibrated`, `geometry_calibrated`, `camera_height_m`, `use_tf`, `require_ultrasonics`, `command_source`, `operator`. Calibration launch flags stay false even with a custom YAML; pass them explicitly after physical verification.

TF is enabled by default. Supply measured `base_link` transforms for depth optical and ultrasonic frames. Transforms are looked up at observation timestamps. `use_tf:=false` supports only a measured level forward camera mount using `camera_height_m` and optical z forward, x right, y down.

## Input contracts

| Topic | Type | Contract |
|---|---|---|
| `d405/depth/image_rect_raw` | sensor_msgs/Image | Rectified 16UC1 mm or 32FC1 m; invalid central depths reject scene |
| `d405/depth/camera_info` | sensor_msgs/CameraInfo | Depth-matching frame, dimensions and pinhole intrinsics |
| `ultrasonic/{fl,fr,rl,rr}` | sensor_msgs/Range | Calibrated TF; +x sensor ray; -inf always brakes |
| `odom` | nav_msgs/Odometry | `odom` to rear-axle `base_link`, normalized orientation, speed within 0.2 s |
| `goal` | geometry_msgs/PoseStamped | `base_link` goal refreshed within 0.5 s |
| `vehicle/status` | vehicle_interfaces/VehicleStatus | Healthy armed status within 0.3 s |
| `autonomy/cmd_candidate` | vehicle_interfaces/ActuatorCommand | Candidate mode only, expires in 0.2 s |
| `autonomy/cmd_json` | std_msgs/String | json/authenticated modes only, same expiry |

Sensor observations require odometry within 50 ms. The latest matching pose is used without interpolation. Observations are stored with their observation pose and compensated into current base coordinates. Missing matching odometry or TF rejects measurements. Ultrasound +inf supplies health but no occupied/free cells; NaN is invalid. When `require_ultrasonics:=true`, all four must be healthy within 0.3 s. Too-close -inf brakes in every mode.

## Outputs and mapping

| Topic | Contents |
|---|---|
| `mapping/local` | Conservative current OccupancyGrid in base_link |
| `mapping/odom` | Bounded occupancy history, 5 s TTL, for visualization |
| `fusion/obstacles` | Fresh geometric obstacle PoseArray |
| `scene/summary` | JSON freshness, obstacle count and clearance |
| `planning/path` | Bicycle-lattice Path with headings |
| `vehicle/cmd` | Gated ActuatorCommand |
| `safety/reason` | Current braking/gating reason |
| `perception/semantic` | Optional JSON classes, boxes, confidence, track IDs and depth-derived positions |

Depth rays clear only from the calibrated camera origin to measured obstacle endpoints. Height filtering retains 0.04-0.6 m obstacle returns. Fused/memorized points only add occupancy; old map free space never authorizes motion. Unknown cells, including beneath the padded vehicle footprint, block motion. A camera view alone may not provide sufficient coverage to move: do not fill unknown space with free values to bypass this. An empty obstacle array alone is not evidence of free space.

## Planning, control and safety

The bounded forward bicycle-lattice search checks an oriented padded vehicle footprint along every primitive. Unreachable/behind goals, unknown/out-of-bounds space and exhausted searches yield empty routes. `command_source:=controller` uses pure pursuit; it never arms the vehicle. External candidate modes also require a fresh feasible route and independently check the requested steering sweep.

Motion requires fresh scene, odometry, goal, status and command plus calibrated geometry. Faults, invalid values, requested braking, too-close ultrasound, stale inputs or unknown/collision space produce full braking. Throttle is capped at 0.25. The assumed stopping distance is `0.25 + abs(speed)*0.3 + speed**2/(2*0.8)` metres, with geometry included in the sweep. Tune margins and deceleration using measured stopping tests.

## Dead reckoning and semantics

`wheel_imu_odometry` consumes measured `wheel/velocity` (TwistStamped in base_link) and calibrated `imu/mpu6050` yaw rate within 50 ms. It publishes Odometry and odom-to-base_link TF. `imu_axes_verified` defaults false. Non-increasing time or a time gap latches the estimator off until restart/relocalization. No speed is invented from throttle. Use a single odometry/TF publisher. This estimator drifts and requires a real wheel velocity source; it is not SLAM or absolute localization.

`semantic_perception` approximately synchronizes RGB and **depth aligned to color**, with matching color intrinsics, calibrated TF and <=50 ms skew. It runs YOLO, class-aware IoU tracking and median box-depth localization. Missing valid depth gives null rather than guessed position. Tracks are heuristic identities. Semantics are published separately and do not issue traffic-rule commands. Signal-state recognition, lane/relevance association and traffic-rule behavior need separate validation before controlling the car.

## Authenticated command ingress

Choose one source at startup: `candidate`, `json`, `authenticated` or `controller`. Authenticated mode has no raw candidate subscriber. Set `AUTONOMY_COMMAND_KEY` in the process environment to at least 32 random bytes encoded in hex; never store it in YAML, ROS parameters or source. Launch with `command_source:=authenticated operator:=driver`. A fresh receiver session is logged at startup.

Packets require HMAC-SHA256, the configured operator, the receiver session, increasing sequence and a fresh timestamp. Rejections clear the active request. Receiver restarts invalidate prior sessions. The signer prints a packet without sending or arming:

```powershell
$env:PYTHONPATH='ros2_ws/src/autonomy'
python -m autonomy.comms --session RECEIVER_SESSION --operator driver --seq 1 --throttle 0.1
```

Use the same secret at sender/receiver and relay immediately: packets expire in 0.2 s. Default signing timestamps use wall time; pass `--stamp` for simulated ROS time. Payload fields: `seq`, `stamp`, `throttle`, `steering`, optional `brake`. There are no arming commands. Trusted local `json` mode uses unsigned payloads.

HMAC provides integrity/authentication, not encryption, traffic-rate limiting or graph access control. Network deployment requires a secure relay and DDS Security/graph ACLs. A process permitted to publish directly to `vehicle/cmd` can bypass the pipeline; floods can deny service. No network listener is opened here.

## Outstanding verification and mission-dependent work

- Run the ROS smoke test and real camera, TF, clock and topic integration on Linux/ROS.
- Install model dependencies and verify YOLO accuracy against representative recordings.
- Measure mounts, geometry, steering, wheel scale, IMU axes and actual braking.
- Finish physical HIL, e-stop/watchdog checks and firmware pin/actuator calibration.
- Add absolute localization/SLAM if required by mission distance and drift tolerance.
- Validate signal state, lane relevance, traffic rules and dynamic-agent prediction for the course.

Physical measurements and target-environment checks cannot be completed from this Windows workspace without the car, recordings and ROS runtime. Firmware output-disable and explicit bridge arming requirements remain in place.
