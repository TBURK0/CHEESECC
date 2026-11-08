# Validation evidence and remaining acceptance

Development checks were performed on Windows Python 3.8.3 and Ubuntu 24.04 WSL
with ROS 2 Jazzy / Python 3.12.3. The camera is a D435; the Jetson model and
JetPack/Ubuntu version are pending. Gemini is the intended independent reviewer.

## Executed

- `python -m unittest discover -s tests -v`: 22 tests passed. Includes fail-closed
  gates, per-source freshness, clock rollback, sequence replay rejection, GPS
  accuracy validation, WGS84 transforms, motion frame validation, stop dwell,
  ambiguous traffic lights, path planning and six deterministic replay scenarios.
- `colcon build --symlink-install`: all six packages built, including generated
  C/C++/Python message bindings. A second clean build on temporary Linux storage
  also built all six packages.
- Portable replay: clear, stop-sign and traffic-light missions reached x=3.856 m
  for a 4 m goal. Obstacle and lost-localization fixtures stopped at x=0.600 m;
  stale-depth fixture stopped at x=0.690 m. Straight cross-track RMS is zero by
  construction and is not a curved-course accuracy claim.
- `python3 tests/ros_bag_smoke.py --output FRESH_DIRECTORY`: actual synthetic bag
  recording/replay passed, with 97 local EKF pose samples spanning 5.9999 s,
  monotonic timestamps and maximum stationary drift 0.0 m. Replay ran in
  physical-disabled mode with synthetic mounts and ROS simulation time.

The installed Jazzy environment lacked robot_localization and sudo access.
For validation only, official Ubuntu/ROS binary packages were downloaded with
`apt-get download` and extracted to a user-owned temporary runtime prefix:
robot_localization 3.8.3, geographic_msgs 1.0.6, diagnostic_updater 4.2.7 and
GeographicLib 2.3. No system installation or firmware changes were made. The
documented setup script instead installs dependencies normally on a fresh host.

## ROS scenario checks

All eight actual ROS smoke cases passed: default disarmed, armed clear route,
obstacle, stale sensor, lost localization, perception failure, stop sign and
traffic light. Armed clear run reached x=2.371 m in the short observation window;
maximum proposed speed was 0.3 m/s. Per-case proposal-age p95 ranged from about
23.4 to 45.4 ms in this WSL run; these are not Jetson timing guarantees. Summaries
are included under `simulation/evidence/`. Some startup scheduling gaps correctly
produced temporary fail-closed stops, and loss of raw sensors correctly stopped
motion even when the EKF continued predicting.

`tests/ros_smoke.py` starts real ROS processes, subscribes to EKF/behavior/safety
output and explicitly arms simulation where enabled. Default launch is checked
for zero motion. Fault fixtures must have zero desired speed in the observation
tail. Local EKF poses must be finite, laterally stable and time-monotonic. The
traffic fixture must stop for red/unknown and release after green. The stop-sign
fixture must enter STOP_DWELL. These are smoke tests, not timing certification.

Startup testing found and corrected invalid numeric ROS topic tokens, integer
assignment to ROS float fields, shutdown handling, insufficient subscriber
discovery time for bag replay, and local EKF position uncertainty growth from a
velocity-only configuration. Sensor-specific safety reasons aid diagnosis.
The local EKF now fuses wheel-integrated x/y pose, not correlated wheel velocities
as additional independent measurements. Physical wheel covariance must reflect
slip/drift and cannot use the ideal simulation covariance without evaluation.

## Outstanding hardware and model acceptance

- D435 aligned RGB/depth/calibration acquisition on the actual Jetson, including
  reviewed wrapper/SDK versions, USB bandwidth and calibrated extrinsics.
- Pretrained weights provenance/licensing, target inference dependencies,
  stop-sign accuracy, traffic-light crop classification and unknown-state rates.
  No pretrained weights or target inference benchmarks were executed here.
- A learned traffic-color classifier/optional segmentation module and staged
  course evaluation; current color/path perception uses classical baselines.
- Claude telemetry adapter agreement, clock offset/skew budgets, authenticated
  transport, mount transforms, UNO range/IMU representation and wheel-odometry
  availability. Live phone JSON is deliberately rejected in this branch.
- Curved-course tracking, swept-footprint collision checking, route geofencing,
  range/FOV calibration, noisy sensor tests and measured braking envelope.
- Independent physical emergency-stop validation and separately reviewed actuator
  gateway. Physical desired movement remains unconditionally unsupported here.

The PR is an integration/review foundation. Passing these software checks does
not fulfill target D435/Jetson deployment acceptance or authorize physical motion.
