# Sensor fusion framework

A starting point for fusing the MPU gyro/accel, the extra 3-axis
accelerometers, the ultrasonic sensors and the close-range RGB-D camera on
the RC car. It is meant to be dropped into the existing project repo and
filled in **one increment at a time, each verified on the real car**.

What works today:

- **Step 0 tooling**: message types, run logger, log replay, log summary,
  log plotting, a replay harness for running fusion layers offline.
- The **layer framework**: a `FusionNode` that routes messages to whichever
  layers you have implemented so far.
- A **config guard**: every value that has to be measured on the car starts
  as `null`, and a layer refuses to start until its values are filled in.

What is a skeleton on purpose: the attitude filter, accelerometer array,
pose EKF, obstacle layer and safety monitor. Each file explains the
recommended method and lists its increments with pass checks, mapped to the
test steps below.

## Layout

```
config/fusion.yaml        every parameter; MEASURE = filled in by a test step
fusion/messages.py        timestamped message types (the hardware contract)
fusion/io/logger.py       RunLogger: one gzipped JSON-lines file per run
fusion/io/sources.py      ReplaySource (logs) / LiveSource (your hardware: TODO)
fusion/node.py            FusionNode + Layer base class
fusion/calibration.py     mount rotation, accel calibration (6-position fit: TODO)
fusion/layers/            attitude, accel_array, pose_ekf, obstacles (skeletons)
fusion/safety.py          stopping-distance math + host safety monitor (TODO)
tools/                    check_config, log_summary, plot_log, replay_fusion, make_demo_log
tests/                    tests for everything implemented
```

## Getting started

```bash
pip install -r requirements.txt
python -m pytest tests/                      # all should pass
python -m tools.make_demo_log logs/          # synthetic log, to try the tools
python -m tools.log_summary logs/demo_*.jsonl.gz
python -m tools.plot_log logs/demo_*.jsonl.gz -o demo.png
python -m tools.check_config                 # what still needs measuring
```

## Integrating with the existing repo

The only file that touches your hardware code is `fusion/io/sources.py`
(`LiveSource`). Keep that adapter thin: convert units and timestamps into
the types in `fusion/messages.py`, nothing else. Fusion never imports your
hardware modules, so the same code runs live and on replayed logs.

If your microcontroller already streams sensor data, the adapter is mostly
parsing. If it doesn't stamp readings with its own clock yet, add that to the
firmware first: capture-time stamps from one clock are the foundation for
everything after Step 0.

## Increments

Each row is something you can finish and verify in one or a few sessions.
Work top to bottom; later rows depend on numbers measured earlier.

| Step | Increment | Code to write | Pass check on the car |
|---|---|---|---|
| 0 | Logging & replay | `LiveSource` adapter, logging call in your main loop | Hand-driven run; `log_summary` shows expected rates, no backwards timestamps; `plot_log` shows all streams |
| 1 | Sensor characterization | `fit_six_position`; small scripts for still-data stats | Config filled for noise, bias drift, ranges, beam width |
| 2 | Timing | `time_offset_s`, `vo_latency_s` into config | Tap test: IMU and camera spikes line up within ~10 ms after correction |
| 3 | Attitude | `AttitudeLayer` (3a, 3b, 3c in the file) | Roll/pitch within ~1–2° on the bench and with the motor running |
| 4 | Speed & steering | encoder scale, steering map into config | Odometry error < ~3% over a measured straight; measured steering curve |
| 5 | VO alone | your VO pipeline emitting `VoPose` | Known drift per meter; list of situations where VO fails |
| 6 | Pose EKF | `PoseEKF` (6a–6e in the file) | Hand-driven loop closes near the start; covering the lens doesn't cause jumps |
| 7 | Safety | MCU e-stop/watchdog/kill switch first, then `SafetyMonitor` | Car always stops with margin, including with the host unplugged |
| 8 | Control | your speed loop and path tracker using `node.state` | Follows a taped course at target speed |
| 9 | Obstacles | `ObstacleLayer` | Ignores posts beside the path, stops for posts on it, sees glass via ultrasonics |

## Working rules

- **Log every test.** Use `Event` messages ("tap", "start loop", "lens
  covered") to mark moments you'll want to find later.
- **Develop on replayed logs, verify live.** `tools/replay_fusion.py` runs any
  set of layers over a log and writes CSV, so you can tune without the car.
- **Keep failing logs.** Short logs of failures become regression tests:
  replay them in `tests/` after every change.
- **One change per test session**, written down with the log file name.
- **Never guess a config value.** If a layer needs a number, it comes from a
  measurement and the test log says which one.
