# Synthetic autonomy demo

Start with [the architecture and setup guide](../docs/autonomy_architecture.md).
`replay.py` runs six deterministic scenarios without ROS or third-party packages.
`ros_smoke.py` in tests launches actual nodes against installed ROS Jazzy and
robot_localization. RealSense hardware and model weights are not needed for these
synthetic tests. No Arduino devices or physical actuator outputs are used.

```bash
python3 simulation/replay.py --output /tmp/autonomy-replay
python3 -m unittest discover -s tests -v
```

Logs are JSON containing timestamps, behavior states, requested speeds and safety
decisions. Ideal straight-route replay has zero cross-track error by construction;
it does not measure turn tracking or physical localization accuracy. Default ROS
launch cannot move. Explicit simulation arming is documented in the main guide.
