#!/usr/bin/env bash
set -euo pipefail
# Synthetic data only. Explicit allowlist avoids private phone/GPS/image capture.
ros2 bag record -o "${1:-/tmp/autonomy-synthetic-bag}" \
  /sensors/wheel/odom /sensors/imu /phone/gps \
  /sensors/ultrasonic/sensor_0 /sensors/ultrasonic/sensor_1 /sensors/ultrasonic/sensor_2 /sensors/ultrasonic/sensor_3 \
  /camera/camera/aligned_depth_to_color/image_raw /perception/scene
