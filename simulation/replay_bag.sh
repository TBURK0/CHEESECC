#!/usr/bin/env bash
set -euo pipefail
# Start autonomy.launch.py simulation:=false use_sim_time:=true synthetic_mounts:=true first.
# Physical motion is intentionally unsupported in this mode.
ros2 bag play "$1" --clock
