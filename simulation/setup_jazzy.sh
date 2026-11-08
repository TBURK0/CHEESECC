#!/usr/bin/env bash
set -euo pipefail
# Ubuntu 24.04 development setup. Review before running; target JetPack may differ.
sudo apt-get update
sudo apt-get install -y ros-jazzy-ros-base python3-colcon-common-extensions \
  ros-jazzy-robot-localization ros-jazzy-rviz2 ros-jazzy-cv-bridge \
  ros-jazzy-rosbag2 ros-jazzy-rosbag2-storage-mcap python3-opencv
source /opt/ros/jazzy/setup.bash
cd "$(dirname "$0")/../ros2_ws"
colcon build --symlink-install
