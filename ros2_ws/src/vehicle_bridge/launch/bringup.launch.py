"""Both bridges.  ros2 launch vehicle_bridge bringup.launch.py port:=/dev/ttyUSB0
Simulator (no hardware):  port:=sim:/path/to/uno_q_sim
Arming stays impossible unless allow_arm:=true is passed explicitly."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    vb_cfg = os.path.join(get_package_share_directory('vehicle_bridge'), 'config', 'vehicle_bridge.yaml')
    sb_cfg = os.path.join(get_package_share_directory('sensor_bridge'), 'config', 'sensor_bridge.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('port', default_value='/dev/ttyUSB0'),
        DeclareLaunchArgument('allow_arm', default_value='false'),
        Node(package='vehicle_bridge', executable='vehicle_bridge_node', output='screen',
             parameters=[vb_cfg, {
                 'port': LaunchConfiguration('port'),
                 'allow_arm': ParameterValue(LaunchConfiguration('allow_arm'), value_type=bool)}]),
        Node(package='sensor_bridge', executable='sensor_bridge_node', output='screen',
             parameters=[sb_cfg]),
    ])
