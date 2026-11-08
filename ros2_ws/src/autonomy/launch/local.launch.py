"""Local autonomy only. Launch vehicle_bridge separately; arming is not requested."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('mount_calibrated', default_value='false'),
        DeclareLaunchArgument('camera_height_m', default_value='0.2'),
        DeclareLaunchArgument('geometry_calibrated', default_value='false'),
        DeclareLaunchArgument('command_source', default_value='candidate'),
        DeclareLaunchArgument('operator', default_value='driver'),
        DeclareLaunchArgument('use_tf', default_value='true'),
        DeclareLaunchArgument('require_ultrasonics', default_value='false'),
        DeclareLaunchArgument('config', default_value=os.path.join(
            get_package_share_directory('autonomy'), 'config', 'autonomy.yaml')),
        Node(package='autonomy', executable='autonomy_pipeline', output='screen', parameters=[LaunchConfiguration('config'), {
            'mount_calibrated': ParameterValue(LaunchConfiguration('mount_calibrated'), value_type=bool),
            'camera_height_m': ParameterValue(LaunchConfiguration('camera_height_m'), value_type=float),
            'geometry_calibrated': ParameterValue(LaunchConfiguration('geometry_calibrated'), value_type=bool),
            'command_source': LaunchConfiguration('command_source'),
            'operator': LaunchConfiguration('operator'),
            'use_tf': ParameterValue(LaunchConfiguration('use_tf'), value_type=bool),
            'require_ultrasonics': ParameterValue(LaunchConfiguration('require_ultrasonics'), value_type=bool),
        }]),
    ])
