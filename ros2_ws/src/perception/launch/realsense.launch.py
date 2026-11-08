from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('weights', default_value=''),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')),
            launch_arguments={'align_depth.enable': 'true', 'enable_sync': 'true',
                'enable_color': 'true', 'enable_depth': 'true', 'enable_rgbd': 'true'}.items()),
        Node(package='perception', executable='perception_node',
             parameters=[{'weights': LaunchConfiguration('weights')}])])
