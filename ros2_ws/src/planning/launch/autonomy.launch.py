from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = Path(get_package_share_directory('localization'))/'config'
    sim = LaunchConfiguration('simulation')
    nodes = [DeclareLaunchArgument('simulation', default_value='true'),
             DeclareLaunchArgument('autonomy_enabled', default_value='false'),
             DeclareLaunchArgument('scenario', default_value='clear'),
             DeclareLaunchArgument('synthetic_mounts', default_value=sim),
             DeclareLaunchArgument('use_sim_time', default_value='false')]
    clock = {'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)}
    nodes += [Node(package='planning', executable='simulation_node', condition=IfCondition(sim),
                   parameters=[clock, {'scenario': LaunchConfiguration('scenario')}]),
              Node(package='robot_localization', executable='ekf_node', name='ekf_local',
                   parameters=[str(config/'ekf_local.yaml'), clock], remappings=[('odometry/filtered', '/localization/odom')]),
              Node(package='robot_localization', executable='ekf_node', name='ekf_global',
                   parameters=[str(config/'ekf_global.yaml'), clock], remappings=[('odometry/filtered', '/localization/global')]),
              Node(package='robot_localization', executable='navsat_transform_node', name='navsat_transform',
                   parameters=[str(config/'navsat.yaml'), clock], remappings=[('imu', '/sensors/imu'), ('gps/fix', '/phone/gps'), ('odometry/filtered', '/localization/global')]),
              Node(package='mapping', executable='mapping_node', parameters=[clock]),
              Node(package='planning', executable='planning_node', parameters=[clock]),
              Node(package='safety_supervisor', executable='safety_node', parameters=[clock, {
                  'simulation': ParameterValue(sim, value_type=bool),
                  'autonomy_enabled': ParameterValue(LaunchConfiguration('autonomy_enabled'), value_type=bool)}])]
    # Synthetic mounts are explicit identities. Real extrinsics require measurement.
    for child in ['imu_link', 'gps_link', 'camera_color_optical_frame']+['ultrasonic_{}'.format(i) for i in range(4)]:
        nodes.append(Node(package='tf2_ros', executable='static_transform_publisher',
                          arguments=['--frame-id', 'base_link', '--child-frame-id', child], condition=IfCondition(LaunchConfiguration('synthetic_mounts')), parameters=[clock]))
    return LaunchDescription(nodes)
