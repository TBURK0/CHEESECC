cmake_minimum_required(VERSION 3.8)
project(vehicle_interfaces)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Detection.msg"
  "msg/MotionProposal.msg"
  "msg/ObstacleStatus.msg"
  "msg/SafetyStatus.msg"
  "msg/Scene.msg"
  DEPENDENCIES std_msgs builtin_interfaces)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
