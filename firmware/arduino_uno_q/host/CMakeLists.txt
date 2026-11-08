cmake_minimum_required(VERSION 3.16)
project(uno_q_host CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Werror)

# The SAME core sources the Arduino sketch compiles.
set(CORE ${CMAKE_CURRENT_SOURCE_DIR}/../uno_q_vehicle)
set(CORE_SRC ${CORE}/protocol.cpp ${CORE}/safety.cpp ${CORE}/ultrasonic.cpp
             ${CORE}/imu.cpp ${CORE}/actuators.cpp ${CORE}/vehicle.cpp)
include_directories(${CORE} ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(uno_q_sim sim_main.cpp ${CORE_SRC})
target_compile_definitions(uno_q_sim PRIVATE HOST_SIM)

add_executable(test_core test_core.cpp ${CORE_SRC})
