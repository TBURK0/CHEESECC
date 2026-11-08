# Actuator request from autonomy. The MCU clamps and may refuse it (see docs/serial_protocol.md).
std_msgs/Header header
float32 throttle   # 0..1 of full forward (reverse not supported in v1)
float32 brake      # 0..1 of full brake
float32 steering   # -1..1, positive = left (REP-103)
