std_msgs/Header header
string state                 # DISARMED | ARMED | ESTOP | UNKNOWN (no telemetry yet)
bool link_ok
bool armed
uint32 faults                # bitmask, see docs/serial_protocol.md
string[] fault_names
float32 cmd_age_s
# values currently DRIVEN by the MCU (after limits / safe stop)
float32 throttle
float32 brake
float32 steering
uint32 bad_frames
string firmware_version
bool outputs_enabled_build
bool hw_watchdog_verified
