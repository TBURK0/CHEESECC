std_msgs/Header header
bool healthy
bool stop_sign
# NONE is permitted only when the scene is positively observed and no light exists.
string signal_state
float32 path_confidence
vehicle_interfaces/Detection[] detections
