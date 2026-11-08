"""Sensor fusion framework for the autonomous RC car.

    fusion.messages     timestamped message types (the hardware contract)
    fusion.io           RunLogger, ReplaySource, LiveSource adapter
    fusion.node         FusionNode + Layer base class
    fusion.layers       attitude, accel array, pose EKF, obstacles (skeletons)
    fusion.safety       stopping-distance math + monitor skeleton
"""
