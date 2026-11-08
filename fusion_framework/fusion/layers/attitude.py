"""Layer 1: roll and pitch from the MPU gyro + accelerometer.  (Step 3)

Recommended method: a complementary filter first, Mahony later if needed.

    angle = angle + gyro_rate * dt                   # fast, drifts
    angle = (1 - k) * angle + k * accel_angle        # slow pull toward gravity

The accelerometer measures gravity PLUS the car's own acceleration, so:
  * skip the accel correction when |accel| differs from g by more than
    ~10-15% (hard acceleration, braking, bumps),
  * expect cornering to tilt the estimate slightly; it should recover
    within a second after the corner.

Yaw is NOT observable here (no magnetometer). The pose EKF owns heading.

Increment plan:
  3a. Static: tilt the car to known angles, compare. Pass: within ~1 deg.
  3b. Motor running, wheels off the ground. Vibration shows up here; fix it
      mechanically (foam/gel mount) and with the MPU's DLPF before touching
      filter gains.
  3c. Driving by hand: pitch on braking should return to ~0 within ~1 s.
"""
from __future__ import annotations

import numpy as np

from ..calibration import apply_accel_calib, to_body
from ..node import Layer


class AttitudeLayer(Layer):
    name = "attitude"
    requires = ("imu.rate_hz", "imu.gyro_noise")

    def __init__(self, cfg, accel_gate=0.15, k=0.02):
        super().__init__(cfg)
        self.accel_gate = accel_gate   # fraction of g
        self.k = k                     # correction strength per sample
        self.roll = 0.0
        self.pitch = 0.0
        self._last_t = None
        self.mount = cfg["imu"]["mount_rpy_deg"]
        self.accel_calib = cfg["imu"].get("accel_calib")

    def on_ImuSample(self, msg):
        gyro = to_body(msg.gyro, self.mount)
        accel = to_body(apply_accel_calib(msg.accel, self.accel_calib), self.mount)
        if self._last_t is None:
            self._last_t = msg.t
            # TODO (3a): initialize roll/pitch from the first accel reading
            return
        dt = msg.t - self._last_t
        self._last_t = msg.t
        if dt <= 0:
            return
        # TODO (3a): integrate gyro (gyro[0] -> roll rate, gyro[1] -> pitch rate
        #            for small angles), then blend toward the accel angles:
        #            roll_acc  = atan2(ay, az)
        #            pitch_acc = atan2(-ax, sqrt(ay^2 + az^2))
        #            gated by abs(|accel| - g) < accel_gate * g
        # TODO (3b): log how often the gate rejects accel while driving; if it
        #            is most of the time, vibration filtering comes first.
        raise NotImplementedError("Step 3: attitude filter")

    def state(self):
        return {"roll": self.roll, "pitch": self.pitch}
