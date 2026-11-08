"""Combining the MPU accelerometer with the extra 3-axis accelerometers.
(Steps 1 and 3; optional, add after the single-IMU attitude filter works)

On a rigid body, the specific force at point r_i relative to the MPU is

    f_i = f_0 + alpha x r_i + omega x (omega x r_i)

omega is the gyro reading, so the last term is known. With the MPU plus 3+
non-collinear extra units, solve by least squares for
    f_0    acceleration at the MPU (lower noise than the MPU alone)
    alpha  angular acceleration (useful for detecting slides/spins)
and drop a unit whose residual is much larger than the others (loose mount,
failed sensor).

Decide on the evidence from Step 1 whether this is worth it:
  * extra units much noisier than the MPU -> use them only for fault
    detection, or not at all;
  * units mounted close together or on a line -> alpha is not observable,
    use them for averaging only.

Increment plan:
  A. Log all units side by side at rest and while driving (Step 1).
  B. Average-only combination: compare noise vs. MPU alone.
  C. Full lever-arm solve: check alpha against the derivative of the gyro.
"""
from __future__ import annotations

from ..node import Layer


class AccelArrayLayer(Layer):
    name = "accel_array"
    requires = ("imu.position",)

    def __init__(self, cfg):
        super().__init__(cfg)
        self.units = {u["name"]: u for u in cfg.get("extra_accels", [])}
        self.latest = {}          # name -> (t, calibrated body-frame accel)
        self.f0 = None
        self.alpha = None

    def on_AccelSample(self, msg):
        # TODO (A): calibrate + rotate into body frame, store in self.latest
        raise NotImplementedError("Accel array: store latest sample per unit")

    def on_ImuSample(self, msg):
        # TODO (B/C): combine MPU accel with self.latest samples that are
        #             fresh (within ~1 IMU period of msg.t); skip stale ones.
        pass

    def state(self):
        return {} if self.f0 is None else {"accel_f0": self.f0, "alpha": self.alpha}
