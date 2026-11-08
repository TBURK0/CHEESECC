"""Layer 2: position, heading and speed.  (Step 6, after steps 2, 4 and 5)

Recommended design: an EKF driven by the IMU, corrected by wheel speed and
visual odometry. The car can't slide sideways, so use a kinematic model:

State  s = [x, y, yaw, v, gyro_bias_z, accel_bias_x]

Predict (every IMU sample):
    x   += v * cos(yaw) * dt
    y   += v * sin(yaw) * dt
    yaw += (gyro_z - gyro_bias_z) * dt
    v   += (accel_x_without_gravity - accel_bias_x) * dt
    (gravity removal uses roll/pitch from node.state["roll"/"pitch"])

Updates:
    WheelSpeed   v = meters_per_tick * dticks/dt               (step 4 scale)
    VoPose       x, y, yaw, after mapping the VO frame into the EKF frame;
                 re-map whenever VoPose.reset is True
    stationary   when the gyro is quiet for ~0.25 s: v = 0 and
                 gyro_bias_z = current gyro_z (zero-velocity update)

Two things that matter on the real car and are easy to skip:
  1. VO arrives late (vo_latency_s from step 2). Keep a short history of
     (time, state, covariance, IMU input); when a VoPose arrives, rewind to
     its capture time, apply it, and re-run the predictions since then.
  2. Gate every update: compute the Mahalanobis distance of the innovation
     and reject it above the chi-square 99.9% threshold (3 dof: 16.3). Count
     rejections; a long run of them means the VO jumped and needs re-mapping.

Increment plan, each one verified on REPLAYED logs before running live:
  6a. Predict only, with wheel speed: dead reckoning. Expect drift; measure it.
  6b. + stationary updates: gyro bias should settle during stops.
  6c. + VO updates, in-order (ignore latency): loop-closure error vs 6a.
  6d. + delayed-update history. Replay with deliver_at_arrival=True and
      check the result matches 6c within noise.
  6e. + gating and VO re-mapping. Test: cover the lens mid-run.
"""
from __future__ import annotations

import numpy as np

from ..node import Layer

IX, IY, IYAW, IV, IBG, IBA = range(6)


class PoseEKF(Layer):
    name = "pose"
    requires = ("imu.rate_hz", "imu.gyro_noise", "imu.accel_noise", "imu.gyro_bias_walk")

    def __init__(self, cfg):
        super().__init__(cfg)
        self.x = np.zeros(6)
        self.P = np.diag([1e-4, 1e-4, 1e-4, 1e-2, 1e-4, 0.1])
        self.t = None
        self.rejected = 0

    # ---------------------------------------------------------------- inputs
    def on_ImuSample(self, msg):
        # TODO (6a): gravity-compensate forward accel using node.state roll/pitch,
        #            then self.predict(msg.t, gyro_z, a_fwd)
        raise NotImplementedError("Step 6a: IMU prediction")

    def on_WheelSpeed(self, msg):
        # TODO (6a): convert ticks to speed with vehicle.meters_per_tick
        #            (difference from previous sample), then update v
        pass

    def on_VoPose(self, msg):
        # TODO (6c-6e): map into EKF frame, delayed update, gate
        pass

    # ------------------------------------------------------------- the filter
    def predict(self, t, gyro_z, a_fwd):
        """TODO (6a): state propagation + Jacobian F + process noise Q."""
        raise NotImplementedError

    def update(self, z, H, R, angle_rows=()):
        """Generic EKF update with chi-square gating. Returns True if applied.
        angle_rows: indices in z that are angles (wrap the innovation)."""
        y = z - H @ self.x
        for i in angle_rows:
            y[i] = (y[i] + np.pi) % (2 * np.pi) - np.pi
        S = H @ self.P @ H.T + R
        d2 = float(y @ np.linalg.solve(S, y))
        # TODO (6e): pick the threshold for len(z) dof and reject above it
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(len(self.x)) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T   # Joseph form, stays symmetric PSD
        return True

    def state(self):
        return {"x": self.x[IX], "y": self.x[IY], "yaw": self.x[IYAW], "v": self.x[IV],
                "gyro_bias_z": self.x[IBG], "pos_std": float(np.sqrt(self.P[0, 0] + self.P[1, 1]))}
