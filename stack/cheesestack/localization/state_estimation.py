"""State estimation node: CTRV EKF fusing GPS, wheel odometry, IMU yaw rate and GPS course.

State: [x, y, yaw, v, yaw_rate] in `map` (ENU). Output: `core.VehicleState`
with a 5x5 covariance over that state, curvature from `CurvatureEstimator`,
and longitudinal acceleration (filtered dv/dt of the estimate).

Inputs (pure methods; the Node also subscribes to bus topics):
- `update_gps(GpsMeasurement)`  -> position update (+ optional course update)
- `update_wheels(WheelOdomEstimate)` -> speed (+ yaw rate) update
- `update_imu(ImuSample)` -> yaw-rate update (bias-corrected if an estimator is attached)
- `update_speed`, `update_yaw_rate`, `update_heading` raw scalar updates

Each update first predicts to the measurement stamp. Measurements older than
the filter time are applied without prediction (small-latency approximation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cheesestack.core import Header, MessageBus, Node, NodeStatus, Pose2D, VehicleState, Clock, wrap_angle

from .gps import CHI2_2DOF, GpsMeasurement
from .imu import ImuBiasEstimator, ImuSample
from .kalman import CTRVModel, ExtendedKalmanFilter
from .motion_estimation import CurvatureConfig, CurvatureEstimator
from .wheel_odometry import WheelOdomEstimate

CHI2_1DOF_999 = 10.83


@dataclass
class StateEstimatorConfig:
    rate_hz: float = 50.0
    process: CTRVModel = field(default_factory=CTRVModel)
    init_pos_var: float = 25.0
    init_yaw_var: float = math.pi**2
    init_v_var: float = 4.0
    init_yaw_rate_var: float = 0.1
    gyro_var: float = 1e-4
    course_var: float = math.radians(5.0) ** 2
    course_min_speed_mps: float = 2.0
    gate_gps: bool = True
    gps_gate_probability: float = 0.999
    gate_scalar: bool = True
    max_predict_dt: float = 0.5  # split long predictions into steps
    accel_tau_s: float = 0.3
    curvature: CurvatureConfig = field(default_factory=CurvatureConfig)
    frame_id: str = "map"
    output_topic: str = "/localization/state"
    gps_topic: str = "/localization/gps"
    wheels_topic: str = "/localization/wheel_odom"
    imu_topic: str = "/sensors/imu"


class StateEstimator(Node):
    def __init__(self, config: Optional[StateEstimatorConfig] = None, bus: Optional[MessageBus] = None,
                 clock: Optional[Clock] = None, imu_bias: Optional[ImuBiasEstimator] = None,
                 name: str = "state_estimator") -> None:
        super().__init__(name, bus, clock)
        self.config = config or StateEstimatorConfig()
        self.rate_hz = self.config.rate_hz
        c = self.config
        self.model = c.process
        self.ekf = ExtendedKalmanFilter(np.zeros(5), np.diag([c.init_pos_var, c.init_pos_var, c.init_yaw_var,
                                                              c.init_v_var, c.init_yaw_rate_var]), angle_indices=(2,))
        self.t: Optional[float] = None
        self.position_initialized = False
        self.heading_initialized = False
        self.imu_bias = imu_bias
        self.curvature = CurvatureEstimator(c.curvature)
        self.accel = 0.0
        self.steer = 0.0
        self._seq = 0
        self.last_nis: dict[str, float] = {}
        self.rejected: dict[str, int] = {}
        self._subs = [
            self.bus.subscribe(c.gps_topic, lambda _t, m: self.update_gps(m)),
            self.bus.subscribe(c.wheels_topic, lambda _t, m: self.update_wheels(m)),
            self.bus.subscribe(c.imu_topic, lambda _t, m: self.update_imu(m)),
        ]

    # --- core filter ops -------------------------------------------------
    def predict_to(self, t: float) -> None:
        if self.t is None:
            self.t = t
            return
        dt_total = t - self.t
        if dt_total <= 0:
            return
        n = max(1, int(math.ceil(dt_total / self.config.max_predict_dt)))
        dt = dt_total / n
        v0 = self.ekf.x[3]
        for _ in range(n):
            x = self.ekf.x
            self.ekf.predict(lambda s: self.model.f(s, dt), self.model.Q(x, dt), self.model.jacobian(x, dt))
        k = dt_total / (self.config.accel_tau_s + dt_total)
        self.accel += k * ((self.ekf.x[3] - v0) / dt_total - self.accel)
        self.t = t

    def _scalar(self, name: str, idx: int, z: float, var: float, angle: bool = False) -> bool:
        H = np.zeros((1, 5))
        H[0, idx] = 1.0
        gate = CHI2_1DOF_999 if self.config.gate_scalar else None
        nis = self.ekf.update(np.array([z]), lambda s: s[idx:idx + 1], np.array([[var]]), H=H,
                              residual_angles=(0,) if angle else (), gate=gate)
        self.last_nis[name] = nis
        if gate is not None and nis > gate:
            self.rejected[name] = self.rejected.get(name, 0) + 1
            return False
        return True

    # --- measurement interfaces -----------------------------------------
    def update_gps(self, m: GpsMeasurement) -> bool:
        if not m.accepted or not np.all(np.isfinite(m.xy)):
            return False
        self.predict_to(m.header.stamp)
        if not self.position_initialized:
            self.ekf.x[:2] = m.xy
            self.ekf.P[:2, :2] = m.cov
            self.ekf.P[:2, 2:] = 0.0
            self.ekf.P[2:, :2] = 0.0
            self.position_initialized = True
        else:
            H = np.zeros((2, 5))
            H[0, 0] = H[1, 1] = 1.0
            gate = CHI2_2DOF.get(self.config.gps_gate_probability, 13.816) if self.config.gate_gps else None
            nis = self.ekf.update(m.xy, lambda s: s[:2], m.cov, H=H, gate=gate)
            self.last_nis["gps"] = nis
            if gate is not None and nis > gate:
                self.rejected["gps"] = self.rejected.get("gps", 0) + 1
                return False
        if m.course_yaw is not None and (m.speed_mps or 0.0) >= self.config.course_min_speed_mps:
            self.update_heading(m.header.stamp, m.course_yaw, self.config.course_var)
        return True

    def update_heading(self, t: float, yaw: float, var: float) -> bool:
        self.predict_to(t)
        if not self.heading_initialized:
            self.ekf.x[2] = wrap_angle(yaw)
            self.ekf.P[2, :] = 0.0
            self.ekf.P[:, 2] = 0.0
            self.ekf.P[2, 2] = var
            self.heading_initialized = True
            return True
        return self._scalar("heading", 2, yaw, var, angle=True)

    def update_speed(self, t: float, v: float, var: float) -> bool:
        self.predict_to(t)
        return self._scalar("speed", 3, v, var)

    def update_yaw_rate(self, t: float, r: float, var: float) -> bool:
        self.predict_to(t)
        return self._scalar("yaw_rate", 4, r, var)

    def update_wheels(self, w: WheelOdomEstimate) -> bool:
        ok = self.update_speed(w.stamp, w.v, w.v_var)
        if w.yaw_rate_source in ("rear", "front"):
            self._scalar("wheel_yaw_rate", 4, w.yaw_rate, w.yaw_rate_var)
        return ok

    def update_imu(self, s: ImuSample) -> bool:
        gz = float(s.gyro[2])
        if self.imu_bias is not None:
            self.imu_bias.update(s)
            gz -= float(self.imu_bias.gyro_bias[2])
        return self.update_yaw_rate(s.stamp, gz, self.config.gyro_var)

    # --- output ------------------------------------------------------------
    def state(self, t: Optional[float] = None) -> VehicleState:
        if t is not None:
            self.predict_to(t)
        x = self.ekf.x
        stamp = self.t if self.t is not None else 0.0
        self._seq += 1
        kappa = self.curvature.update(stamp, float(x[3]), float(x[4]), self.steer)
        return VehicleState(
            header=Header(stamp=stamp, frame_id=self.config.frame_id, seq=self._seq, source=self.name),
            pose=Pose2D(float(x[0]), float(x[1]), float(x[2])),
            v=float(x[3]), a=float(self.accel), yaw_rate=float(x[4]), steer=self.steer,
            curvature=kappa, covariance=self.ekf.P.copy(),
        )

    def step(self, t: float) -> None:
        st = self.state(t)
        self.status = NodeStatus.OK if (self.position_initialized and self.heading_initialized) else NodeStatus.DEGRADED
        if np.trace(self.ekf.P[:2, :2]) > 4 * self.config.init_pos_var:
            self.status = NodeStatus.DEGRADED
        self.bus.publish(self.config.output_topic, st)
    # TODO(localization): out-of-order measurement handling via state history + re-filtering,
    # IMU-driven prediction (error-state), lever arms via FrameTree, map-matching updates,
    # health Faults (e.g. LOC_GPS_LOST) once the safety layer's fault codes are agreed.
