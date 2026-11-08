# localization

`map` = ENU tangent plane (x east, y north); yaw CCW from +x. Compass bearing (app GeoMath) -> yaw via
`bearing_deg_to_yaw` (`yaw = pi/2 - bearing`). Path curvature here is **left-positive** (app's is right-positive).

| Module | Capability | Works now | Key TODOs |
|---|---|---|---|
| `gps.py` | GPS integration | NMEA GGA/RMC + checksum, exact WGS84<->ECEF<->ENU, app-compatible equirectangular, HDOP/quality & phone accuracy -> cov, chi2 + speed-jump gating | RAIM/multipath, antenna lever arm, date/week handling |
| `imu.py` | IMU integration | standstill detector, gyro bias + tilt from statics, gravity removal, planar strapdown with ZUPT | bias states in filter, coning/sculling, mount yaw calibration |
| `wheel_odometry.py` | Wheel speed | ticks (wrap-safe)/rpm -> speed, rear/front/bicycle yaw rate, slip detection (consensus + accel limit), low-pass, pose integration | tire-radius autocal, CAN decoding, direction of travel |
| `visual_odometry.py` | Visual odometry | `VisualOdometry` ABC; flat-ground IPM, Umeyama + RANSAC SE(2), covariance from residuals | non-planar (essential/PnP), keyframes/BA, learned matcher |
| `kalman.py` | KF variants | linear KF (Joseph), EKF (analytic/numeric Jacobians, gating), UKF (Merwe, angle-aware), CTRV + kinematic bicycle models | square-root UKF, IMM, error-state EKF |
| `state_estimation.py` | State estimation | `StateEstimator` Node: CTRV EKF fusing GPS pos/course, wheel speed/yaw rate, gyro; publishes `VehicleState` (5x5 cov) on `/localization/state` | OOSM re-filtering, IMU-driven prediction, Fault emission |
| `pose_tracking.py` | Pose tracking | odom->base_link dead reckoning + map->odom correction with delayed-fix history | covariance chain, pose-graph smoothing |
| `heading.py` | Heading | complementary gyro + GPS-course filter with bias integral, reverse handling | dual-antenna, magnetometer, slip angle |
| `motion_estimation.py` | Velocity / curvature | alpha-beta velocity + speed blending + accel; kappa = r v / max(v^2, vmin^2) blended with steer at low speed; Menger path curvature | map curvature, smoothing, understeer |
| `time_sync.py` | Timestamp alignment | clock offset + drift (LS + lower envelope), interpolation buffer (arrays/Pose2D, angle-aware) | PPS/PTP, convex-hull envelope, reboot jumps |
| `frames.py` | Frame transforms | TF-like tree, static + time-interpolated edges, chain lookup via common ancestor | SE3 extrinsics, config loading, bus /tf |
