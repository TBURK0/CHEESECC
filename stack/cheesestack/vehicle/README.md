# vehicle — Vehicle State layer

Internal understanding of the car: parameters, steering, yaw rate, slip, wheel
speeds, dynamics models, diagnostics and health. numpy + `cheesestack.core` only.

| Module | Capability | Works now | Key TODOs |
|---|---|---|---|
| `params.py` | Vehicle parameters | `VehicleParams` (SI) from the app's `cars.json` (`load_catalog(path)`, `load_vehicle_params(id)`; default path resolved relative to the repo). Derives lf/lr, cornering stiffness, Iz, understeer gradient, turn radius, characteristic speed | Measured CG / Iz / Cf, Cr per car |
| `steering.py` | Steering state | Handwheel <-> road-wheel via ratio, curvature conversions (matches `BicycleModel.swift`), backlash (play operator), center deadband, filtered rates, lock saturation | Online offset calibration, variable ratio / EPS compliance, Ackermann |
| `dynamics.py` | Vehicle dynamics | RK4; kinematic bicycle (rear axle); linear dynamic bicycle (CG, friction-capped tires, low-speed kinematic blend, LTI lateral state space, steady state); longitudinal model (drive torque/power/traction after `LongitudinalModel.swift`, drag, rolling, grade) | Nonlinear tires (Pacejka), load transfer, gearbox, actuator lag, reverse |
| `yaw_rate.py` | Yaw-rate estimation | 2-state KF [r, gyro bias] fusing gyro with steady-state bicycle model; model gated out when it disagrees (sliding) | Bank/roll coupling, learned residual model |
| `slip.py` | Slip estimation | Per-wheel slip ratio from body motion + steer; sideslip KF on linear bicycle with gyro + a_y; kinematic at low speed | Bank compensation, online Cf/Cr adaptation, fused reference speed |
| `wheel_speed.py` | Wheel-speed validation | Turn-geometry compensation, median outlier test, GPS cross-check, debounced `Fault`s, tire-radius scale estimate | Direction detection, spin/lock vs sensor fault via IMU |
| `plausibility.py` | Sensor plausibility | Range, NaN, rate-of-change, stuck-value, timeout, cross-sensor checks; default specs for core signals | Noise-floor checks, frozen timestamps |
| `diagnostics.py` | Diagnostic state machines | DTC NO_FAULT/PENDING/CONFIRMED/HEALED with counter debounce, freeze frames, occurrence count, aging over operation cycles, `Fault` export | Persistent DTC memory, UDS status byte |
| `health.py` | Health monitoring | Weighted subsystem aggregation, staleness, non-critical capping, `ok_to_drive` / `degraded` report | Hysteresis, ODD capability mapping |

Tests: `python3 -m pytest tests/vehicle`.
