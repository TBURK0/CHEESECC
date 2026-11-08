# control

Planned trajectory + ego state -> `core.ControlCommand` (steer_wheel_deg, throttle, brake, curvature, target_speed_mps).

**Sign convention.** Internally + = left (CONVENTIONS.md). `ControlCommand.curvature` / `steer_wheel_deg`
use the app's DriveCommand convention (+ = **right**, compass bearings) by default; set
`app_sign_convention=False` for stack sign. The flip happens only in `curvature_control.make_control_command`.

| Module | Capability | Works now | Key TODOs |
|---|---|---|---|
| `vehicle_params.py` | Vehicle model params | CarModel-equivalent fields, full-lock curvature, understeer gradient, `from_car_model(cars.json entry)` | Identify Cf/Cr/Iz per car |
| `reference_path.py` | Path projection | Polyline projection: s, signed cross-track, heading error, interpolated curvature/speed/accel | Spline smoothing, windowed search by default |
| `path_tracking.py` | Path following | Pure pursuit (port of PurePursuit.swift, speed-adaptive lookahead), Stanley, rear-wheel feedback | Circle-intersection lookahead, Stanley yaw damping, reverse |
| `lqr_lateral.py` | Trajectory tracking (lateral) | Dynamic-bicycle error model, ZOH via numpy expm, iterative DARE, speed-bucketed gains, Rajamani feedforward | Preview, delay-augmented state |
| `longitudinal.py` | Trajectory tracking (longitudinal) | Speed PID + accel FF; exact inverse of LongitudinalModel.acceleration -> pedals; ports of targetSpeed / pedals | Grade comp, calibrated pedal maps |
| `dynamics_comp.py` | Dynamics compensation | Understeer FF (K_us v^2 kappa), first-order-lag lead compensator, delay pose prediction | Online lag identification |
| `curvature_control.py` | Curvature control | kappa -> road wheel -> handwheel (BicycleModel.swift), curvature rate/magnitude limit, ControlCommand builder | Speed-dependent rate limits, learned steer map |
| `state_machine.py` | Mode management | OFF/STANDBY/ENGAGED/OVERRIDE/FAULT, torque / steer-divergence / pedal override with debounce + release, latched faults | Safety-layer preconditions, split lat/lon modes |
| `tuning.py` | Tuning infra | `GainSchedule` vs speed, step metrics (rise, overshoot, peak, settling, ss error) | Autotune / sim sweeps |
| `stability.py` | Stability limiting | a_lat and yaw-rate curvature caps, full lock, curve speed limit | Friction estimate, sideslip check |
| `saturation.py` | Saturation | Clamp, rate limiter, PID with clamp / back-calculation anti-windup | — |
| `oscillation.py` | Oscillation detection | Sliding-window sign-change rate + amplitude, Hann FFT dominant frequency | Band-energy ratio, gain backoff hook |
| `actuator_monitor.py` | Actuator monitoring | RMS/max error, cross-correlation lag, stuck detection -> `core.Fault` | System ID (tau + delay) |
| `controller.py` | Integration | `TrajectoryController` (selectable lateral law -> stability -> rate limit -> handwheel; PID pedals; mode gating), `ControlNode` on the bus (`/localization/state`, `/planning/trajectory` -> `/control/command`) | Topic names to be confirmed by pipeline |

Tests: `python3 -m pytest tests/control` (closed-loop kinematic and dynamic sims, app-plant parity).
