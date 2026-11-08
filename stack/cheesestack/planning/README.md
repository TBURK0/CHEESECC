# planning

Pure numpy, no imports from other layers. Lane/road geometry arrives as polylines,
boundary polynomials or `TrackedObject`s; integration (`pipeline.py`) does the wiring.

| Module | Capability | Works now | Key TODOs |
|---|---|---|---|
| `common.py` | Shared primitives | `ReferencePath` (s, yaw, kappa, Frenet <-> Cartesian), quintic/quartic polynomials, `ObstacleBox` + SAT overlap | spline reference, windowed projection |
| `mission.py` | Mission planning | Ordered goals, IDLE/ACTIVE/ARRIVED/ABORTED, pos/heading/speed arrival checks | persistence, goal reordering, per-goal timeouts |
| `route.py` | Route planning | Local `WaypointGraph`, A* (weighted option), snap-to-node planning, `Route` lookup/projection, `lookahead_distance` (port of the app's LookaheadScheduler) | lane-level graph, traffic costs, D* Lite replanning |
| `behavior.py` | Behavioral FSM | LANE_FOLLOW / FOLLOW_VEHICLE / PREPARE_LANE_CHANGE / LANE_CHANGE / STOP / EMERGENCY_STOP from `WorldSummary`; dwell, hysteresis, abort, timeout; target speed | traffic lights/signs, yielding, MRM pull-over, learned policy |
| `lane_centering.py` | Lane centering | Centreline from cubic boundaries (both/one side, width sanity), lookahead target with deadband + slew limit, heading/curvature at lookahead | map fusion, confidence blending |
| `gaps.py` | Gap assessment | Target-lane filtering, gap enumeration, static margin + headway + TTC checks, scoring | predicted (non-CV) motion, covariance, competing mergers |
| `merge.py` | Merge planning | Gap x horizon search, quintic longitudinal profile to propagated gap, accel/jerk limits | interaction-aware, end-of-lane deadline, commitment |
| `overtake.py` | Overtake decision | Pass-time kinematics, oncoming / passing-lane / sight checks, time-saved vs risk cost | mid-manoeuvre abort, speed uncertainty |
| `path_optimization.py` | Path optimization | Quadratic smoothing (single linear solve), resampling, curvature, Frenet lateral quintic sampling with obstacle clearance | inequality constraints (QP solver), banded solve |
| `trajectory.py` | Trajectory generation | Werling-style lattice: quintic lateral x quartic longitudinal (quintic for stop), cost ranking, constraint filtering, braking fallback, `core.Trajectory` output | prediction-layer obstacles, low-speed mode, temporal consistency |
| `constraints.py` | Constraint evaluation | Curvature, lateral accel, accel, jerk, speed, oriented-box collision vs constant-velocity obstacles | uncertainty inflation, road boundaries, curvature rate |
| `comfort.py` | Comfort modeling | ISO 2631-1-style a_v (k=1.4 horizontal), comfort bands, [0,1] score, jerk/accel cost | W_d/W_f frequency weighting, vertical axis, motion sickness dose |

Tests: `python3 -m pytest tests/planning` (44 tests, ~1 s).
