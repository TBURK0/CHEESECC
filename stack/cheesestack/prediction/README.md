# prediction

Future motion of the ego and other agents. Imports only `numpy` + `cheesestack.core`.
Shared output type: `PredictedTrajectory` (times, `[x, y, yaw, v, yaw_rate, a]` states,
per-step 2x2 position covariance, probability); `.to_trajectory()` gives a `core.Trajectory`.

| module | capability | works now | key TODOs |
|---|---|---|---|
| `motion_models.py` | vehicle / ego trajectory prediction | CV, CA, CTRV, CTRA exact-arc rollout, no-reverse stop handling, EKF covariance growth (numeric Jacobian + process noise); `EgoPredictor` from `VehicleState` (curvature fallback) | blend with planner intent; map-constrained rollout |
| `object_prediction.py` | object trajectory prediction | per-class hypothesis sets (vehicle keep/turn L/R/brake, VRU keep/stop, static), yaw-rate/accel from track history, (vx,vy)->(yaw,v) covariance conversion, likelihood-weighted probabilities, pruning | map-aware (lane-following) hypotheses; learned multimodal predictor behind same output |
| `lane_change.py` | lane-change prediction | `Lane` centerline polylines, Frenet projection, features (d, d_dot, rel heading, 1/TLC), `LaneChangeModel` interface, softmax-logistic baseline with GD `fit`, neighbour-lane discovery | temporal smoothing, lane-marking legality, gap acceptance, trained weights |
| `collision.py` | collision prediction | SAT oriented-box overlap + depth, earliest collision time on common grid, Monte-Carlo collision probability from covariance, multi-hypothesis aggregation | yaw uncertainty, analytic/importance-sampled probability, speed-up (vectorised SAT) |
| `ttc.py` | time-to-collision | 1-D TTC, stop-aware constant-accel (enhanced) TTC, 2-D CPA, disc TTC, `TTCEstimator` ego-vs-object in ego frame with bumper gaps | multi-disc footprints, curved-path TTC |
| `risk.py` | risk scoring | TTC ramp + collision prob noisy-OR, class vulnerability, uncertainty inflation -> [0,1] + level; scene aggregation | calibration vs near-miss logs, impact-energy severity, occlusion term |
| `behavior.py` | behavioral forecasting | intents keep/turn L/R/stop/yield/cut-in; features from track history (+ ego lane); Gaussian likelihoods with gates; sticky-HMM forward filter per track; `IntentModel` interface | map features (stop lines, crosswalks, junctions), learned model, feed intents into object-hypothesis priors |
