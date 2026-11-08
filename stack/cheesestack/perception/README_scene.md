# Perception, part B: scene understanding

All modules are numpy-only. Image inputs are HxW / HxWx3 arrays (uint8 or float [0,1]);
geometry (homographies, metric scales) is passed in by the caller, no dependency on
calibration/ingest modules.

| Module | Capability | Works now | Key TODOs |
|---|---|---|---|
| `detection.py` | Object detection | `Detector` ABC, `ModelDetector` adapter for learned backends, greedy + class-aware NMS, `BlobDetector` (median/background differencing + connected components). Shared image helpers (gray, Sobel, HSV, CC labelling, flood fill, hole fill). | Real detector backend, 3D lifting via ground plane, per-class thresholds |
| `classification.py` | Object classification | `Classifier` ABC, `FeatureClassifier` (Gaussian NB over aspect/height/fill/colour; default shape priors, `fit()`), `map_label` COCO/KITTI/nuScenes -> `ObjectClass` | CNN crop classifier, probability calibration |
| `tracking.py` | Multi-object tracking | CV Kalman per track (Joseph update), numpy Hungarian (rectangular), chi-square gating, class-mismatch penalty, tentative/confirmed/lost lifecycle, outputs `TrackedObject` | Ego-motion compensation, CTRV/IMM, appearance cost |
| `lanes.py` | Lane detection | `LaneModel` ABC; classical: homography warp to BEV, gradient + white/yellow threshold, histogram + sliding window, prior-fit search, quadratic fit in metres -> width, offset, heading, curvature, confidence | Learned lane model, Kalman on coefficients, multi-lane & marking type |
| `road_edge.py` | Road-edge detection | Per-row boundary of the ego-connected drivable region, RANSAC quadratic per side, optional metric fit & road width | Curb cues (depth), temporal smoothing, edge type |
| `drivable_space.py` | Drivable space | `DrivableSpaceModel` ABC; seeded Mahalanobis colour+texture model + region growing; per-column free-space boundary; projection to ground via homography | Segmentation net, adaptive seed, BEV temporal fusion |
| `traffic_signs.py` | Traffic-sign recognition | `TrafficSignRecognizer` ABC; HSV colour masks, shape by fill ratio (octagon/circle/triangle/diamond/rect), ring & bar cues -> `SignType`; `SignVoter` IoU-associated temporal voting | Learned classifier, speed-value OCR, 3D localisation |
| `traffic_lights.py` | Traffic-light recognition | `TrafficLightClassifier` ABC; HSV lit-pixel counting with vertical-housing prior; `TrafficLightFilter` HMM forward filter with light-cycle transitions | Arrows/flashing, exposure-robust colour model, map association |
| `occupancy_grid.py` | Occupancy grid | Ego-centred log-odds grid, ray casting of range scans, point/detection-footprint insertion, exponential decay, ego-motion shift (rotation + translation), queries | Beam sensor model, 3D height filtering, core grid message |
| `fusion.py` | Sensor fusion | `SourceRegistry` (noise, reliability, latency), cross-source association (Hungarian on Mahalanobis, one meas/source/cluster), information-form merge or covariance intersection, reliability-weighted class vote, noisy-OR existence | Velocity/extent fusion, track-to-track fusion, OOSM |

Tests: `tests/perception/test_scene_*.py` (synthetic images/scenes, < 1 s total).
