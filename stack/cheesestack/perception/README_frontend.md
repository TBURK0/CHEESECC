# Perception front-end (part A)

Sensor ingest -> sync -> preprocessing/undistortion -> condition + confidence. numpy only.

| Module | Capability | Works now | Key TODOs |
|---|---|---|---|
| `camera_ingest.py` | Camera ingest | `FrameSource` ABC; `SyntheticSource`, `ArraySource`, `NpyDirectorySource`; `FrameRingBuffer`; `CameraIngest` (Node) with seq/time drop, overflow, out-of-order, latency, fps stats; optional bus publish | iPhone/AVFoundation bridge, HW timestamps + clock offset, video decode, zero-copy |
| `sensor_sync.py` | Sensor synchronization | `StreamBuffer` (sorted, nearest, linear interp w/ angle wrap, gap guard); `match_nearest`/`match_interpolated`; greedy `ApproximateTimeSynchronizer` | clock skew estimation, rolling-shutter compensation, SLERP, optimal ROS-style window search |
| `preprocessing.py` | Image preprocessing | to_float/uint8, BT.601 gray, bilinear/nearest resize, normalize, ROI crop, separable Gaussian blur, histogram equalization, tiled CLAHE-lite, `Preprocessor` chain with back-mapping to raw pixels | true CLAHE (tile blend), anti-aliased downscale, letterbox, accelerated backends |
| `calibration.py` | Camera calibration | `CameraIntrinsics`/`CameraExtrinsics` (base_link <-> optical frame), `PinholeCamera` project/unproject/pixel_to_ground/ground homography, `InversePerspectiveMapper` (BEV), normalized DLT homography + 3x4 projection, RQ decomposition, Zhang closed-form + LM refinement (K, k1, k2, poses) | corner detection from images, online extrinsic self-calibration, per-frame road pitch, parameter covariance |
| `distortion.py` | Lens distortion | Brown-Conrady (k1,k2,p1,p2,k3) distort / Newton undistort of points, `remap_bilinear`, cached `UndistortMap`, `distort_image` for sim | fisheye model, optimal new K / valid ROI, fixed-point maps |
| `environment.py` | Environment estimation | `image_stats` (brightness, contrast, range, Laplacian sharpness, saturation, gradient anisotropy); heuristic scores for night/low-light/overexposed/glare/blur/fog/rain/blocked; EMA, primary `Condition`, `visibility`; `ConditionModel` interface | learned classifier, exposure metadata, rain sensor/wiper inputs, sun prior |
| `confidence.py` | Confidence scoring | Temperature (binary + softmax), Platt (line-searched Newton), histogram binning + PAVA calibrators; ECE, Brier; aggregate (mean/geometric/harmonic/min/noisy-or); `SensorHealth` -> health; `ConfidenceScorer` (calibrated x health x visibility, with breakdown) | per-class calibration, online recalibration, conformal intervals, ensemble uncertainty |

Tests: `tests/perception/test_frontend_*.py` (41 tests, < 1 s).
