# cheesestack conventions

Python 3.10+, `numpy` only (plus stdlib). No scipy, torch, opencv. ML models are
defined behind an interface with a classical/numpy baseline so everything runs
and tests on any machine. Heavy backends (ONNX, CoreML, torch) plug in later
behind the same interface.

## Layout
- `cheesestack/<layer>/` one subpackage per layer, one module per capability.
- `tests/<layer>/test_*.py` pytest, fast (< a few seconds per layer), deterministic (seed RNGs).
- Each layer has a short `README.md` listing modules and what is real vs. TODO.

## Dependencies between layers
- Every layer may import `cheesestack.core`. Layers do NOT import each other
  (keeps them independently buildable). Integration lives in `cheesestack/pipeline.py`.
- Put shared message types in `cheesestack/core/types.py` (owner: lead). If a layer
  needs a new local type, define it inside the layer.

## Units and frames
- SI units everywhere: m, s, rad, m/s, m/s^2, rad/s, kg, N. Suffix non-SI names (`_deg`, `_kph`).
- Planar frames: x forward, y left, yaw CCW from +x, angles wrapped with `core.wrap_angle`.
- Frames: `map` (ENU local tangent plane), `odom`, `base_link` (rear-axle center), `camera_<name>`, `imu`, `gps`.
- Image coords: (u, v) pixels, u right, v down. Images are `np.ndarray` HxW or HxWx3, float32 [0,1] or uint8.
- Time: float seconds from a `core.Clock`. Never call `time.time()` in algorithm code.

## Style
- Dataclasses for messages and configs. Each component takes a `<Name>Config` dataclass with defaults.
- Components that run on a tick subclass `core.Node` or expose a pure `update(...)`/`process(...)` method.
- Docstrings say what the algorithm is and its limits. Mark gaps with `# TODO(<layer>):`.
- Confidence values are floats in [0, 1].
