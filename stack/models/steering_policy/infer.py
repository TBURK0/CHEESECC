"""Run the DAVE-2 style steering net on one dummy camera frame.

Weights are a seeded random initialization (see model_card.md), loaded from
steering_policy.npz with the shapes in model_config.json. No torch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 26
INIT_STD = 0.02


def load_json(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def conv2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, stride: int) -> np.ndarray:
    filters, channels, kh, kw = weight.shape
    if x.shape[0] != channels:
        raise ValueError(f"conv expected {channels} channels, got {x.shape[0]}")
    _, height, width = x.shape
    out_h = (height - kh) // stride + 1
    out_w = (width - kw) // stride + 1
    if out_h < 1 or out_w < 1:
        raise ValueError("conv spatial size collapsed; check stride and kernel")
    out = np.empty((filters, out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        hs = i * stride
        for j in range(out_w):
            ws = j * stride
            patch = x[:, hs : hs + kh, ws : ws + kw]
            out[:, i, j] = np.einsum("fckl,ckl->f", weight, patch, optimize=True) + bias
    return out


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0, dtype=np.float32)


def dummy_frame() -> np.ndarray:
    """Deterministic RGB frame, CHW, float32 in [0, 1]. Shape 3x66x200."""
    y = np.linspace(0.15, 0.85, 66, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, 200, dtype=np.float32)[None, :]
    lane = np.exp(-0.5 * ((x - 0.48) / 0.04) ** 2).astype(np.float32)
    red = np.broadcast_to(0.20 + 0.45 * x, (66, 200)).astype(np.float32)
    green = np.broadcast_to(0.35 + 0.40 * y, (66, 200)).astype(np.float32) + 0.25 * lane
    blue = np.broadcast_to(0.25 + 0.15 * y, (66, 200)).astype(np.float32)
    return np.stack([red, green, blue], axis=0)


def normalize(frame: np.ndarray, stats: dict) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(stats["std"], dtype=np.float32)[:, None, None]
    return (frame.astype(np.float32) - mean) / std


def forward(frame: np.ndarray, weights: dict) -> float:
    x = frame
    for name, stride in (
        ("conv1", 2),
        ("conv2", 2),
        ("conv3", 2),
        ("conv4", 1),
        ("conv5", 1),
    ):
        x = relu(conv2d(x, weights[f"{name}.weight"], weights[f"{name}.bias"], stride))
    flat = x.reshape(-1).astype(np.float32)
    for name in ("fc1", "fc2", "fc3"):
        flat = relu(weights[f"{name}.weight"] @ flat + weights[f"{name}.bias"])
    steer = float(np.asarray(weights["fc4.weight"] @ flat + weights["fc4.bias"]).reshape(()))
    return steer


def parameter_count(weights: dict) -> int:
    return int(sum(array.size for array in weights.values()))


def write_weights(path: Path = HERE / "steering_policy.npz") -> int:
    rng = np.random.default_rng(SEED)
    specs = [
        ("conv1.weight", (24, 3, 5, 5)),
        ("conv1.bias", (24,)),
        ("conv2.weight", (36, 24, 5, 5)),
        ("conv2.bias", (36,)),
        ("conv3.weight", (48, 36, 5, 5)),
        ("conv3.bias", (48,)),
        ("conv4.weight", (64, 48, 3, 3)),
        ("conv4.bias", (64,)),
        ("conv5.weight", (64, 64, 3, 3)),
        ("conv5.bias", (64,)),
        ("fc1.weight", (100, 1152)),
        ("fc1.bias", (100,)),
        ("fc2.weight", (50, 100)),
        ("fc2.bias", (50,)),
        ("fc3.weight", (10, 50)),
        ("fc3.bias", (10,)),
        ("fc4.weight", (1, 10)),
        ("fc4.bias", (1,)),
    ]
    arrays = {}
    for name, shape in specs:
        std = 0.35 if name.startswith("fc4") else INIT_STD
        arrays[name] = rng.normal(0.0, std, size=shape).astype(np.float32)
    count = parameter_count(arrays)
    if count != 252219:
        raise RuntimeError(f"expected 252219 parameters, got {count}")
    np.savez_compressed(path, **arrays)
    return count


def main() -> None:
    config = load_json("model_config.json")
    stats = load_json("normalization_stats.json")
    weights = dict(np.load(HERE / config["weights_file"]))
    count = parameter_count(weights)
    expected = int(config["parameter_count"])
    if count != expected:
        raise RuntimeError(f"weight file has {count} parameters, config says {expected}")
    frame = dummy_frame()
    shape = (config["input"]["channels"], config["input"]["height"], config["input"]["width"])
    if frame.shape != shape:
        raise RuntimeError(f"dummy frame {frame.shape} != {shape}")
    steer = forward(normalize(frame, stats), weights)
    print(f"steering_deg: {steer:.4f}")


if __name__ == "__main__":
    main()
