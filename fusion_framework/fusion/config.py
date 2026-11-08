"""Config loading plus a guard against unmeasured values.

    cfg = load_config()
    require(cfg, "imu.gyro_noise", "imu.rate_hz")   # raises if still null
"""
from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "fusion.yaml"


class NotMeasuredError(RuntimeError):
    pass


def load_config(path=None) -> dict:
    with open(path or DEFAULT_PATH) as f:
        return yaml.safe_load(f)


def get(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def require(cfg: dict, *keys: str):
    """Raise if any listed value is still null (not yet measured)."""
    missing = [k for k in keys if get(cfg, k) is None]
    if missing:
        raise NotMeasuredError("Not measured yet: " + ", ".join(missing) +
                               " (see config/fusion.yaml for the test step)")
    return [get(cfg, k) for k in keys]


def unmeasured(cfg: dict, prefix: str = "") -> list[str]:
    """All dotted keys whose value is null."""
    out = []
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if v is None:
            out.append(key)
        elif isinstance(v, dict):
            out.extend(unmeasured(v, key + "."))
    return out
