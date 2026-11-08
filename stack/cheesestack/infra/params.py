"""Hierarchical, typed, thread-safe parameter server.

Parameters live at slash-separated paths like ``/control/lateral/kp``. Each
parameter has a fixed type set at declaration (bool/int/float/str/list); set()
coerces numerics (int -> float) and rejects other type changes. Callbacks can
be registered on an exact path or a namespace prefix (``/control``) and fire
after a successful change with ``(path, old, new)``. Callbacks run outside the
lock so they may read params; exceptions in callbacks are collected, not raised.

Limits: in-process only. TODO(infra): network sync (comms layer), persistence
to disk, per-parameter ranges/descriptions surfaced in a UI, read-only params.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

ParamCallback = Callable[[str, Any, Any], None]


@dataclass
class ParamServerConfig:
    allow_undeclared: bool = True  # set() on a new path declares it
    allow_int_to_float: bool = True


@dataclass
class ParamSpec:
    type: type
    min: Optional[float] = None
    max: Optional[float] = None
    description: str = ""


def normalize_path(path: str) -> str:
    parts = [p for p in path.strip().split("/") if p]
    if not parts:
        raise KeyError("empty parameter path")
    return "/" + "/".join(parts)


class ParamServer:
    def __init__(self, config: Optional[ParamServerConfig] = None) -> None:
        self.config = config or ParamServerConfig()
        self._lock = threading.RLock()
        self._values: dict[str, Any] = {}
        self._specs: dict[str, ParamSpec] = {}
        self._callbacks: dict[str, list[ParamCallback]] = {}
        self.callback_errors: list[tuple[str, BaseException]] = []

    # -- declaration / access -------------------------------------------------
    def declare(self, path: str, default: Any, *, min: Optional[float] = None,
                max: Optional[float] = None, description: str = "") -> Any:
        """Declare a parameter; if it already exists its current value is kept."""
        path = normalize_path(path)
        with self._lock:
            if path in self._values:
                return self._values[path]
            spec = ParamSpec(type(default), min, max, description)
            self._check_range(path, spec, default)
            self._specs[path] = spec
            self._values[path] = copy.deepcopy(default)
            return default

    def get(self, path: str, default: Any = None) -> Any:
        path = normalize_path(path)
        with self._lock:
            return copy.deepcopy(self._values.get(path, default))

    def __contains__(self, path: str) -> bool:
        with self._lock:
            return normalize_path(path) in self._values

    def spec(self, path: str) -> ParamSpec:
        return self._specs[normalize_path(path)]

    def set(self, path: str, value: Any) -> bool:
        """Set a value. Returns True if it changed. Raises TypeError/ValueError."""
        path = normalize_path(path)
        with self._lock:
            if path not in self._values:
                if not self.config.allow_undeclared:
                    raise KeyError(f"undeclared parameter {path}")
                self._specs[path] = ParamSpec(type(value))
                old = None
            else:
                old = self._values[path]
                value = self._coerce(path, value)
            self._check_range(path, self._specs[path], value)
            if path in self._values and old == value:
                return False
            self._values[path] = copy.deepcopy(value)
        self._fire(path, old, value)
        return True

    def update(self, values: dict[str, Any]) -> list[str]:
        """Set many; returns paths that changed. Validates all before applying."""
        with self._lock:
            for p, v in values.items():
                p = normalize_path(p)
                if p in self._values:
                    self._check_range(p, self._specs[p], self._coerce(p, v))
            return [p for p, v in values.items() if self.set(p, v)]

    def delete(self, path: str) -> None:
        path = normalize_path(path)
        with self._lock:
            self._values.pop(path, None)
            self._specs.pop(path, None)

    def list(self, namespace: str = "/") -> dict[str, Any]:
        """All params under ``namespace`` (inclusive), sorted by path."""
        ns = "/" if namespace.strip("/") == "" else normalize_path(namespace)
        with self._lock:
            return {p: copy.deepcopy(v) for p, v in sorted(self._values.items())
                    if ns == "/" or p == ns or p.startswith(ns + "/")}

    def tree(self, namespace: str = "/") -> dict:
        """Nested-dict view of a namespace (relative to it)."""
        ns = "/" if namespace.strip("/") == "" else normalize_path(namespace)
        out: dict = {}
        for p, v in self.list(ns).items():
            rel = p[len(ns):] if ns != "/" else p
            parts = [x for x in rel.split("/") if x]
            if not parts:
                continue
            d = out
            for k in parts[:-1]:
                d = d.setdefault(k, {})
            d[parts[-1]] = v
        return out

    def namespace(self, prefix: str) -> "ParamNamespace":
        return ParamNamespace(self, normalize_path(prefix))

    # -- callbacks --------------------------------------------------------------
    def on_change(self, path_or_ns: str, cb: ParamCallback) -> Callable[[], None]:
        """Register ``cb(path, old, new)``. Returns an unsubscribe function."""
        key = "/" if path_or_ns.strip("/") == "" else normalize_path(path_or_ns)
        with self._lock:
            self._callbacks.setdefault(key, []).append(cb)

        def cancel() -> None:
            with self._lock:
                if cb in self._callbacks.get(key, []):
                    self._callbacks[key].remove(cb)
        return cancel

    def _fire(self, path: str, old: Any, new: Any) -> None:
        with self._lock:
            cbs = []
            for key, lst in self._callbacks.items():
                if key == "/" or path == key or path.startswith(key + "/"):
                    cbs.extend(lst)
        for cb in cbs:
            try:
                cb(path, old, new)
            except Exception as e:  # noqa: BLE001 - never let a listener break set()
                self.callback_errors.append((path, e))

    # -- snapshot -----------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._values)

    def restore(self, snap: dict[str, Any], *, notify: bool = True) -> list[str]:
        """Restore a snapshot exactly (params absent from it are removed)."""
        with self._lock:
            changed = [p for p in set(self._values) | set(snap)
                       if self._values.get(p, _MISSING) != snap.get(p, _MISSING)]
            olds = {p: self._values.get(p) for p in changed}
            for p in set(self._values) - set(snap):
                self.delete(p)
            for p, v in snap.items():
                self._values[p] = copy.deepcopy(v)
                self._specs.setdefault(p, ParamSpec(type(v)))
        if notify:
            for p in sorted(changed):
                self._fire(p, olds[p], snap.get(p))
        return sorted(changed)

    # -- helpers ------------------------------------------------------------------
    def _coerce(self, path: str, value: Any) -> Any:
        tp = self._specs[path].type
        if isinstance(value, tp) and not (tp is int and isinstance(value, bool)):
            return value
        if tp is float and isinstance(value, int) and not isinstance(value, bool) \
                and self.config.allow_int_to_float:
            return float(value)
        raise TypeError(f"{path} is {tp.__name__}, got {type(value).__name__}")

    @staticmethod
    def _check_range(path: str, spec: ParamSpec, value: Any) -> None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if spec.min is not None and value < spec.min:
                raise ValueError(f"{path}={value} < min {spec.min}")
            if spec.max is not None and value > spec.max:
                raise ValueError(f"{path}={value} > max {spec.max}")


_MISSING = object()


class ParamNamespace:
    """A view of a ParamServer rooted at a prefix: ns.get("kp") -> /prefix/kp."""

    def __init__(self, server: ParamServer, prefix: str) -> None:
        self.server = server
        self.prefix = prefix

    def _p(self, name: str) -> str:
        return self.prefix + "/" + name.strip("/")

    def declare(self, name: str, default: Any, **kw) -> Any:
        return self.server.declare(self._p(name), default, **kw)

    def get(self, name: str, default: Any = None) -> Any:
        return self.server.get(self._p(name), default)

    def set(self, name: str, value: Any) -> bool:
        return self.server.set(self._p(name), value)

    def namespace(self, name: str) -> "ParamNamespace":
        return ParamNamespace(self.server, self._p(name))
