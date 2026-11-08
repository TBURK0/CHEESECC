"""Layered configuration management.

Resolution order (later wins)::

    dataclass defaults  <-  JSON/TOML file  <-  environment variables  <-  explicit overrides

Configs are plain (possibly nested) dataclasses, so every component's
``<Name>Config`` can be loaded the same way. Values coming from env vars are
strings and are coerced to the field's annotated type (bool/int/float/str,
lists via JSON). Nested dataclass fields use ``__`` in env names and ``.`` in
override keys, e.g. ``CHEESE_CONTROL__KP=0.4`` or ``{"control.kp": 0.4}``.

Limits: Union types other than ``Optional[X]`` are not coerced; TOML needs
Python 3.11+ (``tomllib``) - on 3.10 a TOML file raises a clear error.
"""

from __future__ import annotations

import dataclasses
import json
import os
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Type, TypeVar

try:  # Python 3.11+
    import tomllib as _toml  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - 3.10 fallback
    _toml = None

C = TypeVar("C")


class ConfigError(ValueError):
    """Raised for unknown keys, bad types or failed validation."""


@dataclass
class ConfigLoaderConfig:
    env_prefix: str = "CHEESE_"
    nested_sep_env: str = "__"
    nested_sep_key: str = "."
    strict: bool = True  # unknown keys raise instead of being ignored


def to_dict(cfg: Any) -> dict:
    """Dataclass -> plain dict (recursively), suitable for JSON."""
    if not is_dataclass(cfg):
        raise TypeError("to_dict expects a dataclass instance")
    return dataclasses.asdict(cfg)


def _resolved_hints(cls: type) -> dict:
    try:
        return typing.get_type_hints(cls)
    except Exception:  # pragma: no cover - forward refs that cannot resolve
        return {f.name: f.type for f in fields(cls)}


def _unwrap_optional(tp: Any) -> Any:
    origin = typing.get_origin(tp)
    if origin is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def coerce(value: Any, tp: Any) -> Any:
    """Coerce ``value`` (often a string from env) to annotated type ``tp``."""
    tp = _unwrap_optional(tp)
    if value is None:
        return None
    if tp is Any:
        return value
    if tp is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        s = str(value).strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
        raise ConfigError(f"cannot parse bool from {value!r}")
    if tp in (int, float, str):
        try:
            if tp is int and isinstance(value, float) and not value.is_integer():
                raise ConfigError(f"refusing lossy float->int for {value!r}")
            return tp(value)
        except (TypeError, ValueError) as e:
            raise ConfigError(f"cannot coerce {value!r} to {tp.__name__}") from e
    origin = typing.get_origin(tp)
    if origin in (list, tuple):
        if isinstance(value, str):
            value = json.loads(value)
        args = typing.get_args(tp)
        inner = args[0] if args else Any
        seq = [coerce(v, inner) for v in value]
        return seq if origin is list else tuple(seq)
    if origin is dict:
        if isinstance(value, str):
            value = json.loads(value)
        return dict(value)
    if isinstance(tp, type) and is_dataclass(tp):
        if isinstance(value, tp):
            return value
        if isinstance(value, Mapping):
            return from_dict(tp, value)
    return value


def from_dict(cls: Type[C], data: Mapping[str, Any], strict: bool = True) -> C:
    """Build dataclass ``cls`` from a (nested) mapping; missing keys keep defaults."""
    hints = _resolved_hints(cls)
    names = {f.name for f in fields(cls) if f.init}
    unknown = set(data) - names
    if unknown and strict:
        raise ConfigError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {k: coerce(v, hints.get(k, Any)) for k, v in data.items() if k in names}
    return cls(**kwargs)


def deep_merge(base: dict, upd: Mapping) -> dict:
    """Recursively merge ``upd`` into a copy of ``base``."""
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_file(path: str | os.PathLike) -> dict:
    """Read a JSON or TOML file into a dict (by extension)."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".json":
        return json.loads(p.read_text())
    if suffix == ".toml":
        if _toml is None:  # pragma: no cover
            raise ConfigError("TOML config requires Python 3.11+ (tomllib)")
        with p.open("rb") as fh:
            return _toml.load(fh)
    raise ConfigError(f"unsupported config file type: {p.suffix}")


def _set_path(d: dict, path: list[str], value: Any) -> None:
    for key in path[:-1]:
        d = d.setdefault(key, {})
        if not isinstance(d, dict):
            raise ConfigError(f"cannot set nested key under non-mapping {key!r}")
    d[path[-1]] = value


def _field_paths(cls: type, prefix: tuple = ()) -> dict[tuple, Any]:
    """All leaf field paths of a nested dataclass type -> annotated type."""
    out: dict[tuple, Any] = {}
    for name, tp in _resolved_hints(cls).items():
        base = _unwrap_optional(tp)
        if isinstance(base, type) and is_dataclass(base):
            out.update(_field_paths(base, prefix + (name,)))
        else:
            out[prefix + (name,)] = tp
    return out


def env_layer(cls: type, environ: Mapping[str, str], cfg: Optional[ConfigLoaderConfig] = None) -> dict:
    """Extract the env-var layer for dataclass ``cls`` (only known fields)."""
    cfg = cfg or ConfigLoaderConfig()
    out: dict = {}
    for path, _tp in _field_paths(cls).items():
        name = cfg.env_prefix + cfg.nested_sep_env.join(p.upper() for p in path)
        if name in environ:
            _set_path(out, list(path), environ[name])
    return out


Validator = Callable[[Any], None]


class ConfigLoader:
    """Resolve a config dataclass from defaults, file, env and overrides.

    ``validators`` are callables that raise (``ConfigError``/``ValueError``) on
    invalid configs. A dataclass may also define ``validate(self)``.
    """

    def __init__(self, cls: Type[C], config: Optional[ConfigLoaderConfig] = None,
                 validators: Optional[list[Validator]] = None) -> None:
        if not is_dataclass(cls):
            raise TypeError("ConfigLoader expects a dataclass type")
        self.cls = cls
        self.config = config or ConfigLoaderConfig()
        self.validators = list(validators or [])
        self.sources: list[str] = []

    def load(self, path: Optional[str | os.PathLike] = None,
             environ: Optional[Mapping[str, str]] = None,
             overrides: Optional[Mapping[str, Any]] = None) -> C:
        self.sources = ["defaults"]
        data = to_dict(self.cls())
        if path is not None:
            data = deep_merge(data, load_file(path))
            self.sources.append(f"file:{path}")
        env = os.environ if environ is None else environ
        ev = env_layer(self.cls, env, self.config)
        if ev:
            data = deep_merge(data, ev)
            self.sources.append("env")
        if overrides:
            ov: dict = {}
            for k, v in overrides.items():
                _set_path(ov, k.split(self.config.nested_sep_key), v)
            data = deep_merge(data, ov)
            self.sources.append("overrides")
        obj = self._build(self.cls, data)
        self.validate(obj)
        return obj

    def _build(self, cls: type, data: Mapping) -> Any:
        hints = _resolved_hints(cls)
        names = {f.name for f in fields(cls) if f.init}
        unknown = set(data) - names
        if unknown and self.config.strict:
            raise ConfigError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
        kwargs = {}
        for k, v in data.items():
            if k not in names:
                continue
            tp = _unwrap_optional(hints.get(k, Any))
            if isinstance(tp, type) and is_dataclass(tp) and isinstance(v, Mapping):
                kwargs[k] = self._build(tp, v)
            else:
                kwargs[k] = coerce(v, hints.get(k, Any))
        return cls(**kwargs)

    def validate(self, obj: Any) -> None:
        own = getattr(obj, "validate", None)
        try:
            if callable(own):
                own()
            for v in self.validators:
                v(obj)
        except ConfigError:
            raise
        except ValueError as e:
            raise ConfigError(str(e)) from e


def save(cfg: Any, path: str | os.PathLike) -> None:
    """Write a dataclass config as JSON (TOML writing is not in stdlib)."""
    # TODO(infra): TOML writer, comments/provenance per key, schema export.
    Path(path).write_text(json.dumps(to_dict(cfg), indent=2, sort_keys=True))


def load_config(cls: Type[C], path=None, environ=None, overrides=None, **kw) -> C:
    """Convenience wrapper around ``ConfigLoader(cls).load(...)``."""
    return ConfigLoader(cls, **kw).load(path=path, environ=environ, overrides=overrides)
