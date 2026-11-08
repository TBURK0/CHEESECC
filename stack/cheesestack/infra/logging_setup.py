"""Structured JSON logging on top of stdlib ``logging``.

``setup_logging(LoggingConfig(...))`` configures a named root (default
``"cheesestack"``) with a JSON formatter (one object per line: ts, level,
logger, msg, plus any ``extra={...}`` fields and exception text), per-module
levels (``{"cheesestack.planning": "DEBUG"}``), an optional stream handler and
an optional size-rotating file handler. It is idempotent: handlers it installed
earlier are removed before new ones are added, so tests can call it repeatedly.

The timestamp is wall time (logging is an ops concern, not algorithm code);
pass ``clock`` to stamp records with stack/sim time as ``t`` as well.

TODO(infra): ship logs over the uplink, rate limiting of repeated messages,
redaction of sensitive fields, journald/syslog handlers.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from dataclasses import dataclass, field
from typing import IO, Optional

from cheesestack.core import Clock

_STD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}
_TAG = "_cheesestack_handler"


@dataclass
class LoggingConfig:
    root: str = "cheesestack"
    level: str = "INFO"
    module_levels: dict = field(default_factory=dict)
    json: bool = True
    stream: bool = True
    file_path: Optional[str] = None
    max_bytes: int = 5_000_000
    backup_count: int = 3
    propagate: bool = False


class JsonFormatter(logging.Formatter):
    def __init__(self, clock: Optional[Clock] = None) -> None:
        super().__init__()
        self.clock = clock

    def format(self, record: logging.LogRecord) -> str:
        obj = {
            "ts": round(record.created, 6),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if self.clock is not None:
            obj["t"] = self.clock.now()
        for k, v in record.__dict__.items():
            if k not in _STD_ATTRS and not k.startswith("_"):
                obj[k] = v
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        return json.dumps(obj, default=str)


def setup_logging(config: Optional[LoggingConfig] = None, clock: Optional[Clock] = None,
                  stream: Optional[IO] = None) -> logging.Logger:
    cfg = config or LoggingConfig()
    root = logging.getLogger(cfg.root)
    for h in list(root.handlers):
        if getattr(h, _TAG, False):
            root.removeHandler(h)
            h.close()
    root.setLevel(cfg.level.upper())
    root.propagate = cfg.propagate
    fmt: logging.Formatter = JsonFormatter(clock) if cfg.json else logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers: list[logging.Handler] = []
    if cfg.stream:
        handlers.append(logging.StreamHandler(stream or sys.stderr))
    if cfg.file_path:
        handlers.append(logging.handlers.RotatingFileHandler(
            cfg.file_path, maxBytes=cfg.max_bytes, backupCount=cfg.backup_count, encoding="utf-8"))
    for h in handlers:
        h.setFormatter(fmt)
        setattr(h, _TAG, True)
        root.addHandler(h)
    for name, lvl in cfg.module_levels.items():
        logging.getLogger(name).setLevel(str(lvl).upper())
    return root


def get_logger(name: str, root: str = "cheesestack") -> logging.Logger:
    """Logger under the stack root (``get_logger("planning")`` -> cheesestack.planning)."""
    if name == root or name.startswith(root + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{root}.{name}")
