"""CI helper: import every module, then run pytest.

Used by ``.github/workflows/stack-ci.yml`` and runnable locally::

    python -m cheesestack.infra.ci            # import check + pytest
    python -m cheesestack.infra.ci --imports  # import check only

The import check walks the ``cheesestack`` package with ``pkgutil`` and imports
every module, catching syntax errors, missing deps and import cycles even in
modules no test touches. pytest runs in a subprocess so its exit code is
reported faithfully.

TODO(infra): coverage threshold, ruff/mypy when the toolchain allows it,
timing budget per test layer, JUnit XML upload.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CIConfig:
    package: str = "cheesestack"
    test_paths: list = field(default_factory=lambda: ["tests"])
    pytest_args: list = field(default_factory=lambda: ["-q"])
    exclude: list = field(default_factory=list)  # module name prefixes to skip


@dataclass
class ImportReport:
    ok: list = field(default_factory=list)
    failed: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.failed


def discover_modules(package: str) -> list[str]:
    pkg = importlib.import_module(package)
    names = [package]
    for info in pkgutil.walk_packages(pkg.__path__, prefix=package + "."):
        names.append(info.name)
    return sorted(set(names))


def check_imports(config: Optional[CIConfig] = None) -> ImportReport:
    cfg = config or CIConfig()
    rep = ImportReport()
    try:
        names = discover_modules(cfg.package)
    except Exception:
        rep.failed[cfg.package] = traceback.format_exc()
        return rep
    for name in names:
        if any(name.startswith(p) for p in cfg.exclude) or name.endswith(".__main__"):
            continue
        try:
            importlib.import_module(name)
            rep.ok.append(name)
        except Exception:  # noqa: BLE001 - report everything
            rep.failed[name] = traceback.format_exc(limit=3)
    return rep


def run_pytest(config: Optional[CIConfig] = None, cwd: Optional[str] = None) -> int:
    cfg = config or CIConfig()
    cmd = [sys.executable, "-m", "pytest", *cfg.test_paths, *cfg.pytest_args]
    return subprocess.call(cmd, cwd=cwd)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--imports", action="store_true", help="only run the import check")
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("tests", nargs="*", default=["tests"])
    a = ap.parse_args(argv)
    cfg = CIConfig(test_paths=a.tests, exclude=a.exclude)
    rep = check_imports(cfg)
    print(f"[ci] imported {len(rep.ok)} modules, {len(rep.failed)} failed")
    for name, tb in rep.failed.items():
        print(f"[ci] FAIL import {name}\n{tb}")
    if not rep.passed:
        return 1
    if a.imports:
        return 0
    root = Path(__file__).resolve().parents[2]
    return run_pytest(cfg, cwd=str(root))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
