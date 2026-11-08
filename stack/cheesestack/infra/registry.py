"""Model registry: versioned artifacts, sha256, stage promotion, rollback.

Layout (``root/``)::

    <model>/v0001/artifact.<ext>   copied artifact (immutable)
    <model>/v0001/meta.json        ModelVersion metadata (sha256, metrics, ...)
    <model>/stages.json            {"stages": {"prod": 3, ...}, "history": [...]}

``register`` copies a file (or writes bytes) into a new monotonically numbered
version and records its sha256. ``promote(name, version, "staging")`` points a
stage at a version, appending to a history log; ``rollback(name, "prod")``
restores the stage's previous target. ``load_path``/``verify`` re-hash the
artifact so a corrupted or tampered file is caught before deployment.

Stages: dev -> staging -> prod. Promotion to prod requires the version to have
been in staging first unless ``force=True`` (cheap guardrail).

Writes go through temp-file + ``os.replace``. Single writer per model assumed.
TODO(infra): signing (ed25519) and on-device verification, remote backend
(S3/GCS), OTA rollout percentages per vehicle, CoreML/ONNX format checks.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from cheesestack.core import Clock, WallClock

STAGES = ("dev", "staging", "prod")


@dataclass
class ModelRegistryConfig:
    require_staging_before_prod: bool = True


@dataclass
class ModelVersion:
    name: str
    version: int
    sha256: str
    artifact: str  # path relative to registry root
    size_bytes: int
    created: float
    metadata: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)


class RegistryError(RuntimeError):
    pass


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class ModelRegistry:
    def __init__(self, root: str | os.PathLike, config: Optional[ModelRegistryConfig] = None,
                 clock: Optional[Clock] = None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config or ModelRegistryConfig()
        self.clock = clock or WallClock()

    # -- versions -------------------------------------------------------------
    def models(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def versions(self, name: str) -> list[ModelVersion]:
        d = self.root / name
        if not d.exists():
            return []
        out = []
        for vd in sorted(d.glob("v[0-9]*")):
            meta = vd / "meta.json"
            if meta.exists():
                out.append(ModelVersion(**json.loads(meta.read_text())))
        return out

    def get(self, name: str, version: int) -> ModelVersion:
        meta = self.root / name / f"v{version:04d}" / "meta.json"
        if not meta.exists():
            raise RegistryError(f"{name} v{version} not found")
        return ModelVersion(**json.loads(meta.read_text()))

    def register(self, name: str, artifact: str | os.PathLike | bytes, *,
                 metadata: Optional[dict] = None, metrics: Optional[dict] = None,
                 ext: str = ".bin") -> ModelVersion:
        if not name or "/" in name or name.startswith("."):
            raise RegistryError(f"bad model name {name!r}")
        existing = self.versions(name)
        version = (existing[-1].version + 1) if existing else 1
        vd = self.root / name / f"v{version:04d}"
        vd.mkdir(parents=True, exist_ok=False)
        if isinstance(artifact, (bytes, bytearray)):
            dst = vd / f"artifact{ext}"
            dst.write_bytes(bytes(artifact))
        else:
            src = Path(artifact)
            dst = vd / f"artifact{src.suffix or ext}"
            shutil.copyfile(src, dst)
        mv = ModelVersion(name, version, sha256_file(dst), str(dst.relative_to(self.root)),
                          dst.stat().st_size, float(self.clock.now()),
                          dict(metadata or {}), dict(metrics or {}))
        _atomic_write(vd / "meta.json", json.dumps(asdict(mv), indent=2, sort_keys=True))
        return mv

    def verify(self, name: str, version: int) -> bool:
        mv = self.get(name, version)
        return sha256_file(self.root / mv.artifact) == mv.sha256

    def load_path(self, name: str, stage_or_version: str | int = "prod") -> Path:
        """Verified path to the artifact for a stage name or explicit version."""
        if isinstance(stage_or_version, int):
            version = stage_or_version
        else:
            version = self.stage_version(name, stage_or_version)
            if version is None:
                raise RegistryError(f"{name}: nothing in stage {stage_or_version}")
        if not self.verify(name, version):
            raise RegistryError(f"{name} v{version}: sha256 mismatch")
        return self.root / self.get(name, version).artifact

    # -- stages ------------------------------------------------------------------
    def _stages_path(self, name: str) -> Path:
        return self.root / name / "stages.json"

    def _read_stages(self, name: str) -> dict:
        p = self._stages_path(name)
        if p.exists():
            return json.loads(p.read_text())
        return {"stages": {}, "history": []}

    def stage_version(self, name: str, stage: str) -> Optional[int]:
        return self._read_stages(name)["stages"].get(stage)

    def stages(self, name: str) -> dict:
        return dict(self._read_stages(name)["stages"])

    def history(self, name: str) -> list[dict]:
        return list(self._read_stages(name)["history"])

    def promote(self, name: str, version: int, stage: str, *, force: bool = False,
                note: str = "") -> None:
        if stage not in STAGES:
            raise RegistryError(f"unknown stage {stage}; expected one of {STAGES}")
        self.get(name, version)  # existence
        if not self.verify(name, version):
            raise RegistryError(f"{name} v{version}: sha256 mismatch, refusing promotion")
        st = self._read_stages(name)
        if stage == "prod" and self.config.require_staging_before_prod and not force:
            was_staged = any(h["stage"] == "staging" and h["version"] == version for h in st["history"])
            if not was_staged:
                raise RegistryError(f"{name} v{version} was never in staging")
        st["history"].append({"stage": stage, "version": version,
                              "previous": st["stages"].get(stage),
                              "t": float(self.clock.now()), "note": note, "action": "promote"})
        st["stages"][stage] = version
        _atomic_write(self._stages_path(name), json.dumps(st, indent=2))

    def rollback(self, name: str, stage: str = "prod") -> Optional[int]:
        """Point ``stage`` back at its previous target. Returns the new version."""
        st = self._read_stages(name)
        current = st["stages"].get(stage)
        if current is None:
            raise RegistryError(f"{name}: nothing in {stage} to roll back")
        # Walk history backwards for the entry that set the current target.
        prev = None
        for h in reversed(st["history"]):
            if h["stage"] == stage and h["version"] == current and h.get("action") == "promote":
                prev = h["previous"]
                break
        if prev is None:
            st["stages"].pop(stage)
        else:
            st["stages"][stage] = prev
        st["history"].append({"stage": stage, "version": prev, "previous": current,
                              "t": float(self.clock.now()), "note": "", "action": "rollback"})
        _atomic_write(self._stages_path(name), json.dumps(st, indent=2))
        return prev
