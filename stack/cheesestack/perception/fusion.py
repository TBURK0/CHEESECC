"""Generic multi-source detection fusion.

Sources (camera detector, radar, another phone camera, V2X ...) are
registered in a :class:`SourceRegistry` with a default measurement
covariance, a range-dependent noise growth, a reliability weight and a
latency. Each fusion cycle :class:`DetectionFuser`:

1. converts every core.Detection with a `box3d` into a
   :class:`FusionMeasurement` (position + 2x2 covariance; covariance from
   `attributes["covariance"]` or the source default, grown with range),
2. clusters measurements across sources: sources are processed in turn,
   each one's measurements are associated to existing clusters with the
   Hungarian algorithm on Mahalanobis distance (gated by chi-square),
   at most one measurement per source per cluster,
3. merges each cluster: information-weighted mean (P = (sum Pi^-1)^-1,
   x = P sum Pi^-1 xi), or covariance intersection when sources may be
   correlated; class by reliability*score-weighted vote; existence score
   by noisy-OR of reliability*score.

Output: one core.Detection per cluster with attributes `sources`,
`covariance`, `n_sources`.

Limits: no out-of-sequence handling beyond latency compensation by a
caller-provided velocity, no track-level (T2T) fusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from cheesestack.core import BoundingBox3D, Detection, Header, ObjectClass, Pose2D

from .tracking import INF_COST, hungarian


@dataclass
class SourceInfo:
    name: str
    position_std: float = 0.5  # m, at zero range
    range_std_gain: float = 0.02  # extra std per metre of range
    reliability: float = 1.0  # [0,1]
    latency_s: float = 0.0
    provides_class: bool = True


class SourceRegistry:
    def __init__(self) -> None:
        self._sources: dict[str, SourceInfo] = {}

    def register(self, info: SourceInfo) -> None:
        self._sources[info.name] = info

    def get(self, name: str) -> SourceInfo:
        if name not in self._sources:
            raise KeyError(f"unregistered fusion source {name!r}")
        return self._sources[name]

    def __contains__(self, name: str) -> bool:
        return name in self._sources

    def names(self) -> list[str]:
        return list(self._sources)


@dataclass
class FusionMeasurement:
    source: str
    stamp: float
    cls: ObjectClass
    score: float
    z: np.ndarray  # (2,)
    R: np.ndarray  # (2,2)
    detection: Detection


@dataclass
class FusionConfig:
    gate_chi2: float = 9.21
    use_covariance_intersection: bool = False
    min_score: float = 0.05
    frame_id: str = "base_link"


@dataclass
class _Cluster:
    members: list = field(default_factory=list)
    x: np.ndarray = None
    P: np.ndarray = None


def information_merge(zs: Sequence[np.ndarray], Rs: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Independent-measurement fusion (product of Gaussians)."""
    info = sum(np.linalg.inv(R) for R in Rs)
    P = np.linalg.inv(info)
    x = P @ sum(np.linalg.inv(R) @ z for z, R in zip(zs, Rs))
    return x, P


def covariance_intersection(zs: Sequence[np.ndarray], Rs: Sequence[np.ndarray], n_grid: int = 21
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Covariance intersection (consistent under unknown correlation). Weights by trace-min
    grid search for 2 inputs; for >2 inputs, weights proportional to 1/trace."""
    if len(zs) == 1:
        return zs[0].copy(), Rs[0].copy()
    if len(zs) == 2:
        best = None
        for w in np.linspace(0.0, 1.0, n_grid):
            info = w * np.linalg.inv(Rs[0]) + (1 - w) * np.linalg.inv(Rs[1])
            if np.linalg.det(info) <= 0:
                continue
            P = np.linalg.inv(info)
            if best is None or np.trace(P) < best[0]:
                x = P @ (w * np.linalg.inv(Rs[0]) @ zs[0] + (1 - w) * np.linalg.inv(Rs[1]) @ zs[1])
                best = (np.trace(P), x, P)
        return best[1], best[2]
    ws = np.array([1.0 / np.trace(R) for R in Rs])
    ws /= ws.sum()
    info = sum(w * np.linalg.inv(R) for w, R in zip(ws, Rs))
    P = np.linalg.inv(info)
    x = P @ sum(w * np.linalg.inv(R) @ z for w, z, R in zip(ws, zs, Rs))
    return x, P


class DetectionFuser:
    def __init__(self, registry: SourceRegistry, config: Optional[FusionConfig] = None) -> None:
        self.registry = registry
        self.config = config or FusionConfig()

    def to_measurement(self, d: Detection, source: Optional[str] = None,
                       stamp: Optional[float] = None) -> Optional[FusionMeasurement]:
        if d.box3d is None:
            return None
        src = source or d.header.source
        info = self.registry.get(src)
        z = np.array([d.box3d.pose.x, d.box3d.pose.y], float)
        if "covariance" in d.attributes:
            R = np.asarray(d.attributes["covariance"], float)[:2, :2]
        else:
            std = info.position_std + info.range_std_gain * float(np.hypot(*z))
            R = np.eye(2) * std ** 2
        vel = d.attributes.get("velocity")
        t = d.header.stamp
        if stamp is not None and vel is not None:
            # latency / time alignment with a constant-velocity extrapolation
            dt = stamp - (t - info.latency_s)
            z = z + np.asarray(vel, float)[:2] * dt
        return FusionMeasurement(src, t, d.cls, float(d.score), z, R, d)

    def fuse(self, detections: Sequence[Detection], stamp: float) -> list[Detection]:
        """Fuse detections from any number of sources (source = header.source)."""
        cfg = self.config
        by_src: dict[str, list[FusionMeasurement]] = {}
        for d in detections:
            m = self.to_measurement(d, stamp=stamp)
            if m is not None:
                by_src.setdefault(m.source, []).append(m)
        # most reliable source first seeds clusters
        order = sorted(by_src, key=lambda s: -self.registry.get(s).reliability)
        clusters: list[_Cluster] = []
        for src in order:
            ms = by_src[src]
            C = np.full((len(clusters), len(ms)), INF_COST)
            for i, cl in enumerate(clusters):
                for j, m in enumerate(ms):
                    S = cl.P + m.R
                    y = m.z - cl.x
                    d2 = float(y @ np.linalg.solve(S, y))
                    if d2 <= cfg.gate_chi2:
                        C[i, j] = d2
            used = set()
            for i, j in hungarian(C):
                if C[i, j] >= INF_COST:
                    continue
                clusters[i].members.append(ms[j])
                self._merge(clusters[i])
                used.add(j)
            for j, m in enumerate(ms):
                if j not in used:
                    clusters.append(_Cluster(members=[m], x=m.z.copy(), P=m.R.copy()))
        out = []
        for cl in clusters:
            det = self._emit(cl, stamp)
            if det.score >= cfg.min_score:
                out.append(det)
        out.sort(key=lambda d: -d.score)
        return out

    def _merge(self, cl: _Cluster) -> None:
        zs = [m.z for m in cl.members]
        Rs = [m.R for m in cl.members]
        fn = covariance_intersection if self.config.use_covariance_intersection else information_merge
        cl.x, cl.P = fn(zs, Rs)

    def _emit(self, cl: _Cluster, stamp: float) -> Detection:
        votes: dict[ObjectClass, float] = {}
        miss = 1.0
        best_box = None
        best_w = -1.0
        for m in cl.members:
            info = self.registry.get(m.source)
            w = info.reliability * m.score
            if info.provides_class and m.cls != ObjectClass.UNKNOWN:
                votes[m.cls] = votes.get(m.cls, 0.0) + w
            miss *= 1.0 - np.clip(w, 0.0, 1.0)
            if w > best_w:
                best_w, best_box = w, m.detection.box3d
        cls = max(votes, key=votes.get) if votes else ObjectClass.UNKNOWN
        box = BoundingBox3D(Pose2D(float(cl.x[0]), float(cl.x[1]), best_box.pose.yaw),
                            best_box.length, best_box.width, best_box.height)
        return Detection(Header(stamp=stamp, frame_id=self.config.frame_id, source="perception.fusion"),
                         cls, float(1.0 - miss), box3d=box,
                         attributes={"sources": sorted({m.source for m in cl.members}),
                                     "n_sources": len({m.source for m in cl.members}),
                                     "covariance": cl.P.copy(),
                                     "class_votes": {k.value: v for k, v in votes.items()}})
        # TODO(perception): velocity/extent fusion, track-to-track fusion, per-class gating, OOSM handling.
