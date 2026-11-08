"""Coordinate-frame registry (TF-tree-like) using core.Pose2D.

Each frame has at most one parent. An edge stores T_parent_child: the pose of
`child` expressed in `parent`, so ``T.transform_point(p_child) == p_parent``.
Edges are static (single Pose2D) or dynamic (time-stamped, interpolated).

`lookup(target, source, t)` returns T_target_source by walking both frames to
their lowest common ancestor and composing the chain.
Standard frames (CONVENTIONS.md): map, odom, base_link, camera_<name>, imu, gps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from cheesestack.core import Pose2D

from .time_sync import InterpolationBuffer


class FrameLookupError(KeyError):
    pass


@dataclass
class FrameTreeConfig:
    buffer_s: float = 10.0
    max_extrapolation_s: float = 0.1


class FrameTree:
    def __init__(self, config: Optional[FrameTreeConfig] = None) -> None:
        self.config = config or FrameTreeConfig()
        self._parent: dict[str, str] = {}
        self._static: dict[str, Pose2D] = {}
        self._dynamic: dict[str, InterpolationBuffer] = {}

    @property
    def frames(self) -> set[str]:
        return set(self._parent) | set(self._parent.values())

    def _check_cycle(self, parent: str, child: str) -> None:
        f: Optional[str] = parent
        while f is not None:
            if f == child:
                raise ValueError(f"adding {parent}->{child} would create a cycle")
            f = self._parent.get(f)

    def set_static(self, parent: str, child: str, T: Pose2D) -> None:
        if self._parent.get(child, parent) != parent:
            raise ValueError(f"{child} already has parent {self._parent[child]}")
        self._check_cycle(parent, child)
        self._parent[child] = parent
        self._static[child] = T
        self._dynamic.pop(child, None)

    def set_transform(self, parent: str, child: str, T: Pose2D, stamp: float) -> None:
        """Add a time-stamped sample for a dynamic edge (e.g. odom -> base_link)."""
        if self._parent.get(child, parent) != parent:
            raise ValueError(f"{child} already has parent {self._parent[child]}")
        if child not in self._dynamic:
            self._check_cycle(parent, child)
            self._dynamic[child] = InterpolationBuffer(self.config.buffer_s,
                                                       max_extrapolation_s=self.config.max_extrapolation_s)
            self._static.pop(child, None)
        self._parent[child] = parent
        self._dynamic[child].add(stamp, T)

    def _edge(self, child: str, t: Optional[float]) -> Pose2D:
        if child in self._static:
            return self._static[child]
        buf = self._dynamic[child]
        if t is None:
            return buf.interpolate(buf.span[1])
        T = buf.interpolate(t)
        if T is None:
            raise FrameLookupError(f"{self._parent[child]}->{child} unavailable at t={t} (span {buf.span})")
        return T

    def _to_root(self, frame: str, t: Optional[float]) -> tuple[list[str], list[Pose2D]]:
        chain, poses = [frame], []
        while chain[-1] in self._parent:
            poses.append(self._edge(chain[-1], t))
            chain.append(self._parent[chain[-1]])
        return chain, poses

    def lookup(self, target: str, source: str, t: Optional[float] = None) -> Pose2D:
        """T_target_source at time t (latest if None)."""
        if target == source:
            return Pose2D()
        if target not in self.frames or source not in self.frames:
            raise FrameLookupError(f"unknown frame: {target if target not in self.frames else source}")
        cs, ps = self._to_root(source, t)
        ct, pt = self._to_root(target, t)
        common = next((f for f in cs if f in set(ct)), None)
        if common is None:
            raise FrameLookupError(f"{target} and {source} are not connected")
        T_c_s = Pose2D()
        for p in reversed(ps[:cs.index(common)]):
            T_c_s = T_c_s.compose(p)  # build from common down to source
        T_c_t = Pose2D()
        for p in reversed(pt[:ct.index(common)]):
            T_c_t = T_c_t.compose(p)
        return T_c_t.inverse().compose(T_c_s)

    def transform_points(self, target: str, source: str, pts, t: Optional[float] = None) -> np.ndarray:
        return self.lookup(target, source, t).transform_points(pts)
    # TODO(localization): 3D (SE3) edges for camera extrinsics with roll/pitch, serialization of the
    # static tree from a vehicle config file, and /tf-style bus publication.
