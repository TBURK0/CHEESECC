"""Visual odometry: interface + planar baseline from matched 2D features.

Pipeline (baseline):
1. Feature matching is upstream (perception or a learned matcher); we take
   matched pixel pairs or matched ground-plane points.
2. `GroundPlaneProjector` back-projects pixels to the flat ground in
   `base_link` using a pinhole model, camera height and pitch.
3. `ransac_rigid_2d` + `umeyama_2d` estimate the SE(2) motion between frames.

For a static scene, if p0 (frame k-1) and p1 (frame k) are the same world
point in base_link at each time, then p0 = T * p1 where T = pose of base_link_k
expressed in base_link_{k-1}: the ego-motion increment. So we fit curr -> prev.
"""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cheesestack.core import Pose2D


def umeyama_2d(src, dst, with_scale: bool = False) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares similarity/rigid transform with dst ~= s * R @ src + t (Umeyama 1991)."""
    src = np.asarray(src, float).reshape(-1, 2)
    dst = np.asarray(dst, float).reshape(-1, 2)
    if len(src) < 2:
        raise ValueError("need >= 2 correspondences")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1.0
    R = U @ S @ Vt
    var_s = (xs**2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s) if with_scale and var_s > 0 else 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def _to_pose(R: np.ndarray, t: np.ndarray) -> Pose2D:
    return Pose2D(float(t[0]), float(t[1]), math.atan2(R[1, 0], R[0, 0]))


@dataclass
class RansacConfig:
    threshold: float = 0.15  # inlier residual (units of points; m on ground plane)
    iterations: int = 200
    min_inliers: int = 6
    seed: int = 0


def ransac_rigid_2d(src, dst, config: Optional[RansacConfig] = None) -> tuple[Pose2D, np.ndarray]:
    """Robust SE(2) fit. Returns (transform src->dst, inlier mask). 2-point minimal sets."""
    cfg = config or RansacConfig()
    src = np.asarray(src, float).reshape(-1, 2)
    dst = np.asarray(dst, float).reshape(-1, 2)
    n = len(src)
    if n < 2:
        raise ValueError("need >= 2 correspondences")
    rng = np.random.default_rng(cfg.seed)
    best = np.zeros(n, bool)
    best_err = np.inf
    for _ in range(cfg.iterations):
        i, j = rng.choice(n, 2, replace=False)
        if np.linalg.norm(src[i] - src[j]) < 1e-6:
            continue
        R, t, _ = umeyama_2d(src[[i, j]], dst[[i, j]])
        res = np.linalg.norm(src @ R.T + t - dst, axis=1)
        inl = res < cfg.threshold
        err = res[inl].sum()
        if inl.sum() > best.sum() or (inl.sum() == best.sum() and err < best_err):
            best, best_err = inl, err
    if best.sum() < 2:
        best = np.ones(n, bool)
    R, t, _ = umeyama_2d(src[best], dst[best])
    # One refinement pass with the final model.
    res = np.linalg.norm(src @ R.T + t - dst, axis=1)
    refined = res < cfg.threshold
    if refined.sum() >= 2:
        best = refined
        R, t, _ = umeyama_2d(src[best], dst[best])
    return _to_pose(R, t), best


@dataclass
class CameraModel:
    fx: float = 1000.0
    fy: float = 1000.0
    cx: float = 960.0
    cy: float = 540.0
    height_m: float = 1.3  # optical center above ground
    pitch_rad: float = 0.0  # positive = looking down
    mount_x_m: float = 1.5  # in base_link
    mount_y_m: float = 0.0


class GroundPlaneProjector:
    """Flat-ground inverse perspective mapping pixels -> base_link (x fwd, y left).

    Camera optical frame: x right, y down, z forward. Assumes zero roll/yaw mount.
    """

    def __init__(self, cam: CameraModel) -> None:
        self.cam = cam
        cp, sp = math.cos(cam.pitch_rad), math.sin(cam.pitch_rad)
        # optical -> vehicle-aligned (x fwd, y left, z up), then pitch down about y.
        base = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
        pitch = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], float)
        self.R = pitch @ base

    def pixels_to_ground(self, uv) -> tuple[np.ndarray, np.ndarray]:
        """Returns (points (N,2) in base_link, valid mask). Rays above the horizon are invalid."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        c = self.cam
        rays = np.stack([(uv[:, 0] - c.cx) / c.fx, (uv[:, 1] - c.cy) / c.fy, np.ones(len(uv))], axis=1)
        d = rays @ self.R.T
        valid = d[:, 2] < -1e-6
        s = np.where(valid, c.height_m / np.where(valid, -d[:, 2], 1.0), np.nan)
        pts = d[:, :2] * s[:, None] + np.array([c.mount_x_m, c.mount_y_m])
        return pts, valid

    def ground_to_pixels(self, pts) -> np.ndarray:
        pts = np.asarray(pts, float).reshape(-1, 2)
        c = self.cam
        p = np.column_stack([pts[:, 0] - c.mount_x_m, pts[:, 1] - c.mount_y_m, np.full(len(pts), -c.height_m)])
        q = p @ self.R  # vehicle -> optical (R^T applied to rows)
        return np.column_stack([c.fx * q[:, 0] / q[:, 2] + c.cx, c.fy * q[:, 1] / q[:, 2] + c.cy])


@dataclass
class VoResult:
    stamp: float
    delta: Pose2D  # motion of base_link from previous frame to current, in previous base_link
    covariance: np.ndarray  # 3x3 over (dx, dy, dyaw)
    inlier_ratio: float
    num_inliers: int
    ok: bool


class VisualOdometry(abc.ABC):
    """Interface: plug a learned/feature-based VO here (e.g. a CoreML depth+flow model)."""

    @abc.abstractmethod
    def estimate(self, stamp: float, prev_pts, curr_pts) -> VoResult: ...


@dataclass
class PlanarVoConfig:
    ransac: RansacConfig = field(default_factory=RansacConfig)
    min_inlier_ratio: float = 0.3
    min_inliers: int = 6
    use_ground_projection: bool = True


class PlanarVisualOdometry(VisualOdometry):
    """Baseline: ground-plane points + RANSAC/Umeyama SE(2)."""

    def __init__(self, config: Optional[PlanarVoConfig] = None, projector: Optional[GroundPlaneProjector] = None) -> None:
        self.config = config or PlanarVoConfig()
        self.projector = projector
        self.pose = Pose2D()  # integrated

    def estimate(self, stamp: float, prev_pts, curr_pts) -> VoResult:
        """prev_pts/curr_pts: matched (N,2). Pixels if a projector is set and enabled, else base_link meters."""
        p0 = np.asarray(prev_pts, float).reshape(-1, 2)
        p1 = np.asarray(curr_pts, float).reshape(-1, 2)
        if self.projector is not None and self.config.use_ground_projection:
            p0, v0 = self.projector.pixels_to_ground(p0)
            p1, v1 = self.projector.pixels_to_ground(p1)
            m = v0 & v1
            p0, p1 = p0[m], p1[m]
        n = len(p0)
        bad = VoResult(stamp, Pose2D(), np.eye(3) * 1e3, 0.0, 0, False)
        if n < max(2, self.config.min_inliers):
            return bad
        delta, inl = ransac_rigid_2d(p1, p0, self.config.ransac)
        k = int(inl.sum())
        ratio = k / n
        if k < self.config.min_inliers or ratio < self.config.min_inlier_ratio:
            bad.inlier_ratio, bad.num_inliers = ratio, k
            return bad
        res = delta.transform_points(p1[inl]) - p0[inl]
        sigma2 = max(float((res**2).sum() / max(1, 2 * k - 3)), 1e-6)
        # Gauss-Newton covariance: J = [I, R' * (-y, x)] per point.
        H = np.zeros((3, 3))
        for q in p1[inl]:
            rq = np.array([[math.cos(delta.yaw), -math.sin(delta.yaw)], [math.sin(delta.yaw), math.cos(delta.yaw)]]) @ q
            J = np.array([[1.0, 0.0, -rq[1]], [0.0, 1.0, rq[0]]])
            H += J.T @ J
        cov = sigma2 * np.linalg.pinv(H)
        self.pose = self.pose.compose(delta)
        return VoResult(stamp, delta, cov, ratio, k, True)
    # TODO(localization): non-planar scenes (essential matrix / PnP with depth), scale from
    # wheel odometry for monocular VO, keyframing and local bundle adjustment, rolling-shutter.
