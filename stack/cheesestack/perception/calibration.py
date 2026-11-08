"""Pinhole camera model, ground-plane homography / IPM, and calibration from correspondences.

Frames: `base_link` is x forward, y left, z up (CONVENTIONS.md). The camera
*optical* frame is z forward, x right, y down. `CameraExtrinsics` stores the
camera mount pose in base_link as translation + (roll, pitch, yaw) applied as
R = Rz(yaw) Ry(pitch) Rx(roll) on top of the fixed body->optical rotation, so
pitch > 0 tilts the camera *down* toward the road.

Calibration:
- `estimate_homography`: normalized DLT (Hartley), >= 4 points.
- `estimate_projection_dlt`: 3x4 P from >= 6 non-coplanar 3D-2D points, plus
  `decompose_projection` (RQ) -> K, R, t.
- `calibrate_zhang`: Zhang's closed-form intrinsics from >= 3 planar views,
  per-view extrinsics, then optional Levenberg-Marquardt refinement of
  (fx, fy, cx, cy, k1, k2, per-view pose) on reprojection error.

# TODO(perception): corner detection (checkerboard/ChArUco) from images, online
#   extrinsic self-calibration from lane vanishing points, road-pitch estimation
#   per frame for IPM, rolling-shutter model, covariance of calibrated params.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .distortion import BrownConrady, remap_bilinear

# body (x fwd, y left, z up) <- optical (x right, y down, z fwd)
R_BODY_FROM_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Axis-angle (3,) -> rotation matrix."""
    rvec = np.asarray(rvec, dtype=float)
    th = np.linalg.norm(rvec)
    if th < 1e-12:
        return np.eye(3)
    k = rvec / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def rodrigues_inv(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle (3,)."""
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if math.pi - th < 1e-6:
        # near pi: axis from the symmetric part
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.maximum(np.diag(A), 0))
        i = int(np.argmax(k))
        k = A[:, i] / max(k[i], 1e-12)
        return k / np.linalg.norm(k) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th))
    return w * th


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 0
    height: int = 0
    skew: float = 0.0

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, self.skew, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @classmethod
    def from_K(cls, K: np.ndarray, width: int = 0, height: int = 0) -> "CameraIntrinsics":
        return cls(K[0, 0], K[1, 1], K[0, 2], K[1, 2], width, height, K[0, 1])

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_rad: float) -> "CameraIntrinsics":
        fx = (width / 2) / math.tan(hfov_rad / 2)
        return cls(fx, fx, (width - 1) / 2, (height - 1) / 2, width, height)

    def scaled(self, sx: float, sy: Optional[float] = None) -> "CameraIntrinsics":
        """Intrinsics after resizing the image by (sx, sy) (pixel-center convention)."""
        sy = sx if sy is None else sy
        return CameraIntrinsics(self.fx * sx, self.fy * sy, (self.cx + 0.5) * sx - 0.5, (self.cy + 0.5) * sy - 0.5,
                                int(round(self.width * sx)), int(round(self.height * sy)), self.skew * sx)


@dataclass
class CameraExtrinsics:
    """Camera mount pose in base_link (m, rad)."""

    x: float = 1.5
    y: float = 0.0
    z: float = 1.3
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    @property
    def R_base_cam(self) -> np.ndarray:
        """Rotation taking optical-frame vectors to base_link."""
        return rot_z(self.yaw) @ rot_y(self.pitch) @ rot_x(self.roll) @ R_BODY_FROM_OPTICAL

    @property
    def t_base_cam(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @classmethod
    def from_Rt(cls, R_base_cam: np.ndarray, t_base_cam: np.ndarray) -> "CameraExtrinsics":
        M = R_base_cam @ R_BODY_FROM_OPTICAL.T  # = Rz Ry Rx
        pitch = math.asin(-np.clip(M[2, 0], -1, 1))
        roll = math.atan2(M[2, 1], M[2, 2])
        yaw = math.atan2(M[1, 0], M[0, 0])
        t = np.asarray(t_base_cam, float)
        return cls(float(t[0]), float(t[1]), float(t[2]), roll, pitch, yaw)


@dataclass
class PinholeCamera:
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics = field(default_factory=CameraExtrinsics)
    distortion: Optional[BrownConrady] = None
    name: str = "camera_front"

    @property
    def K(self) -> np.ndarray:
        return self.intrinsics.K

    @property
    def P(self) -> np.ndarray:
        """3x4 projection from base_link points to pixels (ignores distortion)."""
        R, t = self.extrinsics.R_base_cam, self.extrinsics.t_base_cam
        return self.K @ np.hstack([R.T, (-R.T @ t)[:, None]])

    def base_to_cam(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, float).reshape(-1, 3)
        R, t = self.extrinsics.R_base_cam, self.extrinsics.t_base_cam
        return (pts - t) @ R

    def cam_to_base(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, float).reshape(-1, 3)
        R, t = self.extrinsics.R_base_cam, self.extrinsics.t_base_cam
        return pts @ R.T + t

    def project(self, pts_base: np.ndarray, min_depth: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        """base_link 3D points (N,3) -> pixels (N,2) and a validity mask (in front of camera
        and inside the image if width/height known)."""
        pc = self.base_to_cam(pts_base)
        z = pc[:, 2]
        front = z > min_depth
        zs = np.where(front, z, 1.0)
        xy = pc[:, :2] / zs[:, None]
        if self.distortion is not None and not self.distortion.is_identity:
            xy = self.distortion.distort_normalized(xy)
        K = self.K
        uv = np.stack([K[0, 0] * xy[:, 0] + K[0, 1] * xy[:, 1] + K[0, 2], K[1, 1] * xy[:, 1] + K[1, 2]], axis=1)
        valid = front.copy()
        w, h = self.intrinsics.width, self.intrinsics.height
        if w and h:
            valid &= (uv[:, 0] >= -0.5) & (uv[:, 0] <= w - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] <= h - 0.5)
        return uv, valid

    def unproject(self, uv: np.ndarray) -> np.ndarray:
        """Pixels (N,2) -> unit ray directions in base_link (N,3)."""
        uv = np.asarray(uv, float).reshape(-1, 2)
        K = self.K
        y = (uv[:, 1] - K[1, 2]) / K[1, 1]
        x = (uv[:, 0] - K[0, 2] - K[0, 1] * y) / K[0, 0]
        xy = np.stack([x, y], axis=1)
        if self.distortion is not None and not self.distortion.is_identity:
            xy = self.distortion.undistort_normalized(xy)
        rays = np.hstack([xy, np.ones((len(xy), 1))]) @ self.extrinsics.R_base_cam.T
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    def pixel_to_ground(self, uv: np.ndarray, ground_z: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Intersect pixel rays with the plane z = ground_z (flat-world). Returns (N,2) xy and validity
        (ray hits ground in front of the camera)."""
        rays = self.unproject(uv)
        t0 = self.extrinsics.t_base_cam
        dz = rays[:, 2]
        ok = dz < -1e-9
        s = np.where(ok, (ground_z - t0[2]) / np.where(ok, dz, -1.0), np.nan)
        pts = t0[None, :2] + s[:, None] * rays[:, :2]
        return pts, ok & (s > 0)

    def ground_homography(self, ground_z: float = 0.0) -> np.ndarray:
        """H (3x3) with [u v 1]^T ~ H [x y 1]^T for ground points (z = ground_z), no distortion."""
        P = self.P
        return np.column_stack([P[:, 0], P[:, 1], P[:, 2] * ground_z + P[:, 3]])


@dataclass
class IPMConfig:
    x_range: tuple[float, float] = (3.0, 40.0)  # forward (m)
    y_range: tuple[float, float] = (-8.0, 8.0)  # left (m)
    resolution: float = 0.1  # m per BEV pixel
    fill: float = 0.0


class InversePerspectiveMapper:
    """Bird's-eye view warp via the ground homography. Output row 0 = farthest x,
    column 0 = most-left y (so the BEV image looks like a top-down map with ego at bottom)."""

    def __init__(self, camera: PinholeCamera, config: Optional[IPMConfig] = None) -> None:
        self.camera = camera
        self.config = config or IPMConfig()
        c = self.config
        nx = int(round((c.x_range[1] - c.x_range[0]) / c.resolution))
        ny = int(round((c.y_range[1] - c.y_range[0]) / c.resolution))
        xs = c.x_range[1] - (np.arange(nx) + 0.5) * c.resolution
        ys = c.y_range[1] - (np.arange(ny) + 0.5) * c.resolution
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        self.xy = np.stack([X, Y], axis=-1)
        pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
        uv, valid = camera.project(pts)
        uv[~valid] = np.nan
        self.map_u = uv[:, 0].reshape(nx, ny)
        self.map_v = uv[:, 1].reshape(nx, ny)
        self.shape = (nx, ny)

    def warp(self, img: np.ndarray) -> np.ndarray:
        return remap_bilinear(img, self.map_u, self.map_v, self.config.fill)

    def bev_to_base(self, rc: np.ndarray) -> np.ndarray:
        """BEV (row, col) -> base_link (x, y)."""
        rc = np.asarray(rc, float).reshape(-1, 2)
        c = self.config
        return np.stack([c.x_range[1] - (rc[:, 0] + 0.5) * c.resolution,
                         c.y_range[1] - (rc[:, 1] + 0.5) * c.resolution], axis=1)


# ----------------------------------------------------------------------------- DLT / Zhang


def _normalize_pts(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalization: centroid to origin, mean distance sqrt(dim)."""
    p = np.asarray(p, float)
    d = p.shape[1]
    c = p.mean(axis=0)
    md = np.mean(np.linalg.norm(p - c, axis=1))
    s = math.sqrt(d) / md if md > 1e-12 else 1.0
    T = np.eye(d + 1)
    T[:d, :d] *= s
    T[:d, d] = -s * c
    ph = np.hstack([p, np.ones((len(p), 1))]) @ T.T
    return ph, T


def estimate_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Normalized DLT: H with dst ~ H src. src, dst: (N,2), N >= 4."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError("need >= 4 matching points")
    s, Ts = _normalize_pts(src)
    d, Td = _normalize_pts(dst)
    A = []
    for (x, y, w), (u, v, q) in zip(s, d):
        A.append([0, 0, 0, -q * x, -q * y, -q * w, v * x, v * y, v * w])
        A.append([q * x, q * y, q * w, 0, 0, 0, -u * x, -u * y, -u * w])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float).reshape(-1, 2)
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return ph[:, :2] / ph[:, 2:3]


def estimate_projection_dlt(X: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """3x4 projection matrix from >= 6 non-coplanar world points (N,3) and pixels (N,2)."""
    X, uv = np.asarray(X, float), np.asarray(uv, float)
    if len(X) < 6:
        raise ValueError("need >= 6 points")
    Xn, TX = _normalize_pts(X)
    un, Tu = _normalize_pts(uv)
    A = []
    for Xi, (u, v, w) in zip(Xn, un):
        A.append(np.concatenate([np.zeros(4), -w * Xi, v * Xi]))
        A.append(np.concatenate([w * Xi, np.zeros(4), -u * Xi]))
    _, _, Vt = np.linalg.svd(np.asarray(A))
    Pn = Vt[-1].reshape(3, 4)
    return np.linalg.inv(Tu) @ Pn @ TX


def rq(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RQ decomposition of a 3x3 matrix: M = R @ Q, R upper-triangular with positive diagonal."""
    Q, R = np.linalg.qr(np.flipud(M).T)
    R = np.flipud(R.T)[:, ::-1]
    Q = Q.T[::-1, :]
    sg = np.sign(np.diag(R))
    sg[sg == 0] = 1
    S = np.diag(sg)
    return R @ S, S @ Q


def decompose_projection(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P = K [R | t] -> (K normalized K[2,2]=1, R (det +1), t)."""
    P = np.asarray(P, float)
    if np.linalg.det(P[:, :3]) < 0:
        P = -P
    K, R = rq(P[:, :3])
    t = np.linalg.solve(K, P[:, 3])
    scale = K[2, 2]
    return K / scale, R, t


def checkerboard_points(rows: int, cols: int, square: float) -> np.ndarray:
    """Inner-corner grid on the board plane (Z=0), (rows*cols, 3), row-major."""
    j, i = np.meshgrid(np.arange(cols), np.arange(rows))
    return np.column_stack([j.ravel() * square, i.ravel() * square, np.zeros(rows * cols)])


def _v_ij(H: np.ndarray, i: int, j: int) -> np.ndarray:
    hi, hj = H[:, i], H[:, j]
    return np.array([hi[0] * hj[0], hi[0] * hj[1] + hi[1] * hj[0], hi[1] * hj[1],
                     hi[2] * hj[0] + hi[0] * hj[2], hi[2] * hj[1] + hi[1] * hj[2], hi[2] * hj[2]])


def zhang_intrinsics(homographies: Sequence[np.ndarray]) -> np.ndarray:
    """Closed-form K from >= 3 plane->image homographies (Zhang 2000, appendix B)."""
    V = []
    for H in homographies:
        V.append(_v_ij(H, 0, 1))
        V.append(_v_ij(H, 0, 0) - _v_ij(H, 1, 1))
    if len(homographies) == 2:
        V.append([0, 1, 0, 0, 0, 0])  # zero-skew constraint
    _, _, Vt = np.linalg.svd(np.asarray(V))
    b = Vt[-1]
    if b[0] < 0:
        b = -b
    B11, B12, B22, B13, B23, B33 = b
    den = B11 * B22 - B12 ** 2
    v0 = (B12 * B13 - B11 * B23) / den
    lam = B33 - (B13 ** 2 + v0 * (B12 * B13 - B11 * B23)) / B11
    alpha = math.sqrt(lam / B11)
    beta = math.sqrt(lam * B11 / den)
    gamma = -B12 * alpha ** 2 * beta / lam
    u0 = gamma * v0 / beta - B13 * alpha ** 2 / lam
    return np.array([[alpha, gamma, u0], [0, beta, v0], [0, 0, 1.0]])


def extrinsics_from_homography(K: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Board->camera (R, t) from a plane homography; R re-orthogonalized via SVD."""
    Ki = np.linalg.inv(K)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    lam = 1.0 / np.linalg.norm(Ki @ h1)
    r1, r2 = lam * Ki @ h1, lam * Ki @ h2
    t = lam * Ki @ h3
    if t[2] < 0:  # board must be in front of camera
        r1, r2, t = -r1, -r2, -t
    R = np.column_stack([r1, r2, np.cross(r1, r2)])
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    return R, t


def project_points(X: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray,
                   dist: Optional[BrownConrady] = None) -> np.ndarray:
    """Project world points (N,3) given world->camera (R, t)."""
    pc = np.asarray(X, float) @ R.T + t
    xy = pc[:, :2] / pc[:, 2:3]
    if dist is not None:
        xy = dist.distort_normalized(xy)
    return np.stack([K[0, 0] * xy[:, 0] + K[0, 1] * xy[:, 1] + K[0, 2], K[1, 1] * xy[:, 1] + K[1, 2]], axis=1)


@dataclass
class CalibrationConfig:
    refine: bool = True
    estimate_distortion: bool = True  # k1, k2 only
    max_iters: int = 50
    tol: float = 1e-10


@dataclass
class CalibrationResult:
    K: np.ndarray
    dist: BrownConrady
    rotations: list[np.ndarray]
    translations: list[np.ndarray]
    rms_px: float

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return CameraIntrinsics.from_K(self.K)


def _reproj_residuals(params: np.ndarray, obj: Sequence[np.ndarray], img: Sequence[np.ndarray], use_dist: bool) -> np.ndarray:
    fx, fy, cx, cy = params[:4]
    off = 4
    dist = None
    if use_dist:
        dist = BrownConrady(params[4], params[5])
        off = 6
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    res = []
    for i, (X, uv) in enumerate(zip(obj, img)):
        p = params[off + 6 * i: off + 6 * i + 6]
        res.append((project_points(X, K, rodrigues(p[:3]), p[3:], dist) - uv).ravel())
    return np.concatenate(res)


def _levenberg_marquardt(fun, x0: np.ndarray, max_iters: int, tol: float) -> np.ndarray:
    """Small dense LM with forward-difference Jacobian. Fine for <~100 params."""
    x = x0.astype(float).copy()
    r = fun(x)
    cost = r @ r
    lam = 1e-3
    for _ in range(max_iters):
        J = np.empty((r.size, x.size))
        for k in range(x.size):
            h = 1e-6 * max(1.0, abs(x[k]))
            xk = x.copy()
            xk[k] += h
            J[:, k] = (fun(xk) - r) / h
        g = J.T @ r
        A = J.T @ J
        improved = False
        for _ in range(10):
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            xn = x + step
            rn = fun(xn)
            cn = rn @ rn
            if cn < cost:
                x, r, lam = xn, rn, max(lam / 3, 1e-12)
                improved = cost - cn > tol * max(cost, 1e-30)
                cost = cn
                break
            lam *= 4
        if not improved:
            break
    return x


def calibrate_zhang(object_points: Sequence[np.ndarray], image_points: Sequence[np.ndarray],
                    config: Optional[CalibrationConfig] = None) -> CalibrationResult:
    """Calibrate from planar targets. object_points[i]: (N,3) with Z=0 (or (N,2));
    image_points[i]: (N,2) observed pixels. Needs >= 3 views (2 assumes zero skew)."""
    c = config or CalibrationConfig()
    obj = [np.column_stack([np.asarray(o, float)[:, :2], np.zeros(len(o))]) for o in object_points]
    img = [np.asarray(p, float) for p in image_points]
    if len(obj) < 2:
        raise ValueError("need >= 2 views")
    Hs = [estimate_homography(o[:, :2], p) for o, p in zip(obj, img)]
    K = zhang_intrinsics(Hs)
    K[0, 1] = 0.0  # assume zero skew downstream
    Rs, ts = zip(*[extrinsics_from_homography(K, H) for H in Hs])
    Rs, ts = list(Rs), list(ts)
    dist = BrownConrady()
    if c.refine:
        x0 = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]] + ([0.0, 0.0] if c.estimate_distortion else [])
        for R, t in zip(Rs, ts):
            x0 += list(rodrigues_inv(R)) + list(t)
        fun = lambda p: _reproj_residuals(p, obj, img, c.estimate_distortion)
        x = _levenberg_marquardt(fun, np.asarray(x0), c.max_iters, c.tol)
        K = np.array([[x[0], 0, x[2]], [0, x[1], x[3]], [0, 0, 1.0]])
        off = 4
        if c.estimate_distortion:
            dist = BrownConrady(float(x[4]), float(x[5]))
            off = 6
        Rs = [rodrigues(x[off + 6 * i: off + 6 * i + 3]) for i in range(len(obj))]
        ts = [x[off + 6 * i + 3: off + 6 * i + 6].copy() for i in range(len(obj))]
    res = np.concatenate([(project_points(o, K, R, t, dist) - p).ravel() for o, p, R, t in zip(obj, img, Rs, ts)])
    rms = float(np.sqrt(np.mean(res.reshape(-1, 2) ** 2) * 2))
    return CalibrationResult(K, dist, Rs, ts, rms)
