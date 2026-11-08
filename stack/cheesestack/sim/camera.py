"""Camera simulation: pinhole camera rendering a synthetic road scene to numpy.

Rendering is a per-pixel ground-plane ray cast:
  1. back-project every pixel through the pinhole model, intersect with z = 0;
  2. map ground hits to road Frenet (s, d) via `Road.project_many`;
  3. colour: road surface, solid edge lines, dashed lane dividers, grass; sky above
     the horizon; exponential fog with distance;
  4. actors are drawn as shaded image-space boxes (projected 3-D cuboid hull),
     painter's order far -> near; their 2-D boxes are returned as ground truth labels;
  5. lighting (brightness/contrast/gamma), vignetting and Gaussian pixel noise.

Frames: camera optical frame x right, y down, z forward. Mount pose in base_link
(x fwd, y left, z up) given by (x, y, z, pitch) with positive pitch looking down.

Limits: flat world, no texture/shadows/occlusion by terrain, boxes only (no
meshes), near-plane culling instead of clipping.
# TODO(sim): lens distortion, rolling shutter, motion blur, HDR/auto-exposure,
# per-pixel depth & segmentation outputs, mesh/sprite actors, traffic signs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from cheesestack.core import BoundingBox2D, ObjectClass, Pose2D

from .physics import ActorTruth
from .roads import Road


@dataclass
class PinholeCamera:
    width: int = 160
    height: int = 120
    fx: float = 120.0
    fy: float = 120.0
    cx: Optional[float] = None
    cy: Optional[float] = None
    mount_x: float = 1.5
    mount_y: float = 0.0
    mount_z: float = 1.3
    pitch: float = math.radians(3.0)
    yaw: float = 0.0

    def __post_init__(self) -> None:
        if self.cx is None:
            self.cx = 0.5 * (self.width - 1)
        if self.cy is None:
            self.cy = 0.5 * (self.height - 1)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    @property
    def R_cam_from_base(self) -> np.ndarray:
        """Rows are the camera x/y/z axes expressed in base_link."""
        cp, sp, cy, sy = math.cos(self.pitch), math.sin(self.pitch), math.cos(self.yaw), math.sin(self.yaw)
        z = np.array([cp * cy, cp * sy, -sp])
        x = np.array([sy, -cy, 0.0])
        y = np.cross(z, x)
        return np.stack([x, y, z])

    @property
    def t_base(self) -> np.ndarray:
        return np.array([self.mount_x, self.mount_y, self.mount_z])

    def base_to_cam(self, pts_base: np.ndarray) -> np.ndarray:
        return (np.asarray(pts_base, float).reshape(-1, 3) - self.t_base) @ self.R_cam_from_base.T

    def project(self, pts_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixel coords (N, 2) and validity mask (in front of camera) for optical-frame points."""
        p = np.asarray(pts_cam, float).reshape(-1, 3)
        valid = p[:, 2] > 1e-6
        z = np.where(valid, p[:, 2], 1.0)
        uv = np.stack([self.fx * p[:, 0] / z + self.cx, self.fy * p[:, 1] / z + self.cy], axis=1)
        return uv, valid

    def project_base(self, pts_base) -> tuple[np.ndarray, np.ndarray]:
        return self.project(self.base_to_cam(pts_base))

    def pixel_rays_base(self) -> np.ndarray:
        """(H, W, 3) unit-free ray directions in base_link for every pixel centre."""
        u, v = np.meshgrid(np.arange(self.width, dtype=float), np.arange(self.height, dtype=float))
        d_cam = np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy, np.ones_like(u)], axis=-1)
        return d_cam @ self.R_cam_from_base

    def ground_points(self, max_distance: float = 120.0) -> tuple[np.ndarray, np.ndarray]:
        """Ground-plane intersection (H, W, 2) in base_link and hit mask."""
        d = self.pixel_rays_base()
        hit = d[..., 2] < -1e-6
        lam = np.where(hit, -self.mount_z / np.where(hit, d[..., 2], -1.0), np.inf)
        g = self.t_base[None, None, :2] + lam[..., None] * d[..., :2]
        dist = np.hypot(g[..., 0] - self.mount_x, g[..., 1] - self.mount_y)
        hit &= dist <= max_distance
        return g, hit


@dataclass
class LightingConfig:
    brightness: float = 1.0
    contrast: float = 1.0
    gamma: float = 1.0
    noise_std: float = 0.01
    vignetting: float = 0.15
    fog_distance: Optional[float] = 150.0  # e-folding distance; None disables
    sky: tuple = (0.55, 0.70, 0.90)
    fog: tuple = (0.75, 0.78, 0.80)

    @staticmethod
    def night() -> "LightingConfig":
        return LightingConfig(brightness=0.25, contrast=0.8, gamma=1.3, noise_std=0.04, sky=(0.05, 0.05, 0.1),
                              fog=(0.08, 0.08, 0.1))

    @staticmethod
    def foggy() -> "LightingConfig":
        return LightingConfig(fog_distance=30.0)


CLASS_COLORS = {
    ObjectClass.CAR: (0.75, 0.10, 0.10),
    ObjectClass.TRUCK: (0.20, 0.30, 0.70),
    ObjectClass.BUS: (0.90, 0.70, 0.10),
    ObjectClass.PEDESTRIAN: (0.95, 0.55, 0.15),
    ObjectClass.BICYCLE: (0.20, 0.70, 0.30),
}


@dataclass
class RenderConfig:
    max_distance: float = 120.0
    road_color: tuple = (0.32, 0.32, 0.34)
    grass_color: tuple = (0.25, 0.45, 0.20)
    marking_color: tuple = (0.95, 0.95, 0.92)
    marking_width: float = 0.15
    dash_length: float = 3.0
    dash_period: float = 9.0
    lighting: LightingConfig = field(default_factory=LightingConfig)


@dataclass
class CameraFrame:
    stamp: float
    image: np.ndarray  # HxWx3 float32 [0, 1]
    boxes: list[tuple[str, ObjectClass, BoundingBox2D]]
    depth_hint: Optional[np.ndarray] = None  # ground distance per pixel (inf for sky)


class CameraRenderer:
    def __init__(self, camera: PinholeCamera, road: Road, config: RenderConfig = RenderConfig(),
                 rng: Optional[np.random.Generator] = None) -> None:
        self.camera, self.road, self.config = camera, road, config
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self._ground, self._hit = camera.ground_points(config.max_distance)

    def render(self, ego_pose: Pose2D, actors: Sequence[ActorTruth] = (), stamp: float = 0.0) -> CameraFrame:
        cam, cfg, L = self.camera, self.config, self.config.lighting
        H, W = cam.height, cam.width
        img = np.empty((H, W, 3), dtype=np.float64)
        img[:] = np.asarray(L.sky)
        g = self._ground[self._hit]  # (N, 2) base_link
        world = ego_pose.transform_points(g)
        s_ego, _ = self.road.project(ego_pose.x, ego_pose.y)
        s, d = self.road.project_many(world, (s_ego - 20.0, s_ego + cfg.max_distance + 20.0))
        col = np.tile(np.asarray(cfg.grass_color, float), (len(g), 1))
        on_road = np.abs(d) <= self.road.half_width
        col[on_road] = cfg.road_color
        hw = 0.5 * cfg.marking_width
        for line in self.road.lane_lines():
            m = np.abs(d - line.d) <= hw
            if line.style == "dashed":
                m &= np.mod(s, cfg.dash_period) < cfg.dash_length
            col[m] = cfg.marking_color
        dist = np.hypot(g[:, 0] - cam.mount_x, g[:, 1] - cam.mount_y)
        if L.fog_distance:
            f = np.exp(-dist / L.fog_distance)[:, None]
            col = f * col + (1 - f) * np.asarray(L.fog)
        img[self._hit] = col
        depth = np.full((H, W), np.inf)
        depth[self._hit] = dist

        boxes = self._draw_actors(img, ego_pose, actors)
        img = self._apply_lighting(img)
        return CameraFrame(stamp, img.astype(np.float32), boxes, depth)

    def _draw_actors(self, img, ego_pose: Pose2D, actors) -> list:
        cam = self.camera
        inv = ego_pose.inverse()
        items = []
        for a in actors:
            rel = inv.compose(a.pose)
            hl, hw = 0.5 * a.length, 0.5 * a.width
            foot = rel.transform_points(np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]]))
            pts = np.concatenate([np.c_[foot, np.zeros(4)], np.c_[foot, np.full(4, a.height)]])
            pc = cam.base_to_cam(pts)
            if np.any(pc[:, 2] < 0.5):
                continue  # TODO(sim): clip against the near plane instead of culling
            uv, _ = cam.project(pc)
            u0, v0 = np.floor(uv.min(axis=0)).astype(int)
            u1, v1 = np.ceil(uv.max(axis=0)).astype(int)
            if u1 < 0 or v1 < 0 or u0 >= cam.width or v0 >= cam.height:
                continue
            items.append((float(pc[:, 2].mean()), a, (u0, v0, u1, v1)))
        boxes = []
        for depth, a, (u0, v0, u1, v1) in sorted(items, key=lambda x: -x[0]):
            cu0, cv0, cu1, cv1 = max(u0, 0), max(v0, 0), min(u1, cam.width), min(v1, cam.height)
            base = np.asarray(CLASS_COLORS.get(a.cls, (0.6, 0.6, 0.6)))
            rows = np.linspace(1.1, 0.7, max(1, cv1 - cv0))[:, None, None]  # vertical shading
            patch = np.clip(base[None, None, :] * rows, 0, 1)
            if self.config.lighting.fog_distance:
                f = math.exp(-depth / self.config.lighting.fog_distance)
                patch = f * patch + (1 - f) * np.asarray(self.config.lighting.fog)
            img[cv0:cv1, cu0:cu1] = patch
            boxes.append((a.name, a.cls, BoundingBox2D(float(cu0), float(cv0), float(cu1 - cu0), float(cv1 - cv0))))
        return boxes

    def _apply_lighting(self, img: np.ndarray) -> np.ndarray:
        L, cam = self.config.lighting, self.camera
        out = (img - 0.5) * L.contrast + 0.5
        out = out * L.brightness
        if L.vignetting > 0:
            u, v = np.meshgrid(np.linspace(-1, 1, cam.width), np.linspace(-1, 1, cam.height))
            out = out * (1.0 - L.vignetting * (u * u + v * v) / 2.0)[..., None]
        out = np.clip(out, 0.0, 1.0) ** L.gamma
        if L.noise_std > 0:
            out = out + self.rng.normal(0.0, L.noise_std, out.shape)
        return np.clip(out, 0.0, 1.0)
