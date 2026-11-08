"""Synthetic-road environment generation.

A road is a reference line (the road centerline) built from segments whose
curvature varies linearly with arc length: straights (k = 0), arcs (k const)
and clothoids / Euler spirals (k linear in s). Heading is integrated
analytically, position with Simpson's rule per sample interval, so the
centerline is accurate to ~1e-9 m for any realistic sample spacing.

Lane convention: `num_lanes` lanes of `lane_width` centred on the reference
line. Lane 0 is the rightmost lane; lateral offset `d` is positive to the
left (consistent with x-forward / y-left). The Frenet pair (s, d) is the
arc length along the centerline and the signed lateral offset.

Limits: planar (no elevation/banking), constant lane count/width per road,
no junctions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from cheesestack.core import Pose2D


@dataclass(frozen=True)
class RoadSegment:
    """Segment with curvature linear in local arc length: k(s) = k_start + (k_end - k_start) s / L."""

    length: float
    k_start: float = 0.0
    k_end: float = 0.0
    kind: str = "straight"

    @staticmethod
    def straight(length: float) -> "RoadSegment":
        return RoadSegment(float(length), 0.0, 0.0, "straight")

    @staticmethod
    def arc(length: float, curvature: float) -> "RoadSegment":
        return RoadSegment(float(length), float(curvature), float(curvature), "arc")

    @staticmethod
    def arc_radius(radius: float, angle_rad: float) -> "RoadSegment":
        """Arc turning `angle_rad` (sign gives direction, + = left) with |radius|."""
        k = math.copysign(1.0 / abs(radius), angle_rad)
        return RoadSegment.arc(abs(angle_rad) * abs(radius), k)

    @staticmethod
    def clothoid(length: float, k_start: float, k_end: float) -> "RoadSegment":
        return RoadSegment(float(length), float(k_start), float(k_end), "clothoid")

    def curvature(self, s_local):
        s = np.clip(s_local, 0.0, self.length)
        return self.k_start + (self.k_end - self.k_start) * s / self.length

    @property
    def heading_change(self) -> float:
        return 0.5 * (self.k_start + self.k_end) * self.length


@dataclass(frozen=True)
class LaneLine:
    """A painted line at constant lateral offset `d`."""

    d: float
    style: str  # "solid" | "dashed"


class Road:
    """Sampled road reference line with Frenet queries."""

    def __init__(
        self,
        segments: Sequence[RoadSegment],
        start: Pose2D = Pose2D(),
        num_lanes: int = 1,
        lane_width: float = 3.5,
        sample_ds: float = 0.5,
        speed_limit: float = 25.0,
    ) -> None:
        if not segments:
            raise ValueError("Road needs at least one segment")
        if num_lanes < 1:
            raise ValueError("num_lanes must be >= 1")
        self.segments = list(segments)
        self.start = start
        self.num_lanes = int(num_lanes)
        self.lane_width = float(lane_width)
        self.sample_ds = float(sample_ds)
        self.speed_limit = float(speed_limit)
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        s_all = [np.array([0.0])]
        x_all = [np.array([self.start.x])]
        y_all = [np.array([self.start.y])]
        yaw_all = [np.array([self.start.yaw])]
        k_all = [np.array([self.segments[0].k_start])]
        s0, x0, y0, yaw0 = 0.0, self.start.x, self.start.y, self.start.yaw
        for seg in self.segments:
            L = seg.length
            n = max(1, int(math.ceil(L / self.sample_ds)))
            h = L / n
            dk = (seg.k_end - seg.k_start) / L
            sl = np.linspace(0.0, L, n + 1)

            def heading(s, yaw0=yaw0, seg=seg, dk=dk):
                return yaw0 + seg.k_start * s + 0.5 * dk * s * s

            ya, ym, yb = heading(sl[:-1]), heading(sl[:-1] + 0.5 * h), heading(sl[1:])
            dx = h / 6.0 * (np.cos(ya) + 4.0 * np.cos(ym) + np.cos(yb))
            dy = h / 6.0 * (np.sin(ya) + 4.0 * np.sin(ym) + np.sin(yb))
            xs = x0 + np.cumsum(dx)
            ys = y0 + np.cumsum(dy)
            s_all.append(s0 + sl[1:])
            x_all.append(xs)
            y_all.append(ys)
            yaw_all.append(heading(sl[1:]))
            k_all.append(seg.curvature(sl[1:]))
            s0, x0, y0, yaw0 = s0 + L, xs[-1], ys[-1], float(heading(L))
        self.s = np.concatenate(s_all)
        self.x = np.concatenate(x_all)
        self.y = np.concatenate(y_all)
        self.yaw_unwrapped = np.concatenate(yaw_all)
        self.kappa = np.concatenate(k_all)
        self.points = np.stack([self.x, self.y], axis=1)

    # -------------------------------------------------------------- geometry
    @property
    def length(self) -> float:
        return float(self.s[-1])

    @property
    def half_width(self) -> float:
        return 0.5 * self.num_lanes * self.lane_width

    def lane_offset(self, lane: int) -> float:
        """Lateral offset of lane `lane` centre (lane 0 = rightmost)."""
        if not 0 <= lane < self.num_lanes:
            raise IndexError(f"lane {lane} out of range [0, {self.num_lanes})")
        return (lane - 0.5 * (self.num_lanes - 1)) * self.lane_width

    def lane_index(self, d: float) -> Optional[int]:
        """Lane containing lateral offset `d`, or None if off-road."""
        i = int(math.floor((d + self.half_width) / self.lane_width))
        return i if 0 <= i < self.num_lanes else None

    def lane_lines(self) -> list[LaneLine]:
        lines = [LaneLine(-self.half_width, "solid")]
        for i in range(1, self.num_lanes):
            lines.append(LaneLine(-self.half_width + i * self.lane_width, "dashed"))
        lines.append(LaneLine(self.half_width, "solid"))
        return lines

    def sample_at(self, s, d=0.0):
        """Vectorized (x, y, yaw) at arc lengths `s` and lateral offsets `d`.

        Beyond the road ends the reference line is extrapolated along the end tangent.
        """
        s = np.atleast_1d(np.asarray(s, dtype=float))
        d = np.broadcast_to(np.asarray(d, dtype=float), s.shape)
        sc = np.clip(s, 0.0, self.length)
        x = np.interp(sc, self.s, self.x)
        y = np.interp(sc, self.s, self.y)
        yaw = np.interp(sc, self.s, self.yaw_unwrapped)
        extra = s - sc
        x = x + extra * np.cos(yaw)
        y = y + extra * np.sin(yaw)
        # Heading of an offset curve equals the reference heading.
        x = x - d * np.sin(yaw)
        y = y + d * np.cos(yaw)
        return x, y, (yaw + math.pi) % (2.0 * math.pi) - math.pi

    def pose_at(self, s: float, d: float = 0.0) -> Pose2D:
        x, y, yaw = self.sample_at(s, d)
        return Pose2D(float(x[0]), float(y[0]), float(yaw[0]))

    def curvature_at(self, s: float) -> float:
        """Reference-line curvature at s (0 beyond the ends)."""
        if s < 0 or s > self.length:
            return 0.0
        return float(np.interp(s, self.s, self.kappa))

    def heading_at(self, s: float) -> float:
        return self.pose_at(s).yaw

    def line_points(self, d: float, ds: Optional[float] = None) -> np.ndarray:
        """(N, 2) polyline at constant lateral offset d (lane centre or boundary)."""
        s = self.s if ds is None else np.arange(0.0, self.length + 1e-9, ds)
        x, y, _ = self.sample_at(s, d)
        return np.stack([x, y], axis=1)

    def boundaries(self, ds: Optional[float] = None) -> dict:
        """Centerline, left/right road edges and lane divider polylines."""
        return {
            "center": self.line_points(0.0, ds),
            "left": self.line_points(self.half_width, ds),
            "right": self.line_points(-self.half_width, ds),
            "dividers": [self.line_points(l.d, ds) for l in self.lane_lines() if l.style == "dashed"],
            "lane_centers": [self.line_points(self.lane_offset(i), ds) for i in range(self.num_lanes)],
        }

    # ------------------------------------------------------------ projection
    def project_many(self, xy, s_window: Optional[tuple[float, float]] = None, chunk: int = 1024):
        """Project points onto the reference line. Returns (s, d) arrays.

        Exact nearest-point projection onto the sampled polyline (O(N*M) brute
        force, chunked). `s_window` restricts the search to a stretch of road,
        which the camera renderer uses to keep this cheap.
        # TODO(sim): spatial hash / Newton refinement on the analytic clothoid for large maps.
        """
        P = np.asarray(xy, dtype=float).reshape(-1, 2)
        n_seg = len(self.s) - 1
        i0, i1 = 0, n_seg
        if s_window is not None:
            i0 = max(0, int(np.searchsorted(self.s, s_window[0])) - 1)
            i1 = min(n_seg, int(np.searchsorted(self.s, s_window[1])) + 1)
            i1 = max(i1, i0 + 1)
        A = self.points[i0:i1]
        B = self.points[i0 + 1 : i1 + 1]
        AB = B - A
        L2 = np.maximum(np.einsum("ij,ij->i", AB, AB), 1e-12)
        lo = np.zeros(len(A))
        hi = np.ones(len(A))
        if i0 == 0:
            lo[0] = -np.inf
        if i1 == n_seg:
            hi[-1] = np.inf
        s_out = np.empty(len(P))
        d_out = np.empty(len(P))
        for c in range(0, len(P), chunk):
            p = P[c : c + chunk]
            AP = p[:, None, :] - A[None, :, :]
            t = np.einsum("nmk,mk->nm", AP, AB) / L2
            t = np.clip(t, lo, hi)
            foot = A[None] + t[..., None] * AB[None]
            dist2 = np.sum((p[:, None, :] - foot) ** 2, axis=2)
            j = np.argmin(dist2, axis=1)
            tj = t[np.arange(len(p)), j]
            seg_len = np.sqrt(L2[j])
            s_out[c : c + chunk] = self.s[i0 + j] + tj * seg_len
            ap = AP[np.arange(len(p)), j]
            cross = AB[j, 0] * ap[:, 1] - AB[j, 1] * ap[:, 0]
            d_out[c : c + chunk] = cross / seg_len
        return s_out, d_out

    def project(self, x: float, y: float, s_window: Optional[tuple[float, float]] = None) -> tuple[float, float]:
        s, d = self.project_many([[x, y]], s_window)
        return float(s[0]), float(d[0])

    # ---------------------------------------------------------- constructors
    @staticmethod
    def straight(length: float = 300.0, **kw) -> "Road":
        return Road([RoadSegment.straight(length)], **kw)

    @staticmethod
    def curve(
        radius: float = 150.0,
        angle_rad: float = math.pi / 2,
        lead_in: float = 40.0,
        transition: float = 30.0,
        lead_out: float = 60.0,
        **kw,
    ) -> "Road":
        """Straight -> clothoid -> arc -> clothoid -> straight (curvature-continuous)."""
        k = math.copysign(1.0 / radius, angle_rad)
        arc_angle = abs(angle_rad) - abs(k) * transition  # each clothoid turns k*T/2
        if arc_angle <= 0:
            raise ValueError("transition too long for requested angle")
        segs = [
            RoadSegment.straight(lead_in),
            RoadSegment.clothoid(transition, 0.0, k),
            RoadSegment.arc(arc_angle * radius, k),
            RoadSegment.clothoid(transition, k, 0.0),
            RoadSegment.straight(lead_out),
        ]
        return Road(segs, **kw)


@dataclass
class RoadGeneratorConfig:
    n_pieces: int = 5
    min_length: float = 30.0
    max_length: float = 120.0
    max_curvature: float = 1.0 / 120.0
    p_straight: float = 0.4
    min_transition: float = 15.0
    max_transition: float = 50.0
    lane_counts: tuple[int, ...] = (1, 2, 3)
    lane_width_range: tuple[float, float] = (3.0, 3.75)
    sample_ds: float = 0.5
    seed: int = 0


class RoadGenerator:
    """Procedural, curvature-continuous roads: random straights/arcs joined by clothoids.

    # TODO(sim): junctions, lane merges/splits, elevation, OpenDRIVE export.
    """

    def __init__(self, config: RoadGeneratorConfig = RoadGeneratorConfig()) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def generate(self, rng: Optional[np.random.Generator] = None, start: Pose2D = Pose2D()) -> Road:
        c = self.config
        rng = rng or self._rng
        segs: list[RoadSegment] = [RoadSegment.straight(float(rng.uniform(c.min_length, c.max_length)))]
        k_cur = 0.0
        for _ in range(c.n_pieces):
            if rng.random() < c.p_straight:
                k_next = 0.0
            else:
                k_next = float(rng.uniform(0.3, 1.0) * c.max_curvature * rng.choice([-1.0, 1.0]))
            if not math.isclose(k_next, k_cur):
                segs.append(RoadSegment.clothoid(float(rng.uniform(c.min_transition, c.max_transition)), k_cur, k_next))
            length = float(rng.uniform(c.min_length, c.max_length))
            segs.append(RoadSegment.straight(length) if k_next == 0.0 else RoadSegment.arc(length, k_next))
            k_cur = k_next
        if k_cur != 0.0:
            segs.append(RoadSegment.clothoid(c.min_transition, k_cur, 0.0))
        return Road(
            segs,
            start=start,
            num_lanes=int(rng.choice(c.lane_counts)),
            lane_width=float(rng.uniform(*c.lane_width_range)),
            sample_ds=c.sample_ds,
        )
