"""Curvature modeling: discrete curvature, smoothing, clothoids and cubic splines.

Sign convention: curvature > 0 turns LEFT (CCW), consistent with yaw CCW from +x.

* `discrete_curvature`   Menger (circumscribed-circle) curvature at each vertex.
* `smooth_polyline`      Laplacian smoothing with pinned endpoints (shrinks curves slightly).
* `smooth_signal`        edge-padded moving average for 1D profiles (e.g. kappa(s)).
* `ClothoidSegment`      kappa(s) = kappa0 + dkappa * s, integrated with Gauss-Legendre.
* `fit_clothoid_g1`      G1 Hermite fit (two poses -> clothoid) by damped Newton.
* `fit_linear_curvature` least-squares clothoid curvature model on (s, kappa) samples.
* `CubicSpline2D`        natural cubic spline parameterized by chord length.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from cheesestack.core import wrap_angle


@dataclass
class CurvatureConfig:
    smoothing_iterations: int = 10
    smoothing_alpha: float = 0.5  # Laplacian step in (0, 1]
    signal_window: int = 5  # odd, samples
    resample_ds: float = 1.0  # m, for curvature_profile


# ----------------------------------------------------------- discrete curvature


def discrete_curvature(points) -> np.ndarray:
    """Signed Menger curvature at every vertex of an (N, 2) polyline.

    kappa_i = 2 * cross(b-a, c-b) / (|b-a| |c-b| |c-a|). Endpoints copy their
    neighbor. Sensitive to vertex noise; smooth or resample first.
    """
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    n = len(p)
    k = np.zeros(n)
    if n < 3:
        return k
    a, b, c = p[:-2], p[1:-1], p[2:]
    ab, bc, ac = b - a, c - b, c - a
    cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
    denom = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1) * np.linalg.norm(ac, axis=1)
    k[1:-1] = np.where(denom > 1e-12, 2.0 * cross / np.maximum(denom, 1e-12), 0.0)
    k[0], k[-1] = k[1], k[-2]
    return k


def heading_rate_curvature(points) -> np.ndarray:
    """Alternative estimator: d(heading)/ds with central differences on vertices."""
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(p) < 3:
        return np.zeros(len(p))
    seg = np.diff(p, axis=0)
    h = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
    L = np.linalg.norm(seg, axis=1)
    k = np.zeros(len(p))
    k[1:-1] = (h[1:] - h[:-1]) / (0.5 * (L[1:] + L[:-1]))
    k[0], k[-1] = k[1], k[-2]
    return k


def curvature_profile(polyline, ds: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(s, kappa) every ~ds along a `Polyline` (needs .points, .s, .length).

    Curvature is evaluated on the original vertices (resampling first would put
    collinear samples inside straight segments and read zero) and linearly
    interpolated onto the uniform grid.
    """
    n = max(3, int(math.ceil(polyline.length / ds)) + 1)
    s = np.linspace(0.0, polyline.length, n)
    return s, np.interp(s, polyline.s, discrete_curvature(polyline.points))


# -------------------------------------------------------------------- smoothing


def smooth_polyline(points, iterations: int = 10, alpha: float = 0.5) -> np.ndarray:
    """Laplacian smoothing, endpoints pinned. Each pass moves vertices by
    alpha * (neighbor mean - vertex). Shrinks arcs towards their chord.
    # TODO(mapping): Taubin (lambda/mu) smoothing or QP smoothing with
    # deviation bounds to avoid shrinkage.
    """
    p = np.asarray(points, dtype=float).reshape(-1, 2).copy()
    if len(p) < 3:
        return p
    for _ in range(max(0, iterations)):
        p[1:-1] += alpha * (0.5 * (p[:-2] + p[2:]) - p[1:-1])
    return p


def smooth_signal(values, window: int = 5) -> np.ndarray:
    """Centered moving average with edge padding (output same length)."""
    v = np.asarray(values, dtype=float)
    w = max(1, int(window) | 1)  # force odd
    if w == 1 or len(v) == 0:
        return v.copy()
    pad = w // 2
    vp = np.pad(v, pad, mode="edge")
    return np.convolve(vp, np.ones(w) / w, mode="valid")


# --------------------------------------------------------------------- clothoid

_GL_X, _GL_W = np.polynomial.legendre.leggauss(32)


@dataclass
class ClothoidSegment:
    """Euler spiral: theta(s) = theta0 + kappa0 s + 0.5 dkappa s^2, s in [0, length]."""

    x0: float
    y0: float
    theta0: float
    kappa0: float
    dkappa: float
    length: float

    def heading(self, s) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        return self.theta0 + self.kappa0 * s + 0.5 * self.dkappa * s * s

    def curvature(self, s) -> np.ndarray:
        return self.kappa0 + self.dkappa * np.asarray(s, dtype=float)

    def point_at(self, s: float) -> np.ndarray:
        """Position at arc length s via 32-point Gauss-Legendre quadrature."""
        if s == 0.0:
            return np.array([self.x0, self.y0])
        u = 0.5 * s * (_GL_X + 1.0)
        th = self.heading(u)
        return np.array([self.x0 + 0.5 * s * np.dot(_GL_W, np.cos(th)),
                         self.y0 + 0.5 * s * np.dot(_GL_W, np.sin(th))])

    def end_pose(self) -> tuple[float, float, float]:
        p = self.point_at(self.length)
        return float(p[0]), float(p[1]), float(wrap_angle(float(self.heading(self.length))))

    def sample(self, n: int = 50) -> np.ndarray:
        """(n, 2) points uniformly in s (cumulative per-interval quadrature)."""
        s = np.linspace(0.0, self.length, max(2, n))
        pts = np.zeros((len(s), 2))
        pts[0] = (self.x0, self.y0)
        x5, w5 = np.polynomial.legendre.leggauss(5)
        for i in range(1, len(s)):
            a, b = s[i - 1], s[i]
            u = 0.5 * (b - a) * (x5 + 1.0) + a
            th = self.heading(u)
            pts[i] = pts[i - 1] + 0.5 * (b - a) * np.array([np.dot(w5, np.cos(th)), np.dot(w5, np.sin(th))])
        return pts


def fit_clothoid_g1(p0, theta0: float, p1, theta1: float, max_iter: int = 50,
                    tol: float = 1e-8) -> Optional[ClothoidSegment]:
    """Clothoid from pose (p0, theta0) to pose (p1, theta1), matching both headings (G1).

    Unknowns (kappa0, dkappa, L); residual (x_end, y_end, theta_end) error.
    Damped Newton with a finite-difference Jacobian. The heading change is
    taken in (-pi, pi] (principal branch). Returns None if not converged.
    # TODO(mapping): Bertolazzi-Frego closed-form-initialized solver for robust
    # convergence on large turning angles / loops.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    chord_v = p1 - p0
    chord = float(np.linalg.norm(chord_v))
    if chord < 1e-9:
        return None
    phi = math.atan2(chord_v[1], chord_v[0])
    a0 = wrap_angle(theta0 - phi)
    a1 = wrap_angle(theta1 - phi)
    dth = a1 - a0  # total heading change on principal branch
    # Initial guess from the small-angle solution: heading relative to the chord is
    # quadratic in s with theta(0)=a0, theta(L)=a1 and zero net lateral offset.
    L = chord * (1.0 + (a0 * a0 + a1 * a1) / 6.0)
    x = np.array([(-4.0 * a0 - 2.0 * a1) / L, 6.0 * (a0 + a1) / (L * L), L])
    target_th = theta0 + dth  # unwrapped target

    def resid(v):
        c = ClothoidSegment(p0[0], p0[1], theta0, v[0], v[1], v[2])
        e = c.point_at(v[2])
        return np.array([(e[0] - p1[0]) / chord, (e[1] - p1[1]) / chord, float(c.heading(v[2])) - target_th])

    r = resid(x)
    for _ in range(max_iter):
        if np.linalg.norm(r) < tol:
            break
        J = np.zeros((3, 3))
        for j in range(3):
            h = 1e-7 * max(1.0, abs(x[j]))
            xp = x.copy()
            xp[j] += h
            J[:, j] = (resid(xp) - r) / h
        try:
            step = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(J, -r, rcond=None)[0]
        lam = 1.0
        while lam > 1e-4:
            xn = x + lam * step
            if xn[2] > 0:
                rn = resid(xn)
                if np.linalg.norm(rn) < np.linalg.norm(r):
                    x, r = xn, rn
                    break
            lam *= 0.5
        else:
            break
    if np.linalg.norm(r) > 1e-6:
        return None
    return ClothoidSegment(float(p0[0]), float(p0[1]), float(theta0), float(x[0]), float(x[1]), float(x[2]))


def fit_linear_curvature(s, kappa) -> tuple[float, float]:
    """Least-squares (kappa0, dkappa) for kappa(s) = kappa0 + dkappa * s."""
    s = np.asarray(s, dtype=float)
    k = np.asarray(kappa, dtype=float)
    A = np.column_stack([np.ones_like(s), s])
    (k0, dk), *_ = np.linalg.lstsq(A, k, rcond=None)
    return float(k0), float(dk)


# ---------------------------------------------------------------- cubic spline


class CubicSpline1D:
    """Natural cubic spline y(x) for strictly increasing x (dense tridiagonal solve)."""

    def __init__(self, x, y) -> None:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) < 2 or np.any(np.diff(x) <= 0):
            raise ValueError("x must be strictly increasing with >= 2 samples")
        self.x, self.a = x, y
        n = len(x)
        h = np.diff(x)
        A = np.zeros((n, n))
        rhs = np.zeros(n)
        A[0, 0] = A[-1, -1] = 1.0
        for i in range(1, n - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2.0 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            rhs[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
        # TODO(mapping): banded/Thomas solver for long splines (dense is O(n^3)).
        self.c = np.linalg.solve(A, rhs)
        self.b = (y[1:] - y[:-1]) / h - h * (2.0 * self.c[:-1] + self.c[1:]) / 3.0
        self.d = (self.c[1:] - self.c[:-1]) / (3.0 * h)

    def _idx(self, t):
        t = np.clip(np.asarray(t, dtype=float), self.x[0], self.x[-1])
        i = np.clip(np.searchsorted(self.x, t, side="right") - 1, 0, len(self.x) - 2)
        return i, t - self.x[i]

    def __call__(self, t, deriv: int = 0):
        i, dx = self._idx(t)
        if deriv == 0:
            return self.a[i] + self.b[i] * dx + self.c[i] * dx ** 2 + self.d[i] * dx ** 3
        if deriv == 1:
            return self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2
        if deriv == 2:
            return 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        raise ValueError("deriv must be 0, 1 or 2")


class CubicSpline2D:
    """Parametric natural cubic spline through (N, 2) points, parameter = chord length."""

    def __init__(self, points) -> None:
        p = np.asarray(points, dtype=float).reshape(-1, 2)
        keep = np.ones(len(p), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-9
        p = p[keep]
        self.s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
        self.sx = CubicSpline1D(self.s, p[:, 0])
        self.sy = CubicSpline1D(self.s, p[:, 1])

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def position(self, s) -> np.ndarray:
        return np.stack([self.sx(s), self.sy(s)], axis=-1)

    def heading(self, s):
        return np.arctan2(self.sy(s, 1), self.sx(s, 1))

    def curvature(self, s):
        dx, dy = self.sx(s, 1), self.sy(s, 1)
        ddx, ddy = self.sx(s, 2), self.sy(s, 2)
        return (dx * ddy - dy * ddx) / np.maximum((dx * dx + dy * dy) ** 1.5, 1e-12)

    def sample(self, ds: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(s, points (N,2), heading, curvature) every ~ds."""
        n = max(2, int(math.ceil(self.length / ds)) + 1)
        s = np.linspace(0.0, self.length, n)
        return s, self.position(s), self.heading(s), self.curvature(s)
