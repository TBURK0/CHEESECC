"""Trajectory generation: Frenet lattice sampling (Werling et al. 2010 style).

Lateral motion d(t) is a quintic from the current (d, d_dot, d_ddot) to
(d_end, 0, 0) at T_lat; longitudinal motion s(t) is a quartic from (s, s_dot,
s_ddot) to (v_end, 0) at T_lon (velocity keeping) — or a quintic to a stop
position (stopping mode). All candidates span a common horizon: after T the
lateral offset is held and the longitudinal speed continues at its final value.

Candidates are converted to Cartesian through the `ReferencePath`, ranked by
    J = J_lat + J_lon + w_comfort * comfort_cost
with J_lat = k_j ∫d'''^2 + k_t T_lat + k_d (d_end - d_target)^2 and
     J_lon = k_j ∫s'''^2 + k_t T_lon + k_v (v_end - v_target)^2,
then checked against `ConstraintChecker` in cost order; the first feasible one is
returned as a `core.Trajectory`. If none is feasible, a fallback braking
trajectory along the current lateral offset is returned and flagged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cheesestack.core import Header, Trajectory, TrajectoryPoint

from .comfort import ComfortModel
from .common import ObstacleBox, QuarticPolynomial, QuinticPolynomial, ReferencePath
from .constraints import ConstraintChecker, ConstraintConfig, ConstraintReport


@dataclass
class FrenetState:
    s: float
    s_d: float
    s_dd: float = 0.0
    d: float = 0.0
    d_d: float = 0.0
    d_dd: float = 0.0


@dataclass
class TrajectoryConfig:
    horizon_s: float = 5.0
    dt: float = 0.1
    lat_times_s: tuple = (2.5, 3.5, 4.5)
    lon_times_s: tuple = (2.0, 3.0, 4.0, 5.0)
    d_offsets: tuple = (-3.6, -1.8, -0.9, 0.0, 0.9, 1.8, 3.6)  # end offsets relative to d_target
    dv_samples: tuple = (-4.0, -2.0, 0.0, 2.0)
    k_jerk: float = 0.1
    k_time: float = 0.1
    k_d: float = 1.0
    k_v: float = 1.0
    k_lat: float = 1.0
    k_lon: float = 1.0
    w_comfort: float = 0.5
    fallback_decel: float = 4.0
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)


@dataclass
class Candidate:
    t: np.ndarray
    s: np.ndarray
    s_d: np.ndarray
    d: np.ndarray
    d_d: np.ndarray
    arr: np.ndarray  # (N, 7) t, x, y, yaw, v, a, curvature
    cost: float
    d_end: float
    v_end: float
    T_lat: float
    T_lon: float


@dataclass
class TrajectoryResult:
    trajectory: Trajectory
    feasible: bool
    fallback: bool
    cost: float
    n_candidates: int
    n_checked: int
    report: Optional[ConstraintReport]
    chosen: Optional[Candidate] = None


def _profile_with_hold(poly, T: float, t: np.ndarray, order_kind: str):
    """Evaluate a polynomial up to T and extrapolate after T: lateral holds position,
    longitudinal continues at end velocity."""
    tt = np.minimum(t, T)
    x, v, a, j = poly.x(tt), poly.v(tt), poly.acc(tt), poly.jerk(tt)
    after = t > T
    if order_kind == "lon":
        x = np.where(after, x + v * (t - T), x)
    else:
        v = np.where(after, 0.0, v)
    a = np.where(after, 0.0, a)
    j = np.where(after, 0.0, j)
    return x, v, a, j


def frenet_to_cartesian(ref: ReferencePath, t, s, s_d, d, d_d) -> np.ndarray:
    """(N, 7) array t, x, y, yaw, v, a, curvature from Frenet profiles."""
    x, y = ref.to_cartesian(s, d)
    _, _, ryaw, rk = ref.interpolate(s)
    one_m = np.maximum(1.0 - rk * d, 1e-3)
    vs = s_d * one_m
    v = np.hypot(vs, d_d)
    yaw = ryaw + np.arctan2(d_d, np.maximum(vs, 1e-3))
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    a = np.gradient(v, t) if len(t) > 1 else np.zeros_like(t)
    if len(t) > 1:
        dyaw = np.gradient(np.unwrap(yaw), t)
        k = np.where(v > 0.5, dyaw / np.maximum(v, 1e-3), rk)
    else:
        k = rk
    return np.stack([t, x, y, yaw, v, a, k], axis=1)


class FrenetTrajectoryGenerator:
    def __init__(self, ref: ReferencePath, config: TrajectoryConfig = TrajectoryConfig()) -> None:
        self.ref = ref
        self.config = config
        self.checker = ConstraintChecker(config.constraints)
        self.comfort = ComfortModel()

    def candidates(self, st: FrenetState, v_target: float, d_target: float = 0.0,
                   stop_s: Optional[float] = None) -> list[Candidate]:
        c = self.config
        t = np.arange(0.0, c.horizon_s + 1e-9, c.dt)
        lon_profiles = []
        if stop_s is not None:
            dist = stop_s - st.s
            for T in c.lon_times_s:
                if dist <= 0:
                    break
                q = QuinticPolynomial(st.s, st.s_d, st.s_dd, stop_s, 0.0, 0.0, T)
                s, sv, sa, sj = _profile_with_hold(q, T, t, "lon")
                lon_profiles.append((s, sv, sa, sj, 0.0, T, 0.0))
        else:
            for T in c.lon_times_s:
                for dv in c.dv_samples:
                    v1 = max(0.0, v_target + dv)
                    q = QuarticPolynomial(st.s, st.s_d, st.s_dd, v1, 0.0, T)
                    s, sv, sa, sj = _profile_with_hold(q, T, t, "lon")
                    lon_profiles.append((s, sv, sa, sj, v1, T, (v1 - v_target) ** 2))
        out = []
        for T_lat in c.lat_times_s:
            for off in c.d_offsets:
                d1 = d_target + off
                ql = QuinticPolynomial(st.d, st.d_d, st.d_dd, d1, 0.0, 0.0, T_lat)
                d, dv_, da, dj = _profile_with_hold(ql, T_lat, t, "lat")
                j_lat = c.k_jerk * float(np.sum(dj**2) * c.dt) + c.k_time * T_lat + c.k_d * (d1 - d_target) ** 2
                for s, sv, sa, sj, v1, T_lon, vdev in lon_profiles:
                    j_lon = c.k_jerk * float(np.sum(sj**2) * c.dt) + c.k_time * T_lon + c.k_v * vdev
                    arr = frenet_to_cartesian(self.ref, t, s, sv, d, dv_)
                    comfort = self.comfort.evaluate_array(arr).cost
                    cost = c.k_lat * j_lat + c.k_lon * j_lon + c.w_comfort * comfort
                    out.append(Candidate(t, s, sv, d, dv_, arr, cost, d1, v1, T_lat, T_lon))
        out.sort(key=lambda cd: cd.cost)
        return out

    def fallback(self, st: FrenetState) -> np.ndarray:
        """Brake at `fallback_decel` along the current offset (lateral velocity damped)."""
        c = self.config
        t = np.arange(0.0, c.horizon_s + 1e-9, c.dt)
        v0 = max(st.s_d, 0.0)
        t_stop = v0 / c.fallback_decel
        tt = np.minimum(t, t_stop)
        s = st.s + v0 * tt - 0.5 * c.fallback_decel * tt**2
        sv = np.maximum(v0 - c.fallback_decel * t, 0.0)
        d = np.full_like(t, st.d)
        return frenet_to_cartesian(self.ref, t, s, sv, d, np.zeros_like(t))

    def generate(self, st: FrenetState, v_target: float, d_target: float = 0.0,
                 obstacles: Optional[list[ObstacleBox]] = None, stamp: float = 0.0,
                 frame_id: str = "map", stop_s: Optional[float] = None,
                 max_checks: int = 200) -> TrajectoryResult:
        cands = self.candidates(st, v_target, d_target, stop_s)
        checked = 0
        last_rep = None
        for cd in cands[:max_checks]:
            checked += 1
            rep = self.checker.check_array(cd.arr, obstacles, stop_at_first=True)
            last_rep = rep
            if rep.feasible:
                return TrajectoryResult(to_trajectory(cd.arr, stamp, frame_id), True, False, cd.cost, len(cands), checked, rep, cd)
        arr = self.fallback(st)
        return TrajectoryResult(to_trajectory(arr, stamp, frame_id), False, True, float("inf"), len(cands), checked, last_rep)

    # TODO(planning): dynamic (time-indexed) obstacle prediction from the prediction layer,
    # low-speed mode (lateral as function of s when v ~ 0), warm-start / temporal
    # consistency cost against the previous trajectory, vectorised candidate evaluation.


def to_trajectory(arr: np.ndarray, stamp: float = 0.0, frame_id: str = "map", source: str = "planning") -> Trajectory:
    pts = [TrajectoryPoint(float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[6])) for r in arr]
    return Trajectory(Header(stamp=stamp, frame_id=frame_id, source=source), pts)
