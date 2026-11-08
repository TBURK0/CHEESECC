"""GPS integration: NMEA parsing, WGS84 <-> local ENU, fix-quality -> covariance, outlier gating.

Frames: the `map` frame is an ENU local tangent plane anchored at an origin
lat/lon/alt. x = east, y = north, so stack yaw (CCW from +x) relates to a
compass bearing / course-over-ground (clockwise from north, as used by the
app's GeoMath) by ``yaw = pi/2 - bearing``.

What works now:
- NMEA 0183 GGA and RMC parsing (any talker id: GP, GN, GL, GA...) with checksum check.
- Exact WGS84 geodetic <-> ECEF <-> ENU, plus the app's equirectangular
  approximation (`equirectangular_to_local`) for parity with GeoMath.toMeters.
- HDOP / fix quality / phone horizontal accuracy -> 2x2 covariance.
- Chi-square innovation gating against a predicted position and a
  max-speed plausibility check against the last accepted fix.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from cheesestack.core import Header

# --- WGS84 -----------------------------------------------------------------
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_EP2 = (WGS84_A**2 - WGS84_B**2) / WGS84_B**2
APP_EARTH_RADIUS_M = 6_371_000.0  # matches theapp GeoMath.earthRadiusM
KNOTS_TO_MPS = 0.514444

# 2-DoF chi-square quantiles used for gating.
CHI2_2DOF = {0.90: 4.605, 0.95: 5.991, 0.99: 9.210, 0.999: 13.816}


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sl, cl = math.sin(lat), math.cos(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sl * sl)
    return np.array([
        (n + alt_m) * cl * math.cos(lon),
        (n + alt_m) * cl * math.sin(lon),
        (n * (1.0 - WGS84_E2) + alt_m) * sl,
    ])


def ecef_to_geodetic(ecef) -> tuple[float, float, float]:
    """ECEF -> (lat_deg, lon_deg, alt_m). Bowring initial guess + 3 Newton-style refinements (sub-mm)."""
    x, y, z = (float(v) for v in ecef)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    theta = math.atan2(z * WGS84_A, p * WGS84_B)
    lat = math.atan2(z + WGS84_EP2 * WGS84_B * math.sin(theta) ** 3,
                     p - WGS84_E2 * WGS84_A * math.cos(theta) ** 3)
    for _ in range(3):
        sl = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sl * sl)
        alt = p / math.cos(lat) - n if abs(math.cos(lat)) > 1e-12 else abs(z) - WGS84_B
        lat = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + alt)))
    sl = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sl * sl)
    alt = p / math.cos(lat) - n if abs(math.cos(lat)) > 1e-12 else abs(z) - WGS84_B
    return math.degrees(lat), math.degrees(lon), alt


def _ecef_to_enu_rot(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sl, cl, so, co = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
    return np.array([
        [-so, co, 0.0],
        [-sl * co, -sl * so, cl],
        [cl * co, cl * so, sl],
    ])


@dataclass(frozen=True)
class GeoPoint:
    lat_deg: float
    lon_deg: float
    alt_m: float = 0.0


class LocalTangentPlane:
    """Exact WGS84 ENU projection about a fixed origin. Use for the `map` frame."""

    def __init__(self, origin: GeoPoint) -> None:
        self.origin = origin
        self._ecef0 = geodetic_to_ecef(origin.lat_deg, origin.lon_deg, origin.alt_m)
        self._R = _ecef_to_enu_rot(origin.lat_deg, origin.lon_deg)

    def to_enu(self, lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
        return self._R @ (geodetic_to_ecef(lat_deg, lon_deg, alt_m) - self._ecef0)

    def to_geodetic(self, e: float, n: float, u: float = 0.0) -> GeoPoint:
        ecef = self._R.T @ np.array([e, n, u]) + self._ecef0
        return GeoPoint(*ecef_to_geodetic(ecef))


def equirectangular_to_local(lat_deg: float, lon_deg: float, origin: GeoPoint) -> np.ndarray:
    """Same approximation as the app's GeoMath.toMeters (spherical, R=6371 km). x=east, y=north."""
    lat0 = math.radians(origin.lat_deg)
    x = math.radians(lon_deg - origin.lon_deg) * math.cos(lat0) * APP_EARTH_RADIUS_M
    y = math.radians(lat_deg - origin.lat_deg) * APP_EARTH_RADIUS_M
    return np.array([x, y])


def bearing_deg_to_yaw(bearing_deg: float) -> float:
    """Compass bearing (deg, CW from north) -> stack yaw (rad, CCW from east), wrapped."""
    from cheesestack.core import wrap_angle

    return wrap_angle(math.pi / 2.0 - math.radians(bearing_deg))


def yaw_to_bearing_deg(yaw: float) -> float:
    return (90.0 - math.degrees(yaw)) % 360.0


# --- NMEA ------------------------------------------------------------------
class FixQuality(enum.IntEnum):
    """GGA field 6."""

    INVALID = 0
    GPS = 1
    DGPS = 2
    PPS = 3
    RTK_FIXED = 4
    RTK_FLOAT = 5
    DEAD_RECKONING = 6
    MANUAL = 7
    SIMULATION = 8


@dataclass
class GgaFix:
    utc_s: float  # seconds since UTC midnight
    lat_deg: float
    lon_deg: float
    quality: FixQuality
    num_sats: int
    hdop: float
    alt_m: float
    geoid_sep_m: float = 0.0
    talker: str = "GP"


@dataclass
class RmcFix:
    utc_s: float
    valid: bool
    lat_deg: float
    lon_deg: float
    speed_mps: float
    course_deg: Optional[float]  # CW from true north; None if empty
    date: str = ""  # ddmmyy
    talker: str = "GP"


def nmea_checksum(body: str) -> int:
    cs = 0
    for ch in body:
        cs ^= ord(ch)
    return cs


def _parse_latlon(value: str, hemi: str, deg_digits: int) -> float:
    if not value:
        raise ValueError("empty coordinate")
    deg = float(value[:deg_digits])
    minutes = float(value[deg_digits:])
    out = deg + minutes / 60.0
    return -out if hemi in ("S", "W") else out


def _parse_utc(value: str) -> float:
    if len(value) < 6:
        return float("nan")
    return int(value[0:2]) * 3600.0 + int(value[2:4]) * 60.0 + float(value[4:])


def parse_nmea(line: str, require_checksum: bool = True) -> Optional[Union[GgaFix, RmcFix]]:
    """Parse one NMEA sentence. Returns None for unsupported/invalid/bad-checksum sentences."""
    line = line.strip()
    if not line.startswith("$"):
        return None
    if "*" in line:
        body, cs = line[1:].split("*", 1)
        try:
            if int(cs[:2], 16) != nmea_checksum(body):
                return None
        except ValueError:
            return None
    else:
        if require_checksum:
            return None
        body = line[1:]
    f = body.split(",")
    if len(f[0]) != 5:
        return None
    talker, kind = f[0][:2], f[0][2:]
    try:
        if kind == "GGA" and len(f) >= 10:
            q = FixQuality(int(f[6] or 0))
            if q == FixQuality.INVALID or not f[2]:
                return GgaFix(_parse_utc(f[1]), float("nan"), float("nan"), q, 0, float("inf"), float("nan"), talker=talker)
            return GgaFix(
                utc_s=_parse_utc(f[1]),
                lat_deg=_parse_latlon(f[2], f[3], 2),
                lon_deg=_parse_latlon(f[4], f[5], 3),
                quality=q,
                num_sats=int(f[7] or 0),
                hdop=float(f[8]) if f[8] else float("inf"),
                alt_m=float(f[9]) if f[9] else 0.0,
                geoid_sep_m=float(f[11]) if len(f) > 11 and f[11] else 0.0,
                talker=talker,
            )
        if kind == "RMC" and len(f) >= 10:
            valid = f[2] == "A"
            if not f[3]:
                return RmcFix(_parse_utc(f[1]), False, float("nan"), float("nan"), 0.0, None, f[9], talker)
            return RmcFix(
                utc_s=_parse_utc(f[1]),
                valid=valid,
                lat_deg=_parse_latlon(f[3], f[4], 2),
                lon_deg=_parse_latlon(f[5], f[6], 3),
                speed_mps=float(f[7] or 0.0) * KNOTS_TO_MPS,
                course_deg=float(f[8]) if f[8] else None,
                date=f[9],
                talker=talker,
            )
    except (ValueError, IndexError):
        return None
    return None


# --- covariance ----------------------------------------------------------------
# Per-axis 1-sigma user-equivalent range error (m) by fix quality. Rough, tune per receiver.
DEFAULT_UERE_M = {
    FixQuality.GPS: 2.5,
    FixQuality.DGPS: 0.8,
    FixQuality.PPS: 2.0,
    FixQuality.RTK_FIXED: 0.02,
    FixQuality.RTK_FLOAT: 0.3,
    FixQuality.DEAD_RECKONING: 10.0,
    FixQuality.MANUAL: 5.0,
    FixQuality.SIMULATION: 1.0,
}


def hdop_to_covariance(hdop: float, quality: FixQuality = FixQuality.GPS,
                       uere_m: Optional[dict] = None, min_sigma_m: float = 0.01) -> np.ndarray:
    """sigma_axis = HDOP * UERE(quality); isotropic 2x2 covariance (east, north)."""
    table = uere_m or DEFAULT_UERE_M
    if quality == FixQuality.INVALID or not math.isfinite(hdop):
        return np.eye(2) * 1e6
    sigma = max(min_sigma_m, hdop * table.get(quality, 5.0))
    return np.eye(2) * sigma**2


def horizontal_accuracy_to_covariance(accuracy_m: float) -> np.ndarray:
    """Phone (CoreLocation-style) horizontal accuracy radius -> covariance.

    Treats the radius as ~68% of a 2D circular Gaussian (radius ~= 1.515 sigma).
    """
    if accuracy_m is None or accuracy_m <= 0 or not math.isfinite(accuracy_m):
        return np.eye(2) * 1e6
    sigma = accuracy_m / 1.515
    return np.eye(2) * sigma**2


def mahalanobis_sq(innovation, S) -> float:
    innovation = np.asarray(innovation, dtype=float)
    return float(innovation @ np.linalg.solve(S, innovation))


# --- processor -------------------------------------------------------------------
@dataclass
class GpsFix:
    """Sensor-agnostic fix (from NMEA, the phone, or sim)."""

    stamp: float
    lat_deg: float
    lon_deg: float
    alt_m: float = 0.0
    hdop: float = 1.0
    quality: FixQuality = FixQuality.GPS
    horizontal_accuracy_m: Optional[float] = None  # if set, overrides HDOP model
    speed_mps: Optional[float] = None
    course_deg: Optional[float] = None


@dataclass
class GpsMeasurement:
    """Fix projected into `map` (ENU) with covariance and gating verdict."""

    header: Header
    xy: np.ndarray
    cov: np.ndarray
    accepted: bool
    reason: str = "ok"
    speed_mps: Optional[float] = None
    course_yaw: Optional[float] = None  # stack yaw from course over ground
    nis: Optional[float] = None


@dataclass
class GpsConfig:
    origin: Optional[GeoPoint] = None  # None -> first accepted fix becomes origin
    use_exact_wgs84: bool = True  # False -> app-compatible equirectangular
    accepted_qualities: frozenset = frozenset({FixQuality.GPS, FixQuality.DGPS, FixQuality.PPS,
                                               FixQuality.RTK_FIXED, FixQuality.RTK_FLOAT,
                                               FixQuality.SIMULATION})
    max_hdop: float = 10.0
    max_accuracy_m: float = 50.0
    gate_probability: float = 0.999
    max_speed_mps: float = 60.0  # plausibility vs last accepted fix
    speed_jump_margin_m: float = 5.0
    frame_id: str = "map"


class GpsProcessor:
    """Turns raw fixes into gated ENU measurements for the state estimator."""

    def __init__(self, config: Optional[GpsConfig] = None) -> None:
        self.config = config or GpsConfig()
        self._ltp: Optional[LocalTangentPlane] = None
        self._last: Optional[GpsMeasurement] = None
        self._seq = 0
        if self.config.origin is not None:
            self._ltp = LocalTangentPlane(self.config.origin)

    @property
    def origin(self) -> Optional[GeoPoint]:
        return self._ltp.origin if self._ltp else None

    def project(self, lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
        assert self._ltp is not None, "origin not set"
        if self.config.use_exact_wgs84:
            return self._ltp.to_enu(lat_deg, lon_deg, alt_m)[:2]
        return equirectangular_to_local(lat_deg, lon_deg, self._ltp.origin)

    def process(self, fix: GpsFix, predicted_xy=None, predicted_cov=None) -> GpsMeasurement:
        cfg = self.config
        self._seq += 1
        hdr = Header(stamp=fix.stamp, frame_id=cfg.frame_id, seq=self._seq, source="gps")
        if fix.horizontal_accuracy_m is not None:
            cov = horizontal_accuracy_to_covariance(fix.horizontal_accuracy_m)
        else:
            cov = hdop_to_covariance(fix.hdop, fix.quality)
        course = None
        if fix.course_deg is not None and (fix.speed_mps or 0.0) > 0.5:
            course = bearing_deg_to_yaw(fix.course_deg)

        def reject(reason: str, xy=None) -> GpsMeasurement:
            return GpsMeasurement(hdr, np.full(2, np.nan) if xy is None else xy, cov, False, reason,
                                  fix.speed_mps, course)

        if not (math.isfinite(fix.lat_deg) and math.isfinite(fix.lon_deg)):
            return reject("no_position")
        if fix.quality not in cfg.accepted_qualities:
            return reject("low_quality")
        if fix.horizontal_accuracy_m is None and fix.hdop > cfg.max_hdop:
            return reject("high_hdop")
        if fix.horizontal_accuracy_m is not None and fix.horizontal_accuracy_m > cfg.max_accuracy_m:
            return reject("low_accuracy")
        if self._ltp is None:
            self._ltp = LocalTangentPlane(GeoPoint(fix.lat_deg, fix.lon_deg, fix.alt_m))
        xy = self.project(fix.lat_deg, fix.lon_deg, fix.alt_m)

        nis = None
        if predicted_xy is not None:
            P = np.asarray(predicted_cov if predicted_cov is not None else np.zeros((2, 2)), dtype=float)
            nis = mahalanobis_sq(xy - np.asarray(predicted_xy, dtype=float), P + cov)
            thr = CHI2_2DOF.get(cfg.gate_probability, CHI2_2DOF[0.999])
            if nis > thr:
                m = reject("innovation_gate", xy)
                m.nis = nis
                return m
        elif self._last is not None:
            dt = fix.stamp - self._last.header.stamp
            if dt > 0:
                dist = float(np.linalg.norm(xy - self._last.xy))
                if dist > cfg.max_speed_mps * dt + cfg.speed_jump_margin_m:
                    return reject("speed_jump", xy)
        meas = GpsMeasurement(hdr, xy, cov, True, "ok", fix.speed_mps, course, nis)
        self._last = meas
        return meas
    # TODO(localization): RAIM-style multi-hypothesis rejection, multipath detection near buildings,
    # lever-arm compensation (gps antenna -> base_link) using the FrameTree, and GPS week/UTC date handling.


def fix_from_nmea(gga: GgaFix, rmc: Optional[RmcFix] = None, stamp: Optional[float] = None) -> GpsFix:
    """Combine a GGA (position/quality) with an optional RMC (speed/course) into a GpsFix."""
    return GpsFix(
        stamp=gga.utc_s if stamp is None else stamp,
        lat_deg=gga.lat_deg,
        lon_deg=gga.lon_deg,
        alt_m=gga.alt_m,
        hdop=gga.hdop,
        quality=gga.quality,
        speed_mps=rmc.speed_mps if rmc and rmc.valid else None,
        course_deg=rmc.course_deg if rmc and rmc.valid else None,
    )
