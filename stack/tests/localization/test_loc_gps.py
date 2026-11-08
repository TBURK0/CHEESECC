import math

import numpy as np
import pytest

from cheesestack.localization import gps as G


def _sentence(body: str) -> str:
    return f"${body}*{G.nmea_checksum(body):02X}"


def test_parse_gga_and_checksum():
    s = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47"
    f = G.parse_nmea(s)
    assert isinstance(f, G.GgaFix)
    assert f.lat_deg == pytest.approx(48 + 7.038 / 60)
    assert f.lon_deg == pytest.approx(11 + 31.0 / 60)
    assert f.quality == G.FixQuality.GPS and f.num_sats == 8 and f.hdop == 0.9
    assert f.utc_s == pytest.approx(12 * 3600 + 35 * 60 + 19)
    assert G.parse_nmea(s[:-2] + "00") is None  # bad checksum


def test_parse_rmc_southwest():
    body = "GNRMC,010203.50,A,3907.974,N,08430.900,W,10.0,90.0,081126,,"
    f = G.parse_nmea(_sentence(body))
    assert isinstance(f, G.RmcFix) and f.valid and f.talker == "GN"
    assert f.lon_deg < 0 and f.speed_mps == pytest.approx(5.14444)
    assert G.bearing_deg_to_yaw(f.course_deg) == pytest.approx(0.0, abs=1e-12)  # due east


def test_wgs84_roundtrip_and_equirect_agree():
    o = G.GeoPoint(39.1329, -84.5150, 200.0)
    ltp = G.LocalTangentPlane(o)
    enu = ltp.to_enu(39.1339, -84.5130, 205.0)
    back = ltp.to_geodetic(*enu)
    assert back.lat_deg == pytest.approx(39.1339, abs=1e-9)
    assert back.lon_deg == pytest.approx(-84.5130, abs=1e-9)
    assert back.alt_m == pytest.approx(205.0, abs=1e-3)
    approx = G.equirectangular_to_local(39.1339, -84.5130, o)
    assert np.allclose(enu[:2], approx, atol=0.6)  # ~200 m away, sphere vs ellipsoid
    assert enu[0] > 0 and enu[1] > 0


def test_covariance_models():
    assert G.hdop_to_covariance(1.0, G.FixQuality.RTK_FIXED)[0, 0] < G.hdop_to_covariance(1.0, G.FixQuality.GPS)[0, 0]
    assert G.hdop_to_covariance(2.0)[0, 0] == pytest.approx(4 * G.hdop_to_covariance(1.0)[0, 0])
    assert G.horizontal_accuracy_to_covariance(-1)[0, 0] > 1e5


def test_processor_gating():
    p = G.GpsProcessor(G.GpsConfig(origin=G.GeoPoint(39.0, -84.0)))
    m = p.process(G.GpsFix(0.0, 39.0, -84.0))
    assert m.accepted and np.allclose(m.xy, 0, atol=1e-6)
    jump = p.process(G.GpsFix(1.0, 39.01, -84.0))  # ~1.1 km in 1 s
    assert not jump.accepted and jump.reason == "speed_jump"
    gated = p.process(G.GpsFix(2.0, 39.0001, -84.0), predicted_xy=[0, 50], predicted_cov=np.eye(2))
    assert not gated.accepted and gated.reason == "innovation_gate"
    ok = p.process(G.GpsFix(2.0, 39.0001, -84.0), predicted_xy=[0, 11], predicted_cov=np.eye(2))
    assert ok.accepted
    assert not p.process(G.GpsFix(3.0, 39.0, -84.0, quality=G.FixQuality.INVALID)).accepted
    assert not p.process(G.GpsFix(3.0, 39.0, -84.0, hdop=30)).accepted


def test_fix_from_nmea():
    gga = G.parse_nmea("$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47")
    fix = G.fix_from_nmea(gga, stamp=5.0)
    assert fix.stamp == 5.0 and fix.hdop == 0.9
