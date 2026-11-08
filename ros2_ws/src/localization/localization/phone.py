"""Proposed iOS protocol parser. Transport belongs to the infrastructure agent."""
import json
import math
from datetime import datetime, timezone


class PhoneParser:
    def __init__(self):
        self.last_seq = -1
        self.last_stamp = float('-inf')

    def parse(self, payload, now):
        if len(payload) > 8192:
            raise ValueError('oversized telemetry')
        obj = json.loads(payload)
        if obj.get('schema_version') != 1 or obj.get('type') != 'phone.gps':
            raise ValueError('unsupported schema')
        if obj.get('simulation') is not True:
            raise ValueError('live phone ingestion disabled pending transport review')
        sequence = obj.get('sequence')
        if type(sequence) is not int or sequence <= self.last_seq:
            raise ValueError('replayed sequence')
        raw_stamp = obj['timestamp_utc']
        if not raw_stamp.endswith('Z'):
            raise ValueError('UTC Z timestamp required')
        stamp = datetime.fromisoformat(raw_stamp[:-1]+'+00:00').timestamp()
        if not 0 <= now-stamp <= 1.0 or stamp <= self.last_stamp:
            raise ValueError('stale, future or reordered timestamp')
        values = [obj[k] for k in ('latitude_deg', 'longitude_deg', 'altitude_m', 'horizontal_accuracy_m', 'vertical_accuracy_m')]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError('nonfinite GPS')
        lat, lon, alt, horizontal, vertical = values
        if not -90 <= lat <= 90 or not -180 <= lon <= 180 or not 0 < horizontal <= 10 or not 0 < vertical <= 30:
            raise ValueError('invalid or inaccurate fix')
        self.last_seq, self.last_stamp = sequence, stamp
        return dict(stamp=stamp, latitude=lat, longitude=lon, altitude=alt,
                    covariance=[horizontal**2, horizontal**2, vertical**2])


def local_enu(latitude, longitude, altitude, origin):
    """WGS84 geodetic -> ECEF -> ENU; fixed surveyed origin, meters."""
    def ecef(lat, lon, h):
        lat, lon = math.radians(lat), math.radians(lon)
        n = 6378137.0/math.sqrt(1-6.69437999014e-3*math.sin(lat)**2)
        return ((n+h)*math.cos(lat)*math.cos(lon), (n+h)*math.cos(lat)*math.sin(lon),
                (n*(1-6.69437999014e-3)+h)*math.sin(lat))
    a, b = ecef(latitude, longitude, altitude), ecef(*origin)
    dx, dy, dz = [a[i]-b[i] for i in range(3)]
    lat, lon = math.radians(origin[0]), math.radians(origin[1])
    return (-math.sin(lon)*dx+math.cos(lon)*dy,
            -math.sin(lat)*math.cos(lon)*dx-math.sin(lat)*math.sin(lon)*dy+math.cos(lat)*dz,
            math.cos(lat)*math.cos(lon)*dx+math.cos(lat)*math.sin(lon)*dy+math.sin(lat)*dz)


class MotionParser:
    """Synthetic, already mount-corrected ENU orientation. No raw device axes."""
    def __init__(self):
        self.last_seq, self.last_stamp = -1, float('-inf')

    def parse(self, payload, now):
        if len(payload) > 8192:
            raise ValueError('oversized motion')
        obj = json.loads(payload)
        if obj.get('schema_version') != 1 or obj.get('type') != 'phone.motion' or obj.get('simulation') is not True:
            raise ValueError('unsupported motion schema')
        stamp = obj['timestamp_utc']
        if not stamp.endswith('Z'):
            raise ValueError('UTC required')
        stamp = datetime.fromisoformat(stamp[:-1]+'+00:00').timestamp()
        seq = obj['sequence']
        if type(seq) is not int or seq <= self.last_seq or stamp <= self.last_stamp or not 0 <= now-stamp <= 0.3:
            raise ValueError('invalid motion timestamp/sequence')
        if obj.get('frame') != 'base_link_enu':
            raise ValueError('mount-corrected ENU data required')
        q, rate = obj['orientation_xyzw'], obj['angular_velocity_radps']
        variance = obj['orientation_variance_rad2']
        if len(q) != 4 or len(rate) != 3 or any(type(v) not in (float, int) or not math.isfinite(v) for v in q+rate+[variance]):
            raise ValueError('invalid motion values')
        if abs(sum(v*v for v in q)-1) > 0.01 or not 0 < variance <= 0.1:
            raise ValueError('uncertain orientation')
        self.last_seq, self.last_stamp = seq, stamp
        return dict(stamp=stamp, q=q, rate=rate, variance=variance)
