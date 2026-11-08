"""Sensor simulation: GPS, IMU, wheel speed, object-list sensor.

Every sensor consumes a `TruthSnapshot` (ego twin state + actor truths) and
returns a measurement message or None (not due / dropped). Each sensor owns a
deterministic `np.random.Generator` (pass `world.rng(sensor_name)`).

Noise models:
  GPS        white noise + first-order Gauss-Markov bias, Markov dropout bursts,
             scheduled outages (tunnels).
  IMU        accel/gyro white noise + random-walk bias, gravity on z.
  WheelSpeed per-wheel speed from v and yaw rate, scale error, noise, quantisation.
  ObjectList range/FOV gating, range-dependent detection probability and
             position noise, Poisson false positives, class confusion.

# TODO(sim): lever arms (antenna/IMU offsets), latency/timestamp jitter, multipath,
# radar/lidar point models, occlusion between actors, geodetic (lat/lon) output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cheesestack.core import BoundingBox3D, Detection, Header, ObjectClass, Pose2D, VehicleState

from .physics import ActorTruth

G = 9.81


@dataclass
class TruthSnapshot:
    t: float
    ego: VehicleState
    ego_accel_body: np.ndarray = field(default_factory=lambda: np.zeros(2))  # CG (ax, ay)
    actors: list[ActorTruth] = field(default_factory=list)
    track: float = 1.6


# ------------------------------------------------------------------ messages
@dataclass
class GpsFix:
    header: Header
    x: float
    y: float
    std: float
    valid: bool = True


@dataclass
class ImuSample:
    header: Header
    accel: np.ndarray  # (ax, ay, az) m/s^2 body
    gyro: np.ndarray  # (wx, wy, wz) rad/s body


@dataclass
class WheelSpeeds:
    header: Header
    fl: float
    fr: float
    rl: float
    rr: float

    @property
    def mean(self) -> float:
        return 0.25 * (self.fl + self.fr + self.rl + self.rr)


@dataclass
class ObjectList:
    header: Header
    detections: list[Detection]


# ------------------------------------------------------------------- sensors
class Sensor:
    """Base: rate gating + RNG. Subclasses implement `measure`."""

    topic: str = "/sensors/unknown"

    def __init__(self, rate_hz: float, rng: Optional[np.random.Generator] = None, name: str = "") -> None:
        self.rate_hz = rate_hz
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.name = name or type(self).__name__
        self._next_t = -math.inf
        self._seq = 0

    def due(self, t: float) -> bool:
        return t + 1e-9 >= self._next_t

    def sample(self, snap: TruthSnapshot) -> Optional[object]:
        if not self.due(snap.t):
            return None
        period = 1.0 / self.rate_hz
        self._next_t = snap.t + period if self._next_t == -math.inf else self._next_t + period
        self._seq += 1
        return self.measure(snap)

    def header(self, t: float, frame: str) -> Header:
        return Header(stamp=t, frame_id=frame, seq=self._seq, source=f"sim.{self.name}")

    def measure(self, snap: TruthSnapshot):  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class GpsConfig:
    rate_hz: float = 10.0
    noise_std: float = 0.5
    bias_std: float = 1.0
    bias_tau: float = 60.0
    p_dropout_start: float = 0.0  # per sample
    mean_dropout_samples: float = 10.0
    outages: tuple[tuple[float, float], ...] = ()
    report_invalid: bool = False  # emit valid=False fixes instead of None during dropout


class GpsSensor(Sensor):
    topic = "/sensors/gps"

    def __init__(self, config: GpsConfig = GpsConfig(), rng=None) -> None:
        super().__init__(config.rate_hz, rng, "gps")
        self.config = config
        self.bias = self.rng.normal(0.0, config.bias_std, 2) if config.bias_std > 0 else np.zeros(2)
        self.in_dropout = False

    def measure(self, snap: TruthSnapshot):
        c = self.config
        dt = 1.0 / c.rate_hz
        phi = math.exp(-dt / c.bias_tau)
        self.bias = phi * self.bias + c.bias_std * math.sqrt(1 - phi * phi) * self.rng.normal(size=2)
        if self.in_dropout:
            if self.rng.random() < 1.0 / max(1.0, c.mean_dropout_samples):
                self.in_dropout = False
        elif self.rng.random() < c.p_dropout_start:
            self.in_dropout = True
        outage = any(a <= snap.t < b for a, b in c.outages)
        p = snap.ego.pose
        if self.in_dropout or outage:
            if not c.report_invalid:
                return None
            return GpsFix(self.header(snap.t, "map"), math.nan, math.nan, math.inf, valid=False)
        n = self.rng.normal(0.0, c.noise_std, 2)
        std = math.hypot(c.noise_std, c.bias_std)
        return GpsFix(self.header(snap.t, "map"), p.x + self.bias[0] + n[0], p.y + self.bias[1] + n[1], std)


@dataclass
class ImuConfig:
    rate_hz: float = 100.0
    accel_noise_std: float = 0.05
    gyro_noise_std: float = 0.002
    accel_bias_walk: float = 0.001  # m/s^2 per sqrt(s)
    gyro_bias_walk: float = 0.0001  # rad/s per sqrt(s)
    accel_bias0_std: float = 0.05
    gyro_bias0_std: float = 0.002


class ImuSensor(Sensor):
    topic = "/sensors/imu"

    def __init__(self, config: ImuConfig = ImuConfig(), rng=None) -> None:
        super().__init__(config.rate_hz, rng, "imu")
        self.config = config
        self.accel_bias = self.rng.normal(0.0, config.accel_bias0_std, 3)
        self.gyro_bias = self.rng.normal(0.0, config.gyro_bias0_std, 3)

    def measure(self, snap: TruthSnapshot):
        c = self.config
        sq = math.sqrt(1.0 / c.rate_hz)
        self.accel_bias += c.accel_bias_walk * sq * self.rng.normal(size=3)
        self.gyro_bias += c.gyro_bias_walk * sq * self.rng.normal(size=3)
        acc = np.array([snap.ego_accel_body[0], snap.ego_accel_body[1], G])
        gyr = np.array([0.0, 0.0, snap.ego.yaw_rate])
        return ImuSample(
            self.header(snap.t, "imu"),
            acc + self.accel_bias + self.rng.normal(0.0, c.accel_noise_std, 3),
            gyr + self.gyro_bias + self.rng.normal(0.0, c.gyro_noise_std, 3),
        )


@dataclass
class WheelSpeedConfig:
    rate_hz: float = 50.0
    noise_std: float = 0.02
    scale_error: float = 0.005  # tyre radius error fraction
    quantization: float = 0.01  # m/s


class WheelSpeedSensor(Sensor):
    topic = "/sensors/wheel_speed"

    def __init__(self, config: WheelSpeedConfig = WheelSpeedConfig(), rng=None) -> None:
        super().__init__(config.rate_hz, rng, "wheel_speed")
        self.config = config
        self.scale = 1.0 + self.rng.normal(0.0, config.scale_error, 4)

    def measure(self, snap: TruthSnapshot):
        c = self.config
        v, r, half = snap.ego.v, snap.ego.yaw_rate, 0.5 * snap.track
        # TODO(sim): front wheels travel the steered arc; approximate with the rear-axle formula.
        raw = np.array([v - r * half, v + r * half, v - r * half, v + r * half]) * self.scale
        raw = np.maximum(0.0, raw + self.rng.normal(0.0, c.noise_std, 4))
        if c.quantization > 0:
            raw = np.round(raw / c.quantization) * c.quantization
        return WheelSpeeds(self.header(snap.t, "base_link"), *map(float, raw))


@dataclass
class ObjectSensorConfig:
    rate_hz: float = 10.0
    max_range: float = 100.0
    fov_rad: float = math.radians(90.0)
    p_detect_near: float = 0.98
    p_detect_far: float = 0.6  # at max_range (linear in range between)
    pos_noise_std: float = 0.1
    pos_noise_per_m: float = 0.005
    yaw_noise_std: float = 0.03
    false_positive_rate: float = 0.05  # mean count per frame
    p_class_confusion: float = 0.02
    mount: Pose2D = Pose2D(1.5, 0.0, 0.0)  # sensor in base_link


class ObjectListSensor(Sensor):
    """Idealised perception output: object list (Detection with box3d in base_link)."""

    topic = "/sensors/objects"

    def __init__(self, config: ObjectSensorConfig = ObjectSensorConfig(), rng=None) -> None:
        super().__init__(config.rate_hz, rng, "objects")
        self.config = config

    def measure(self, snap: TruthSnapshot):
        c = self.config
        hdr = self.header(snap.t, "base_link")
        world_to_sensor = snap.ego.pose.compose(c.mount).inverse()
        dets: list[Detection] = []
        for a in snap.actors:
            rel_s = world_to_sensor.compose(a.pose)
            rng_m = math.hypot(rel_s.x, rel_s.y)
            bearing = math.atan2(rel_s.y, rel_s.x)
            if rng_m > c.max_range or abs(bearing) > 0.5 * c.fov_rad:
                continue
            p_det = c.p_detect_near + (c.p_detect_far - c.p_detect_near) * rng_m / c.max_range
            if self.rng.random() > p_det:
                continue
            rel = c.mount.compose(rel_s)  # into base_link
            sig = c.pos_noise_std + c.pos_noise_per_m * rng_m
            n = self.rng.normal(0.0, sig, 2)
            pose = Pose2D(rel.x + n[0], rel.y + n[1], rel.yaw + self.rng.normal(0.0, c.yaw_noise_std))
            cls = ObjectClass.UNKNOWN if self.rng.random() < c.p_class_confusion else a.cls
            dets.append(Detection(hdr, cls, float(np.clip(p_det + self.rng.normal(0, 0.05), 0, 1)),
                                  box3d=BoundingBox3D(pose, a.length, a.width, a.height),
                                  attributes={"truth_id": a.name, "range": rng_m, "bearing": bearing}))
        for _ in range(int(self.rng.poisson(c.false_positive_rate))):
            r = self.rng.uniform(2.0, c.max_range)
            b = self.rng.uniform(-0.5, 0.5) * c.fov_rad
            pose = c.mount.compose(Pose2D(r * math.cos(b), r * math.sin(b), self.rng.uniform(-math.pi, math.pi)))
            dets.append(Detection(hdr, ObjectClass.UNKNOWN, float(self.rng.uniform(0.1, 0.5)),
                                  box3d=BoundingBox3D(pose, 1.0, 1.0, 1.0),
                                  attributes={"truth_id": None, "range": r, "bearing": b}))
        return ObjectList(hdr, dets)


def default_sensor_suite(rng_factory) -> list[Sensor]:
    """GPS, IMU, wheel speed and object list, each with its own RNG from `rng_factory(name)`."""
    return [
        GpsSensor(rng=rng_factory("gps")),
        ImuSensor(rng=rng_factory("imu")),
        WheelSpeedSensor(rng=rng_factory("wheel_speed")),
        ObjectListSensor(rng=rng_factory("objects")),
    ]
