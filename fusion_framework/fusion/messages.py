"""Message types: the contract between your existing hardware code and fusion.

Every message carries `t`: the CAPTURE time in seconds on ONE clock (use the
microcontroller's clock and convert camera timestamps onto it). Never stamp
with arrival time on the host; arrival time includes USB/processing delay.

All vectors are in the sensor's own frame as read from hardware. Rotating
into the body frame happens inside fusion, using mount calibration from the
config, so raw logs stay raw and can be re-processed when calibration changes.

To add a message type: define a dataclass here and add it to MESSAGE_TYPES.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Optional


@dataclass
class ImuSample:
    """MPU/ITG gyro + accel, one sample."""
    t: float
    gyro: tuple[float, float, float]      # rad/s
    accel: tuple[float, float, float]     # m/s^2, reads about +9.81 up at rest
    temp_c: Optional[float] = None        # bias drifts with temperature; log it


@dataclass
class AccelSample:
    """One of the extra 3-axis accelerometers."""
    t: float
    unit: str                             # name matching config extra_accels
    accel: tuple[float, float, float]     # m/s^2


@dataclass
class UltrasonicReading:
    t: float                              # time the ping was fired
    sensor: str                           # name matching config ultrasonics
    range_m: Optional[float]              # None = no echo


@dataclass
class WheelSpeed:
    t: float
    ticks: int                            # cumulative encoder count (log raw!)
    speed_mps: Optional[float] = None     # filled once Step 4 gives a scale


@dataclass
class DepthFrameInfo:
    """Metadata for a depth/RGB frame. Store the pixels separately (e.g. a
    .npz or video file per run) and reference them by index, so the main log
    stays small enough to replay quickly."""
    t: float
    frame_index: int
    file: str = ""


@dataclass
class VoPose:
    """Visual odometry output, in the VO's own frame."""
    t: float                              # capture time of the frame, not result time
    x: float
    y: float
    yaw: float
    reset: bool = False                   # True on first frame after tracking loss
    t_result: Optional[float] = None      # when the result was ready (for latency)


@dataclass
class Command:
    """What was sent to the car: needed to evaluate control and safety."""
    t: float
    steer_rad: float
    throttle: float                       # -1..1, or speed setpoint, document which
    source: str = "manual"                # manual / autonomous / estop


@dataclass
class Event:
    """Free-form marker: 'tap', 'start loop', 'covered lens', ... Press a
    button or type a note during tests so you can find moments in the log."""
    t: float
    text: str


MESSAGE_TYPES = {cls.__name__: cls for cls in
                 (ImuSample, AccelSample, UltrasonicReading, WheelSpeed,
                  DepthFrameInfo, VoPose, Command, Event)}


def to_dict(msg) -> dict:
    d = asdict(msg)
    d["_type"] = type(msg).__name__
    return d


def from_dict(d: dict):
    d = dict(d)
    cls = MESSAGE_TYPES[d.pop("_type")]
    kwargs = {}
    for f in fields(cls):
        if f.name in d:
            v = d[f.name]
            kwargs[f.name] = tuple(v) if isinstance(v, list) else v
    return cls(**kwargs)
