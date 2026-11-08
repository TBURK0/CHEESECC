"""Camera ingest: frame sources, a bounded ring buffer, and drop/latency stats.

A `FrameSource` yields `Frame`s (header + image). `CameraIngest` pulls from a
source on each tick, stamps arrival time from a `core.Clock`, pushes into a
`FrameRingBuffer`, detects dropped frames from sequence gaps / timestamp gaps,
and tracks capture->arrival latency. Optionally publishes to a `MessageBus`.

Sources provided:
- `SyntheticSource`: deterministic procedural images (gradient + moving bar + noise).
- `ArraySource`: frames from an in-memory array/list (replay, tests).
- `NpyDirectorySource`: frames from a directory of `.npy` files (sorted by name).

# TODO(perception): AVFoundation/iPhone stream bridge (from ../theapp via comms),
#   hardware timestamps + clock-offset estimation, video decode (H.264/HEVC),
#   zero-copy buffers, exposure/ISO metadata passthrough.
"""

from __future__ import annotations

import abc
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np

from cheesestack.core import Clock, Header, MessageBus, Node


@dataclass
class Frame:
    """One camera image. `header.stamp` is capture time; `arrival` is when ingest saw it."""

    header: Header
    image: np.ndarray
    arrival: float = float("nan")
    meta: dict = field(default_factory=dict)

    @property
    def stamp(self) -> float:
        return self.header.stamp

    @property
    def shape(self) -> tuple:
        return self.image.shape


class FrameSource(abc.ABC):
    """Pull-based frame source. `read()` returns the next frame or None if none ready/exhausted."""

    name: str = "camera"

    @abc.abstractmethod
    def read(self, now: float) -> Optional[Frame]:
        """Return next frame available at time `now` (or None)."""

    def exhausted(self) -> bool:
        return False

    def close(self) -> None:  # pragma: no cover - default no-op
        pass


class SyntheticSource(FrameSource):
    """Deterministic procedural camera at `fps`. Frames are float32 [0,1].

    `drop_every`: simulate a dropped frame every N captures (seq increments, no frame).
    `latency`: capture->delivery delay in seconds.
    """

    def __init__(self, width: int = 64, height: int = 48, fps: float = 30.0, channels: int = 3,
                 noise: float = 0.01, seed: int = 0, latency: float = 0.0, drop_every: int = 0,
                 name: str = "camera_front", t0: float = 0.0) -> None:
        self.width, self.height, self.fps, self.channels = width, height, fps, channels
        self.noise, self.latency, self.drop_every = noise, latency, drop_every
        self.name = name
        self._rng = np.random.default_rng(seed)
        self._seq = 0
        self._t0 = t0

    def _render(self, seq: int) -> np.ndarray:
        h, w = self.height, self.width
        vv, uu = np.mgrid[0:h, 0:w].astype(np.float32)
        img = 0.2 + 0.5 * (vv / max(h - 1, 1))  # sky-to-road gradient
        bar = int((seq * 2) % w)
        img[:, max(0, bar - 1):bar + 2] = 0.95  # moving vertical bar
        img = img + self._rng.normal(0.0, self.noise, img.shape).astype(np.float32)
        img = np.clip(img, 0.0, 1.0)
        if self.channels == 3:
            img = np.stack([img, img * 0.98, img * 0.95], axis=-1)
        return img.astype(np.float32)

    def read(self, now: float) -> Optional[Frame]:
        capture_t = self._t0 + self._seq / self.fps
        if capture_t + self.latency > now + 1e-9:
            return None
        seq = self._seq
        self._seq += 1
        if self.drop_every and seq > 0 and seq % self.drop_every == 0:
            return None  # dropped at the sensor; gap visible in seq
        hdr = Header(stamp=capture_t, frame_id=self.name, seq=seq, source=self.name)
        return Frame(hdr, self._render(seq))


class ArraySource(FrameSource):
    """Replay frames from a sequence of images with given timestamps (default 30 fps)."""

    def __init__(self, images: Sequence[np.ndarray] | np.ndarray, stamps: Optional[Sequence[float]] = None,
                 name: str = "camera_front") -> None:
        self.images = list(images)
        self.stamps = list(stamps) if stamps is not None else [i / 30.0 for i in range(len(self.images))]
        if len(self.stamps) != len(self.images):
            raise ValueError("images and stamps length mismatch")
        self.name = name
        self._i = 0

    def read(self, now: float) -> Optional[Frame]:
        if self._i >= len(self.images) or self.stamps[self._i] > now + 1e-9:
            return None
        i = self._i
        self._i += 1
        return Frame(Header(float(self.stamps[i]), self.name, i, self.name), np.asarray(self.images[i]))

    def exhausted(self) -> bool:
        return self._i >= len(self.images)


class NpyDirectorySource(ArraySource):
    """Frames from `<dir>/*.npy`, sorted by filename. Stamps from optional `stamps.txt` (one per line)."""

    def __init__(self, directory: str, fps: float = 30.0, name: str = "camera_front") -> None:
        files = sorted(f for f in os.listdir(directory) if f.endswith(".npy"))
        images = [np.load(os.path.join(directory, f)) for f in files]  # TODO(perception): lazy/mmap loading
        stamps_path = os.path.join(directory, "stamps.txt")
        if os.path.exists(stamps_path):
            with open(stamps_path) as fh:
                stamps = [float(x) for x in fh.read().split()]
        else:
            stamps = [i / fps for i in range(len(images))]
        super().__init__(images, stamps, name)


class FrameRingBuffer:
    """Fixed-capacity FIFO of frames; oldest evicted on overflow (counted as overflow drops)."""

    def __init__(self, capacity: int = 8) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._q: deque[Frame] = deque()
        self.overflow_drops = 0

    def push(self, frame: Frame) -> Optional[Frame]:
        evicted = None
        if len(self._q) >= self.capacity:
            evicted = self._q.popleft()
            self.overflow_drops += 1
        self._q.append(frame)
        return evicted

    def pop(self) -> Optional[Frame]:
        return self._q.popleft() if self._q else None

    def latest(self) -> Optional[Frame]:
        return self._q[-1] if self._q else None

    def closest(self, stamp: float) -> Optional[Frame]:
        if not self._q:
            return None
        return min(self._q, key=lambda f: abs(f.stamp - stamp))

    def __len__(self) -> int:
        return len(self._q)

    def __iter__(self) -> Iterator[Frame]:
        return iter(list(self._q))

    def clear(self) -> None:
        self._q.clear()


@dataclass
class IngestStats:
    received: int = 0
    dropped_seq: int = 0  # gaps in sequence numbers
    dropped_time: int = 0  # gaps in stamps > period * gap_factor (when seq unavailable/unreliable)
    overflow: int = 0
    out_of_order: int = 0
    latency_mean: float = 0.0
    latency_max: float = 0.0
    fps_estimate: float = 0.0
    _lat_sum: float = 0.0

    @property
    def drop_rate(self) -> float:
        total = self.received + self.dropped_seq
        return self.dropped_seq / total if total else 0.0

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")} | {"drop_rate": self.drop_rate}


@dataclass
class CameraIngestConfig:
    buffer_capacity: int = 8
    expected_fps: float = 30.0
    gap_factor: float = 1.5  # a stamp gap > gap_factor/expected_fps counts as a time drop
    max_frames_per_tick: int = 16
    fps_ema_alpha: float = 0.1
    topic: str = "/perception/camera/front/image"


class CameraIngest(Node):
    """Pulls frames from a source each tick into a ring buffer and maintains `IngestStats`."""

    def __init__(self, source: FrameSource, config: Optional[CameraIngestConfig] = None,
                 bus: Optional[MessageBus] = None, clock: Optional[Clock] = None, publish: bool = False) -> None:
        super().__init__(f"camera_ingest[{source.name}]", bus, clock)
        self.source = source
        self.config = config or CameraIngestConfig()
        self.rate_hz = self.config.expected_fps
        self.buffer = FrameRingBuffer(self.config.buffer_capacity)
        self.stats = IngestStats()
        self.publish = publish
        self._last: Optional[Frame] = None

    def _account(self, frame: Frame) -> None:
        s, c = self.stats, self.config
        s.received += 1
        lat = frame.arrival - frame.stamp
        s._lat_sum += lat
        s.latency_mean = s._lat_sum / s.received
        s.latency_max = max(s.latency_max, lat)
        if self._last is not None:
            dseq = frame.header.seq - self._last.header.seq
            dt = frame.stamp - self._last.stamp
            if dseq <= 0 or dt <= 0:
                s.out_of_order += 1
            else:
                s.dropped_seq += dseq - 1
                if dt > c.gap_factor / c.expected_fps:
                    s.dropped_time += max(1, int(round(dt * c.expected_fps)) - 1)
                inst = 1.0 / dt
                s.fps_estimate = inst if s.fps_estimate == 0 else (1 - c.fps_ema_alpha) * s.fps_estimate + c.fps_ema_alpha * inst
        self._last = frame

    def poll(self, now: Optional[float] = None) -> list[Frame]:
        """Drain everything the source has ready at `now`; returns new frames."""
        now = self.clock.now() if now is None else now
        out = []
        for _ in range(self.config.max_frames_per_tick):
            f = self.source.read(now)
            if f is None:
                break
            f.arrival = now
            self._account(f)
            if self.buffer.push(f) is not None:
                self.stats.overflow += 1
            out.append(f)
            if self.publish:
                self.bus.publish(self.config.topic, f)
        return out

    def step(self, t: float) -> None:
        self.poll(t)

    def latest(self) -> Optional[Frame]:
        return self.buffer.latest()


def frames_from(images: Iterable[np.ndarray], fps: float = 30.0, name: str = "camera_front") -> list[Frame]:
    """Convenience: wrap raw images as frames with synthetic stamps."""
    return [Frame(Header(i / fps, name, i, name), np.asarray(im)) for i, im in enumerate(images)]
