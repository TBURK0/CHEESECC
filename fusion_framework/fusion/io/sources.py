"""Message sources. Fusion code only ever sees a Source, so the same code runs
live on the car and on a recorded log.

    ReplaySource   reads a RunLogger file, yields messages in time order
    LiveSource     adapter around YOUR existing hardware code (fill in)
"""
from __future__ import annotations

import gzip
import heapq
import json
import time
from abc import ABC, abstractmethod
from typing import Iterator

from ..messages import from_dict


class Source(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator:
        """Yield message objects, roughly in time order."""


def read_header(path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.loads(f.readline())


class ReplaySource(Source):
    """Yields logged messages sorted by capture time.

    Logs are written in ARRIVAL order, which is not capture order (camera
    and VO results arrive late). Replay restores capture order using a small
    reorder buffer, which is also what a live system effectively sees.

    deliver_at_arrival=True instead yields VoPose at its t_result, i.e. late,
    exactly as the live system gets it; use this to test delayed-update code.

    reorder_window_s=None yields messages in raw file order (no sorting).
    """

    def __init__(self, path, realtime: float = 0.0, types: set[str] | None = None,
                 reorder_window_s: float = 1.0, deliver_at_arrival: bool = False):
        self.path = path
        self.header = read_header(path)
        self.realtime = realtime            # 0 = as fast as possible, 1 = wall-clock
        self.types = types
        self.window = reorder_window_s
        self.deliver_at_arrival = deliver_at_arrival

    def _key(self, msg):
        if self.deliver_at_arrival and getattr(msg, "t_result", None) is not None:
            return msg.t_result
        return msg.t

    def __iter__(self):
        heap, seq = [], 0
        t_wall0 = t_log0 = None
        with gzip.open(self.path, "rt", encoding="utf-8") as f:
            f.readline()  # header
            for line in f:
                d = json.loads(line)
                if self.types and d["_type"] not in self.types:
                    continue
                msg = from_dict(d)
                if self.window is None:          # raw file (arrival) order
                    yield msg
                    continue
                k = self._key(msg)
                heapq.heappush(heap, (k, seq, msg))
                seq += 1
                # release everything older than the reorder window
                while heap and heap[0][0] < k - self.window:
                    out = heapq.heappop(heap)
                    t_wall0, t_log0 = self._pace(out[0], t_wall0, t_log0)
                    yield out[2]
            while heap:
                out = heapq.heappop(heap)
                t_wall0, t_log0 = self._pace(out[0], t_wall0, t_log0)
                yield out[2]

    def _pace(self, t_msg, t_wall0, t_log0):
        if self.realtime <= 0:
            return t_wall0, t_log0
        if t_wall0 is None:
            return time.monotonic(), t_msg
        wait = (t_msg - t_log0) / self.realtime - (time.monotonic() - t_wall0)
        if wait > 0:
            time.sleep(wait)
        return t_wall0, t_log0


class LiveSource(Source):
    """Adapter to your existing hardware code.

    TODO (Step 0): wrap whatever your repo already has (serial reader, ROS
    subscriber, driver callbacks) so it yields the message types in
    fusion.messages. Keep this class thin: convert units and timestamps,
    nothing else. Calibration and fusion happen downstream so raw logs stay raw.

    Checklist:
      [ ] every message stamped with MCU capture time (convert camera clock)
      [ ] units: rad/s, m/s^2, meters, seconds
      [ ] ultrasonic 'no echo' becomes range_m=None, not 0 or max range
      [ ] encoder logged as raw cumulative ticks
    """

    def __init__(self, hardware):
        self.hw = hardware

    def __iter__(self):
        raise NotImplementedError("Wire your existing hardware interface here (Step 0).")
