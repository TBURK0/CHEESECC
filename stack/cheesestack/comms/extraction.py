"""Signal extraction: turn a CAN frame stream into per-signal time series.

Keys are `"Message.Signal"`. Multiplexed signals only get samples when their
branch is active. `SignalExtractor.align()` resamples several series onto a
common time grid (zero-order hold or linear interpolation) for analysis,
plotting or feeding estimators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

from .can import CanFrame
from .dbc import Database
from .decoding import FrameDecoder, DecoderConfig


@dataclass
class TimeSeries:
    name: str
    t: np.ndarray
    values: np.ndarray
    unit: str = ""

    def __len__(self) -> int:
        return len(self.t)

    def at(self, t: float, method: str = "hold") -> float:
        return float(resample(self, np.array([t]), method)[0])

    def rate_hz(self) -> float:
        if len(self.t) < 2:
            return 0.0
        return (len(self.t) - 1) / float(self.t[-1] - self.t[0]) if self.t[-1] > self.t[0] else float("inf")


def resample(ts: TimeSeries, grid: np.ndarray, method: str = "hold") -> np.ndarray:
    """Sample `ts` at times `grid`. Before the first sample -> NaN.

    method: "hold" (zero-order hold, right for commands/states) or "linear".
    """
    grid = np.asarray(grid, dtype=float)
    out = np.full(grid.shape, np.nan)
    if len(ts) == 0:
        return out
    if method == "linear":
        valid = (grid >= ts.t[0]) & (grid <= ts.t[-1])
        out[valid] = np.interp(grid[valid], ts.t, ts.values.astype(float))
        out[grid > ts.t[-1]] = ts.values[-1]
        return out
    if method != "hold":
        raise ValueError(f"unknown method {method}")
    idx = np.searchsorted(ts.t, grid, side="right") - 1
    ok = idx >= 0
    out[ok] = ts.values[idx[ok]]
    return out


@dataclass
class ExtractorConfig:
    signals: Optional[Sequence[str]] = None  # "Msg.Sig" whitelist; None = all
    skip_invalid: bool = False  # drop frames that fail checksum/counter checks
    decode_choices: bool = False  # keep raw ints (numeric series) by default


class SignalExtractor:
    """Accumulates decoded samples; `series()` returns numpy time series.

    # TODO(comms): bounded ring buffers for online use; streaming to disk (parquet/HDF5).
    """

    def __init__(self, db: Database, config: Optional[ExtractorConfig] = None) -> None:
        self.db = db
        self.config = config or ExtractorConfig()
        self.decoder = FrameDecoder(db, DecoderConfig(decode_choices=self.config.decode_choices))
        self._t: dict[str, list[float]] = {}
        self._v: dict[str, list[float]] = {}
        self._units: dict[str, str] = {}
        self._wanted = set(self.config.signals) if self.config.signals else None

    def process(self, frame: CanFrame) -> None:
        d = self.decoder.decode(frame)
        if d is None or (self.config.skip_invalid and not d.valid):
            return
        msg = self.db.get_message_by_name(d.name)
        for sname, val in d.signals.items():
            key = f"{d.name}.{sname}"
            if self._wanted is not None and key not in self._wanted:
                continue
            if key not in self._t:
                self._t[key], self._v[key] = [], []
                self._units[key] = msg.signal(sname).unit
            self._t[key].append(d.timestamp)
            self._v[key].append(val)

    def process_many(self, frames: Iterable[CanFrame]) -> "SignalExtractor":
        for f in frames:
            self.process(f)
        return self

    def names(self) -> list[str]:
        return sorted(self._t)

    def series(self, name: str) -> TimeSeries:
        t = np.asarray(self._t.get(name, []), dtype=float)
        raw = self._v.get(name, [])
        try:
            v = np.asarray(raw, dtype=float)
        except (TypeError, ValueError):
            v = np.asarray(raw, dtype=object)
        order = np.argsort(t, kind="stable")
        return TimeSeries(name, t[order], v[order], self._units.get(name, ""))

    def all_series(self) -> dict[str, TimeSeries]:
        return {n: self.series(n) for n in self.names()}

    def align(self, names: Sequence[str], rate_hz: float, method: str = "hold",
              t0: Optional[float] = None, t1: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
        """Resample `names` onto a uniform grid. Returns (grid (N,), values (N, len(names)))."""
        ss = [self.series(n) for n in names]
        starts = [s.t[0] for s in ss if len(s)]
        ends = [s.t[-1] for s in ss if len(s)]
        if not starts:
            return np.zeros(0), np.zeros((0, len(names)))
        t0 = min(starts) if t0 is None else t0
        t1 = max(ends) if t1 is None else t1
        grid = t0 + np.arange(int(np.floor((t1 - t0) * rate_hz + 1e-9)) + 1) / rate_hz
        return grid, np.stack([resample(s, grid, method) for s in ss], axis=1)


def extract(db: Database, frames: Iterable[CanFrame], signals: Optional[Sequence[str]] = None) -> dict[str, TimeSeries]:
    """One-shot convenience: frames -> {"Msg.Sig": TimeSeries}."""
    return SignalExtractor(db, ExtractorConfig(signals=signals)).process_many(frames).all_series()
