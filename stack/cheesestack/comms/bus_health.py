"""Bus-health monitoring: bus load, error frames, and the ISO 11898-1 fault
confinement (error-active / error-passive / bus-off) state model.

Bus load is computed from *exact* on-wire bit counts for classic CAN frames:
the frame bitstream (SOF .. CRC-15) is built, bit-stuffing is applied, and the
fixed-form tail (CRC delimiter, ACK, EOF, IFS) is added. CAN-FD frames use the
nominal/data bitrates with a worst-case stuffing estimate.

Error frames follow Linux SocketCAN error-frame encoding (`linux/can/error.h`):
error class in the ID bits, controller details in data[1], TX/RX error
counters in data[6]/data[7].
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from cheesestack.core import Fault, Severity
from .can import CanFrame, STD_ID_MAX

# --- exact classic-CAN frame length ---------------------------------------

CRC15_POLY = 0x4599
# unstuffed tail: CRC delimiter, ACK slot, ACK delimiter, 7 EOF, 3 IFS
TAIL_BITS = 13


def _int_bits(value: int, n: int) -> list[int]:
    return [(value >> (n - 1 - i)) & 1 for i in range(n)]


def crc15(bits: list[int]) -> int:
    crc = 0
    for b in bits:
        nxt = b ^ ((crc >> 14) & 1)
        crc = (crc << 1) & 0x7FFF
        if nxt:
            crc ^= CRC15_POLY
    return crc


def frame_bitstream(frame: CanFrame) -> list[int]:
    """Unstuffed classic-CAN bits from SOF through the CRC sequence."""
    if frame.is_fd:
        raise ValueError("frame_bitstream supports classic CAN only")
    rtr = 1 if frame.is_remote else 0
    dlc = frame.remote_dlc if frame.is_remote else len(frame.data)
    bits = [0]  # SOF
    if frame.is_extended:
        aid = frame.arbitration_id
        bits += _int_bits(aid >> 18, 11) + [1, 1] + _int_bits(aid & 0x3FFFF, 18) + [rtr, 0, 0]
    else:
        bits += _int_bits(frame.arbitration_id, 11) + [rtr, 0, 0]
    bits += _int_bits(dlc, 4)
    for b in frame.data:
        bits += _int_bits(b, 8)
    return bits + _int_bits(crc15(bits), 15)


def stuff_bit_count(bits: list[int]) -> int:
    """Number of stuff bits inserted (after 5 identical bits, stuff bits count toward runs)."""
    count, run, last = 0, 0, None
    for b in bits:
        if b == last:
            run += 1
        else:
            last, run = b, 1
        if run == 5:
            count += 1
            last, run = 1 - b, 1  # the inserted complement bit starts a new run
    return count


def classic_frame_bits(frame: CanFrame, stuffing: bool = True) -> int:
    bits = frame_bitstream(frame)
    return len(bits) + (stuff_bit_count(bits) if stuffing else 0) + TAIL_BITS


def frame_duration_s(frame: CanFrame, bitrate: float = 500_000.0, data_bitrate: float = 2_000_000.0) -> float:
    """Time the frame occupies the bus (incl. IFS)."""
    if frame.is_error:
        return (6 + 8 + 3) / bitrate  # active error flag + delimiter + IFS (superposition ignored)
    if not frame.is_fd:
        return classic_frame_bits(frame) / bitrate
    # CAN-FD estimate. TODO(comms): exact FD bitstream (CRC-17/21, stuff count, fixed stuff bits).
    n = len(frame.data)
    arb = 36 if frame.is_extended else 17  # SOF..BRS
    crc_len = 17 if n <= 16 else 21
    data_phase = 1 + 4 + 8 * n + 4 + crc_len  # ESI, DLC, data, stuff count, CRC
    data_phase += (crc_len + 4) // 4  # fixed stuff bits
    dyn_stuff = (arb + 5 + 8 * n - 1) // 4  # worst-case dynamic stuffing
    arb_stuff = dyn_stuff * arb // (arb + 5 + 8 * n)
    rate_d = data_bitrate if frame.bitrate_switch else bitrate
    return (arb + arb_stuff + TAIL_BITS) / bitrate + (data_phase + dyn_stuff - arb_stuff) / rate_d


# --- bus load ---------------------------------------------------------------

@dataclass
class BusLoadConfig:
    bitrate: float = 500_000.0
    data_bitrate: float = 2_000_000.0
    window_s: float = 1.0


class BusLoadMonitor:
    """Sliding-window bus utilisation in [0, 1]."""

    def __init__(self, config: Optional[BusLoadConfig] = None) -> None:
        self.config = config or BusLoadConfig()
        self._win: deque[tuple[float, float]] = deque()
        self._busy = 0.0
        self.peak = 0.0
        self.total_frames = 0

    def observe(self, frame: CanFrame) -> None:
        d = frame_duration_s(frame, self.config.bitrate, self.config.data_bitrate)
        self._win.append((frame.timestamp, d))
        self._busy += d
        self.total_frames += 1
        self._expire(frame.timestamp)
        self.peak = max(self.peak, self.load())

    def _expire(self, now: float) -> None:
        while self._win and self._win[0][0] <= now - self.config.window_s:
            self._busy -= self._win.popleft()[1]

    def load(self, now: Optional[float] = None) -> float:
        if now is not None:
            self._expire(now)
        return max(0.0, self._busy) / self.config.window_s

    def frame_rate(self, now: Optional[float] = None) -> float:
        if now is not None:
            self._expire(now)
        return len(self._win) / self.config.window_s


# --- fault confinement --------------------------------------------------------

class ErrorState(enum.Enum):
    ERROR_ACTIVE = "error_active"
    ERROR_WARNING = "error_warning"  # a counter >= 96 (controller warning level; still active)
    ERROR_PASSIVE = "error_passive"
    BUS_OFF = "bus_off"


@dataclass
class ErrorModelConfig:
    bitrate: float = 500_000.0
    warning_limit: int = 96
    passive_limit: int = 128
    bus_off_limit: int = 256
    auto_recovery: bool = True
    restart_delay_s: float = 0.1  # controller restart-ms (SocketCAN `restart-ms`)


class ErrorCounterModel:
    """ISO 11898-1 TEC/REC fault confinement.

    TX error: TEC += 8; TX success: TEC -= 1. RX error: REC += 1 (+8 for
    errors detected as the first after an error flag, use `primary=True`);
    RX success: REC -= 1 (from >127 back to 119..127). TEC > 255 -> bus-off;
    recovery needs 128 x 11 recessive bits plus the controller restart delay.
    """

    def __init__(self, config: Optional[ErrorModelConfig] = None) -> None:
        self.config = config or ErrorModelConfig()
        self.tec = 0
        self.rec = 0
        self.bus_off_since: Optional[float] = None
        self.bus_off_count = 0

    @property
    def state(self) -> ErrorState:
        c = self.config
        if self.bus_off_since is not None:
            return ErrorState.BUS_OFF
        if self.tec >= c.passive_limit or self.rec >= c.passive_limit:
            return ErrorState.ERROR_PASSIVE
        if self.tec >= c.warning_limit or self.rec >= c.warning_limit:
            return ErrorState.ERROR_WARNING
        return ErrorState.ERROR_ACTIVE

    def tx_error(self, now: float) -> None:
        if self.bus_off_since is not None:
            return
        self.tec += 8
        if self.tec >= self.config.bus_off_limit:
            self.bus_off_since = now
            self.bus_off_count += 1

    def tx_success(self) -> None:
        if self.bus_off_since is None:
            self.tec = max(0, self.tec - 1)

    def rx_error(self, primary: bool = False) -> None:
        if self.bus_off_since is None:
            self.rec = min(self.rec + (8 if primary else 1), 255)

    def rx_success(self) -> None:
        if self.bus_off_since is not None:
            return
        if self.rec > 127:
            self.rec = 127 - 8 + 1  # ISO: set to a value between 119 and 127
        else:
            self.rec = max(0, self.rec - 1)

    @property
    def recovery_time_s(self) -> float:
        return 128 * 11 / self.config.bitrate + self.config.restart_delay_s

    def step(self, now: float) -> None:
        """Advance time: performs automatic bus-off recovery when enabled."""
        if self.bus_off_since is not None and self.config.auto_recovery:
            if now - self.bus_off_since >= self.recovery_time_s:
                self.restart()

    def restart(self) -> None:
        self.bus_off_since = None
        self.tec = self.rec = 0

    def set_counters(self, tec: int, rec: int) -> None:
        """Adopt controller-reported counters (e.g. from an error frame)."""
        self.tec, self.rec = tec, rec


# SocketCAN error classes (linux/can/error.h)
CAN_ERR_TX_TIMEOUT = 0x001
CAN_ERR_LOSTARB = 0x002
CAN_ERR_CRTL = 0x004
CAN_ERR_PROT = 0x008
CAN_ERR_TRX = 0x010
CAN_ERR_ACK = 0x020
CAN_ERR_BUSOFF = 0x040
CAN_ERR_BUSERROR = 0x080
CAN_ERR_RESTARTED = 0x100
CAN_ERR_CNT = 0x200
CAN_ERR_CRTL_RX_WARNING = 0x04
CAN_ERR_CRTL_TX_WARNING = 0x08
CAN_ERR_CRTL_RX_PASSIVE = 0x10
CAN_ERR_CRTL_TX_PASSIVE = 0x20
CAN_ERR_CRTL_ACTIVE = 0x40


@dataclass
class ErrorFrameInfo:
    classes: list[str]
    bus_off: bool = False
    restarted: bool = False
    passive: bool = False
    warning: bool = False
    tec: Optional[int] = None
    rec: Optional[int] = None


def interpret_error_frame(frame: CanFrame) -> ErrorFrameInfo:
    eid = frame.arbitration_id
    names = {CAN_ERR_TX_TIMEOUT: "tx_timeout", CAN_ERR_LOSTARB: "lost_arbitration", CAN_ERR_CRTL: "controller",
             CAN_ERR_PROT: "protocol", CAN_ERR_TRX: "transceiver", CAN_ERR_ACK: "no_ack", CAN_ERR_BUSOFF: "bus_off",
             CAN_ERR_BUSERROR: "bus_error", CAN_ERR_RESTARTED: "restarted", CAN_ERR_CNT: "counters"}
    info = ErrorFrameInfo([n for bit, n in names.items() if eid & bit])
    d = frame.data.ljust(8, b"\0")
    info.bus_off = bool(eid & CAN_ERR_BUSOFF)
    info.restarted = bool(eid & CAN_ERR_RESTARTED)
    if eid & CAN_ERR_CRTL:
        info.passive = bool(d[1] & (CAN_ERR_CRTL_RX_PASSIVE | CAN_ERR_CRTL_TX_PASSIVE))
        info.warning = bool(d[1] & (CAN_ERR_CRTL_RX_WARNING | CAN_ERR_CRTL_TX_WARNING))
    if eid & CAN_ERR_CNT or d[6] or d[7]:
        info.tec, info.rec = d[6], d[7]
    return info


def make_error_frame(t: float, classes: int, ctrl: int = 0, tec: int = 0, rec: int = 0) -> CanFrame:
    """Build a SocketCAN-style error frame (for tests and simulation)."""
    data = bytes([0, ctrl, 0, 0, 0, 0, tec & 0xFF, rec & 0xFF])
    return CanFrame(classes, data, timestamp=t, is_error=True, is_extended=classes > STD_ID_MAX)


# --- combined monitor ---------------------------------------------------------

@dataclass
class BusHealthConfig:
    load: BusLoadConfig = field(default_factory=BusLoadConfig)
    errors: ErrorModelConfig = field(default_factory=ErrorModelConfig)
    load_warning: float = 0.70
    load_degraded: float = 0.90
    error_rate_warning_hz: float = 1.0
    error_window_s: float = 1.0
    source: str = "comms.bus_health"


class BusHealthMonitor:
    """Feed every frame on the wire (use as a `VirtualCanNetwork` tap) and call
    `check(now)` each tick to get `core.Fault`s.

    Controller TX/RX outcomes can be reported via `error_model` directly; error
    frames update the model from their reported counters / bus-off flags.
    # TODO(comms): read controller state via netlink (`ip -details link`) on Linux.
    """

    def __init__(self, config: Optional[BusHealthConfig] = None) -> None:
        self.config = config or BusHealthConfig()
        self.load = BusLoadMonitor(self.config.load)
        self.error_model = ErrorCounterModel(self.config.errors)
        self._err_times: deque[float] = deque()
        self.error_frames = 0

    def observe(self, frame: CanFrame) -> None:
        self.load.observe(frame)
        if frame.is_error:
            self.error_frames += 1
            self._err_times.append(frame.timestamp)
            info = interpret_error_frame(frame)
            if info.tec is not None:
                self.error_model.set_counters(info.tec, info.rec or 0)
            if info.bus_off and self.error_model.bus_off_since is None:
                self.error_model.bus_off_since = frame.timestamp
                self.error_model.bus_off_count += 1
            if info.restarted:
                self.error_model.restart()

    __call__ = observe

    def error_rate_hz(self, now: float) -> float:
        while self._err_times and self._err_times[0] <= now - self.config.error_window_s:
            self._err_times.popleft()
        return len(self._err_times) / self.config.error_window_s

    def check(self, now: float) -> list[Fault]:
        c = self.config
        self.error_model.step(now)
        faults: list[Fault] = []
        load = self.load.load(now)
        if load >= c.load_degraded:
            faults.append(Fault("CAN_BUS_LOAD_HIGH", Severity.DEGRADED, c.source, now, f"bus load {load:.0%}", {"load": load}))
        elif load >= c.load_warning:
            faults.append(Fault("CAN_BUS_LOAD_HIGH", Severity.WARNING, c.source, now, f"bus load {load:.0%}", {"load": load}))
        rate = self.error_rate_hz(now)
        if rate >= c.error_rate_warning_hz:
            faults.append(Fault("CAN_ERROR_FRAMES", Severity.WARNING, c.source, now, f"{rate:.1f} error frames/s", {"rate_hz": rate}))
        st = self.error_model.state
        data = {"tec": self.error_model.tec, "rec": self.error_model.rec}
        if st == ErrorState.BUS_OFF:
            faults.append(Fault("CAN_BUS_OFF", Severity.CRITICAL, c.source, now, "controller is bus-off", data))
        elif st == ErrorState.ERROR_PASSIVE:
            faults.append(Fault("CAN_ERROR_PASSIVE", Severity.DEGRADED, c.source, now, "controller error-passive", data))
        elif st == ErrorState.ERROR_WARNING:
            faults.append(Fault("CAN_ERROR_WARNING", Severity.WARNING, c.source, now, "error counters >= 96", data))
        return faults
