"""CAN abstraction: frames, acceptance filters, a bus interface, an in-memory
virtual bus, and a SocketCAN adapter.

`CanFrame` covers classic CAN (11-bit / 29-bit IDs, 0..8 bytes, RTR, error
frames) and CAN-FD (lengths up to 64, BRS/ESI flags). Buses are polled
(`recv()` is non-blocking for the virtual bus) so everything stays
deterministic under a `SimClock`.

`VirtualCanNetwork` is a shared "wire": every `VirtualBus` attached to the same
network sees every frame sent by the others (and optionally its own), filtered
by the receiving bus's acceptance filters. Taps (`add_tap`) see every frame on
the wire unfiltered, which is how monitors (timing, bus health, loggers) attach.
"""

from __future__ import annotations

import abc
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Iterator, Optional, Sequence

from cheesestack.core import Clock, SimClock

STD_ID_MAX = 0x7FF
EXT_ID_MAX = 0x1FFFFFFF
CAN_MAX_DLEN = 8
CANFD_MAX_DLEN = 64

# DLC code -> payload length (CAN-FD); codes 0..8 map 1:1.
_FD_DLC_TO_LEN = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64]
FD_VALID_LENGTHS = frozenset(_FD_DLC_TO_LEN)


def dlc_to_len(dlc: int, fd: bool = False) -> int:
    """Map a 4-bit DLC code to a payload length (classic: 9..15 -> 8)."""
    if not 0 <= dlc <= 15:
        raise ValueError(f"DLC out of range: {dlc}")
    return _FD_DLC_TO_LEN[dlc] if fd else min(dlc, 8)


def len_to_dlc(length: int) -> int:
    """Smallest DLC code whose payload length is >= `length`."""
    for code, n in enumerate(_FD_DLC_TO_LEN):
        if n >= length:
            return code
    raise ValueError(f"payload too long for CAN-FD: {length}")


def fd_padded_length(length: int) -> int:
    """Round a payload length up to the next valid CAN-FD length."""
    return dlc_to_len(len_to_dlc(length), fd=True)


@dataclass(frozen=True)
class CanFrame:
    """A single CAN / CAN-FD frame.

    `timestamp` is seconds on the stack clock (0.0 means "unstamped"; buses stamp
    on send). `data` is coerced to immutable `bytes`.
    """

    arbitration_id: int
    data: bytes = b""
    timestamp: float = 0.0
    is_extended: bool = False
    is_fd: bool = False
    is_remote: bool = False
    is_error: bool = False
    bitrate_switch: bool = False
    error_state_indicator: bool = False
    channel: str = ""
    remote_dlc: int = 0  # requested length for RTR frames

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", bytes(self.data))
        max_id = EXT_ID_MAX if self.is_extended else STD_ID_MAX
        if not 0 <= self.arbitration_id <= max_id:
            raise ValueError(
                f"arbitration id 0x{self.arbitration_id:X} out of range for "
                f"{'extended' if self.is_extended else 'standard'} frame"
            )
        n = len(self.data)
        if self.is_fd:
            if self.is_remote:
                raise ValueError("CAN-FD has no remote frames")
            if n not in FD_VALID_LENGTHS:
                raise ValueError(f"invalid CAN-FD length {n}; pad to {fd_padded_length(n)}")
        else:
            if n > CAN_MAX_DLEN:
                raise ValueError(f"classic CAN payload too long: {n}")
            if self.bitrate_switch or self.error_state_indicator:
                raise ValueError("BRS/ESI only valid on CAN-FD frames")
        if self.is_remote and n:
            raise ValueError("remote frames carry no data")

    @property
    def dlc(self) -> int:
        if self.is_remote:
            return self.remote_dlc
        return len_to_dlc(len(self.data))

    def __len__(self) -> int:
        return len(self.data)

    def with_timestamp(self, t: float) -> "CanFrame":
        return replace(self, timestamp=t)

    def __str__(self) -> str:
        idw = 8 if self.is_extended else 3
        return f"{self.timestamp:.6f} {self.channel or '-'} {self.arbitration_id:0{idw}X} [{len(self.data)}] {self.data.hex(' ').upper()}"


@dataclass(frozen=True)
class CanFilter:
    """Acceptance filter, SocketCAN semantics: accept if (id & mask) == (can_id & mask).

    `extended=None` matches both frame formats.
    """

    can_id: int
    can_mask: int = EXT_ID_MAX
    extended: Optional[bool] = None

    def matches(self, frame: CanFrame) -> bool:
        if self.extended is not None and frame.is_extended != self.extended:
            return False
        return (frame.arbitration_id & self.can_mask) == (self.can_id & self.can_mask)


@dataclass
class BusStatistics:
    tx_frames: int = 0
    rx_frames: int = 0
    rx_filtered: int = 0
    rx_overruns: int = 0
    tx_errors: int = 0


class Bus(abc.ABC):
    """Minimal CAN bus interface (python-can-like, but polled)."""

    channel: str = ""

    def __init__(self) -> None:
        self.filters: list[CanFilter] = []
        self.stats = BusStatistics()

    @abc.abstractmethod
    def send(self, frame: CanFrame) -> None: ...

    @abc.abstractmethod
    def recv(self, timeout: Optional[float] = 0.0) -> Optional[CanFrame]: ...

    def set_filters(self, filters: Optional[Sequence[CanFilter]]) -> None:
        """Replace acceptance filters. Empty/None accepts everything."""
        self.filters = list(filters or [])

    def accepts(self, frame: CanFrame) -> bool:
        return not self.filters or any(f.matches(frame) for f in self.filters)

    def recv_all(self) -> list[CanFrame]:
        out = []
        while (f := self.recv(0.0)) is not None:
            out.append(f)
        return out

    def shutdown(self) -> None:
        pass

    def __iter__(self) -> Iterator[CanFrame]:
        return iter(self.recv_all())

    def __enter__(self) -> "Bus":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()


Tap = Callable[[CanFrame], None]


class VirtualCanNetwork:
    """Shared in-memory wire connecting `VirtualBus` endpoints.

    Delivery is synchronous and in send order (no arbitration model).
    # TODO(comms): model arbitration/priority and transmission latency from bitrate.
    """

    def __init__(self, channel: str = "vcan0", clock: Optional[Clock] = None) -> None:
        self.channel = channel
        self.clock = clock or SimClock()
        self._buses: list["VirtualBus"] = []
        self._taps: list[Tap] = []
        self.frames_on_wire = 0

    def attach(self, bus: "VirtualBus") -> None:
        if bus not in self._buses:
            self._buses.append(bus)

    def detach(self, bus: "VirtualBus") -> None:
        if bus in self._buses:
            self._buses.remove(bus)

    def add_tap(self, tap: Tap) -> Callable[[], None]:
        """Observe every frame on the wire. Returns an unsubscribe function."""
        self._taps.append(tap)
        return lambda: self._taps.remove(tap) if tap in self._taps else None

    def transmit(self, frame: CanFrame, sender: Optional["VirtualBus"] = None) -> CanFrame:
        stamped = replace(
            frame,
            timestamp=frame.timestamp or self.clock.now(),
            channel=frame.channel or self.channel,
        )
        self.frames_on_wire += 1
        for tap in list(self._taps):
            tap(stamped)
        for bus in list(self._buses):
            if bus is sender and not bus.config.receive_own_messages:
                continue
            bus._deliver(stamped)
        return stamped


@dataclass
class VirtualBusConfig:
    rx_queue_size: int = 4096
    receive_own_messages: bool = False
    fd: bool = True  # whether this endpoint accepts CAN-FD frames


class VirtualBus(Bus):
    """In-memory CAN endpoint. Create several on one `VirtualCanNetwork`."""

    def __init__(
        self,
        network: Optional[VirtualCanNetwork] = None,
        config: Optional[VirtualBusConfig] = None,
        filters: Optional[Iterable[CanFilter]] = None,
    ) -> None:
        super().__init__()
        self.config = config or VirtualBusConfig()
        self.network = network or VirtualCanNetwork()
        self.channel = self.network.channel
        self._rx: deque[CanFrame] = deque()
        self.set_filters(list(filters or []))
        self._open = True
        self.network.attach(self)

    def send(self, frame: CanFrame) -> None:
        if not self._open:
            raise RuntimeError("bus is shut down")
        if frame.is_fd and not self.config.fd:
            self.stats.tx_errors += 1
            raise ValueError("this endpoint is not CAN-FD capable")
        self.network.transmit(frame, sender=self)
        self.stats.tx_frames += 1

    def _deliver(self, frame: CanFrame) -> None:
        if (frame.is_fd and not self.config.fd) or not self.accepts(frame):
            self.stats.rx_filtered += 1
            return
        if len(self._rx) >= self.config.rx_queue_size:
            self._rx.popleft()  # drop oldest, like a kernel socket buffer
            self.stats.rx_overruns += 1
        self._rx.append(frame)

    def recv(self, timeout: Optional[float] = 0.0) -> Optional[CanFrame]:
        # Virtual bus is polled; `timeout` is accepted for interface compatibility.
        if self._rx:
            self.stats.rx_frames += 1
            return self._rx.popleft()
        return None

    @property
    def pending(self) -> int:
        return len(self._rx)

    def shutdown(self) -> None:
        self._open = False
        self.network.detach(self)


# --- SocketCAN -------------------------------------------------------------

CAN_EFF_FLAG = 0x80000000
CAN_RTR_FLAG = 0x40000000
CAN_ERR_FLAG = 0x20000000
CANFD_BRS = 0x01
CANFD_ESI = 0x02
_CAN_FRAME_FMT = "=IB3x8s"  # struct can_frame (16 bytes)
_CANFD_FRAME_FMT = "=IBB2x64s"  # struct canfd_frame (72 bytes)


def pack_socketcan(frame: CanFrame) -> bytes:
    """Serialize to Linux `struct can_frame` / `struct canfd_frame`."""
    import struct

    can_id = frame.arbitration_id
    if frame.is_extended:
        can_id |= CAN_EFF_FLAG
    if frame.is_remote:
        can_id |= CAN_RTR_FLAG
    if frame.is_error:
        can_id |= CAN_ERR_FLAG
    if frame.is_fd:
        flags = (CANFD_BRS if frame.bitrate_switch else 0) | (CANFD_ESI if frame.error_state_indicator else 0)
        return struct.pack(_CANFD_FRAME_FMT, can_id, len(frame.data), flags, frame.data.ljust(64, b"\0"))
    dlc = frame.remote_dlc if frame.is_remote else len(frame.data)
    return struct.pack(_CAN_FRAME_FMT, can_id, dlc, frame.data.ljust(8, b"\0"))


def unpack_socketcan(raw: bytes, timestamp: float = 0.0, channel: str = "") -> CanFrame:
    """Parse a Linux `struct can_frame` (16 B) or `struct canfd_frame` (72 B)."""
    import struct

    if len(raw) == struct.calcsize(_CANFD_FRAME_FMT):
        can_id, n, flags, data = struct.unpack(_CANFD_FRAME_FMT, raw)
        fd = True
    elif len(raw) == struct.calcsize(_CAN_FRAME_FMT):
        can_id, n, data = struct.unpack(_CAN_FRAME_FMT, raw)
        flags, fd = 0, False
    else:
        raise ValueError(f"unexpected SocketCAN frame size {len(raw)}")
    ext = bool(can_id & CAN_EFF_FLAG)
    rtr = bool(can_id & CAN_RTR_FLAG)
    err = bool(can_id & CAN_ERR_FLAG)
    arb = can_id & (EXT_ID_MAX if ext else STD_ID_MAX) if not err else can_id & EXT_ID_MAX
    return CanFrame(
        arbitration_id=arb,
        data=b"" if rtr else data[:n],
        timestamp=timestamp,
        is_extended=ext or (err and arb > STD_ID_MAX),
        is_fd=fd,
        is_remote=rtr,
        is_error=err,
        bitrate_switch=bool(flags & CANFD_BRS),
        error_state_indicator=bool(flags & CANFD_ESI),
        channel=channel,
        remote_dlc=n if rtr else 0,
    )


@dataclass
class SocketCanConfig:
    channel: str = "can0"
    fd: bool = False
    receive_own_messages: bool = False


class SocketCanBus(Bus):
    """Linux SocketCAN raw-socket adapter. `socket` is imported lazily so the
    module imports on macOS/iOS-side tooling and in CI.

    Timestamps come from the supplied clock at receive time.
    # TODO(comms): use SO_TIMESTAMPNS kernel timestamps; kernel-side filters via
    #   CAN_RAW_FILTER; error-frame mask CAN_RAW_ERR_FILTER; bitrate config via netlink.
    """

    def __init__(self, config: Optional[SocketCanConfig] = None, clock: Optional[Clock] = None) -> None:
        super().__init__()
        import socket  # lazy: only needed on real hardware

        self.config = config or SocketCanConfig()
        self.channel = self.config.channel
        self.clock = clock
        if not hasattr(socket, "AF_CAN"):
            raise OSError("SocketCAN is not available on this platform")
        self._sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        if self.config.fd:
            self._sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FD_FRAMES, 1)
        if self.config.receive_own_messages:
            self._sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_RECV_OWN_MSGS, 1)
        self._sock.bind((self.channel,))

    def send(self, frame: CanFrame) -> None:
        self._sock.send(pack_socketcan(frame))
        self.stats.tx_frames += 1

    def recv(self, timeout: Optional[float] = 0.0) -> Optional[CanFrame]:
        import select

        while True:
            ready, _, _ = select.select([self._sock], [], [], timeout)
            if not ready:
                return None
            raw = self._sock.recv(72)
            t = self.clock.now() if self.clock else 0.0
            frame = unpack_socketcan(raw, timestamp=t, channel=self.channel)
            if self.accepts(frame):
                self.stats.rx_frames += 1
                return frame
            self.stats.rx_filtered += 1

    def shutdown(self) -> None:
        self._sock.close()
