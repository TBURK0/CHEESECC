"""Diagnostic communication: ISO-TP (ISO 15765-2) transport and UDS (ISO 14229) basics.

ISO-TP: single frame (SF), first frame (FF, incl. >4095 B escape), consecutive
frames (CF, 4-bit sequence number), flow control (FC: CTS / WAIT / OVFLW with
block size and STmin). Classic 8-byte frames; CAN-FD single frames up to 62 B.
The transport is polled: call `step()` regularly (each tick); it reads its bus
endpoint, sends flow control / consecutive frames honouring STmin on the
supplied clock, and enforces N_Bs / N_Cr timeouts.

UDS: request/response helpers, negative responses (0x7F SID NRC, incl. 0x78
responsePending), a `UdsServer` (ECU side) with DiagnosticSessionControl,
ECUReset, ReadDataByIdentifier, WriteDataByIdentifier and TesterPresent, and a
`UdsClient` (tester side).

# TODO(comms): SecurityAccess (0x27) seed/key, ReadDTCInformation (0x19),
#   RoutineControl (0x31), download/upload (0x34-0x37), functional addressing
#   (0x7DF), CAN-FD segmented transfers, extended/mixed addressing.
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

from cheesestack.core import Clock, SimClock
from .can import Bus, CanFilter, CanFrame


# ============================== ISO-TP =====================================

class PciType(enum.IntEnum):
    SINGLE = 0
    FIRST = 1
    CONSECUTIVE = 2
    FLOW_CONTROL = 3


class FlowStatus(enum.IntEnum):
    CTS = 0
    WAIT = 1
    OVERFLOW = 2


class IsoTpError(RuntimeError):
    pass


def encode_st_min(st_min_s: float) -> int:
    """STmin seconds -> ISO-TP byte (0x00-0x7F ms, 0xF1-0xF9 = 100-900 us)."""
    if st_min_s <= 0:
        return 0
    if st_min_s < 0.001:
        return 0xF0 + max(1, min(9, round(st_min_s * 1e4)))
    return min(0x7F, int(round(st_min_s * 1000)))


def decode_st_min(b: int) -> float:
    if b <= 0x7F:
        return b / 1000.0
    if 0xF1 <= b <= 0xF9:
        return (b - 0xF0) * 1e-4
    return 0.127  # reserved values: use the maximum (ISO 15765-2 9.6.5.5)


@dataclass
class IsoTpConfig:
    tx_id: int = 0x7E0
    rx_id: int = 0x7E8
    extended_ids: bool = False
    frame_len: int = 8  # 8 for classic CAN; up to 64 for CAN-FD single frames
    padding: Optional[int] = 0xCC  # None = no padding (shortest DLC)
    block_size: int = 0  # BS we advertise as receiver (0 = no further FC)
    st_min_s: float = 0.0  # STmin we advertise as receiver
    max_rx_len: int = 4095
    n_bs_timeout_s: float = 1.0  # sender waiting for FC
    n_cr_timeout_s: float = 1.0  # receiver waiting for CF
    max_wait_frames: int = 10


def _pad(data: bytes, cfg: IsoTpConfig) -> bytes:
    if cfg.padding is None:
        return data
    return data.ljust(cfg.frame_len, bytes([cfg.padding]))


def segment(payload: bytes, frame_len: int = 8) -> list[bytes]:
    """Split a payload into unpadded ISO-TP frame payloads (SF, or FF + CFs)."""
    n = len(payload)
    if n <= frame_len - 1 and n <= 7:
        return [bytes([n]) + payload]
    if frame_len > 8 and n <= frame_len - 2:
        return [bytes([0x00, n]) + payload]
    if n <= 0xFFF:
        head = bytes([0x10 | (n >> 8), n & 0xFF])
    else:
        head = bytes([0x10, 0x00]) + n.to_bytes(4, "big")
    first_take = frame_len - len(head)
    frames = [head + payload[:first_take]]
    sn, i = 1, first_take
    while i < n:
        frames.append(bytes([0x20 | sn]) + payload[i:i + frame_len - 1])
        i += frame_len - 1
        sn = (sn + 1) & 0xF
    return frames


def flow_control(status: FlowStatus, block_size: int = 0, st_min_s: float = 0.0) -> bytes:
    return bytes([0x30 | int(status), block_size & 0xFF, encode_st_min(st_min_s)])


class IsoTpTransport:
    """One ISO-TP connection (a tx_id/rx_id pair) over a dedicated bus endpoint.

    Sets an acceptance filter for `rx_id` on the bus. Not thread-safe; poll
    with `step()`. Completed incoming messages are retrieved with `recv()`.
    """

    def __init__(self, bus: Bus, config: Optional[IsoTpConfig] = None, clock: Optional[Clock] = None) -> None:
        self.bus = bus
        self.cfg = config or IsoTpConfig()
        self.clock = clock or SimClock()
        bus.set_filters([CanFilter(self.cfg.rx_id, extended=self.cfg.extended_ids)])
        self._rx_done: deque[bytes] = deque()
        self.errors: list[str] = []
        # tx state
        self._tx_frames: list[bytes] = []
        self._tx_idx = 0
        self._tx_state = "idle"  # idle | wait_fc | sending
        self._tx_block_left = 0
        self._tx_st_min = 0.0
        self._tx_next_t = 0.0
        self._tx_deadline = 0.0
        self._tx_wait_count = 0
        # rx state
        self._rx_buf = bytearray()
        self._rx_len = 0
        self._rx_sn = 0
        self._rx_block_count = 0
        self._rx_active = False
        self._rx_deadline = 0.0

    # --- public ---
    @property
    def tx_busy(self) -> bool:
        return self._tx_state != "idle"

    def send(self, payload: bytes) -> None:
        if self.tx_busy:
            raise IsoTpError("transmission already in progress")
        if not payload:
            raise IsoTpError("empty payload")
        frames = segment(bytes(payload), self.cfg.frame_len)
        if len(frames) == 1:
            self._send_raw(frames[0])
            return
        self._tx_frames, self._tx_idx = frames, 1
        self._send_raw(frames[0])
        self._tx_state = "wait_fc"
        self._tx_deadline = self.clock.now() + self.cfg.n_bs_timeout_s
        self._tx_wait_count = 0

    def recv(self) -> Optional[bytes]:
        return self._rx_done.popleft() if self._rx_done else None

    def step(self) -> None:
        now = self.clock.now()
        for frame in self.bus.recv_all():
            if frame.arbitration_id == self.cfg.rx_id and frame.data:
                self._on_frame(frame.data, now)
        self._tx_progress(now)
        if self._rx_active and now > self._rx_deadline:
            self._fail_rx("N_Cr timeout")
        if self._tx_state == "wait_fc" and now > self._tx_deadline:
            self._fail_tx("N_Bs timeout")

    # --- internals ---
    def _send_raw(self, data: bytes) -> None:
        self.bus.send(CanFrame(self.cfg.tx_id, _pad(data, self.cfg), is_extended=self.cfg.extended_ids,
                               is_fd=self.cfg.frame_len > 8 and len(_pad(data, self.cfg)) in (12, 16, 20, 24, 32, 48, 64)))

    def _fail_tx(self, why: str) -> None:
        self.errors.append(f"tx: {why}")
        self._tx_state, self._tx_frames = "idle", []

    def _fail_rx(self, why: str) -> None:
        self.errors.append(f"rx: {why}")
        self._rx_active = False
        self._rx_buf = bytearray()

    def _tx_progress(self, now: float) -> None:
        if self._tx_state != "sending":
            return
        while self._tx_idx < len(self._tx_frames) and now >= self._tx_next_t:
            self._send_raw(self._tx_frames[self._tx_idx])
            self._tx_idx += 1
            if self._tx_st_min > 0:
                self._tx_next_t = now + self._tx_st_min
            if self._tx_block_left:
                self._tx_block_left -= 1
                if self._tx_block_left == 0 and self._tx_idx < len(self._tx_frames):
                    self._tx_state = "wait_fc"
                    self._tx_deadline = now + self.cfg.n_bs_timeout_s
                    return
            if self._tx_st_min > 0:
                break  # one CF per STmin interval
        if self._tx_idx >= len(self._tx_frames):
            self._tx_state, self._tx_frames = "idle", []

    def _on_frame(self, d: bytes, now: float) -> None:
        pci = d[0] >> 4
        if pci == PciType.SINGLE:
            n = d[0] & 0x0F
            if n == 0 and len(d) > 8:  # CAN-FD escape
                n, body = d[1], d[2:]
            else:
                body = d[1:]
            if n == 0 or n > len(body):
                self.errors.append("rx: malformed SF")
                return
            if self._rx_active:
                self._fail_rx("SF interrupted segmented reception")
            self._rx_done.append(bytes(body[:n]))
        elif pci == PciType.FIRST:
            n = ((d[0] & 0x0F) << 8) | d[1]
            body = d[2:]
            if n == 0:
                n, body = int.from_bytes(d[2:6], "big"), d[6:]
            if self._rx_active:
                self._fail_rx("FF interrupted segmented reception")
            if n > self.cfg.max_rx_len:
                self._send_raw(flow_control(FlowStatus.OVERFLOW))
                self.errors.append("rx: overflow")
                return
            self._rx_buf, self._rx_len = bytearray(body[:n]), n
            self._rx_sn, self._rx_active, self._rx_block_count = 1, True, 0
            self._rx_deadline = now + self.cfg.n_cr_timeout_s
            self._send_raw(flow_control(FlowStatus.CTS, self.cfg.block_size, self.cfg.st_min_s))
        elif pci == PciType.CONSECUTIVE:
            if not self._rx_active:
                return  # stray CF: ignore (ISO 15765-2)
            sn = d[0] & 0x0F
            if sn != self._rx_sn:
                self._fail_rx(f"wrong SN {sn}, expected {self._rx_sn}")
                return
            self._rx_sn = (sn + 1) & 0xF
            need = self._rx_len - len(self._rx_buf)
            self._rx_buf += d[1:1 + need]
            self._rx_deadline = now + self.cfg.n_cr_timeout_s
            if len(self._rx_buf) >= self._rx_len:
                self._rx_done.append(bytes(self._rx_buf))
                self._rx_active, self._rx_buf = False, bytearray()
                return
            self._rx_block_count += 1
            if self.cfg.block_size and self._rx_block_count >= self.cfg.block_size:
                self._rx_block_count = 0
                self._send_raw(flow_control(FlowStatus.CTS, self.cfg.block_size, self.cfg.st_min_s))
        elif pci == PciType.FLOW_CONTROL:
            if self._tx_state != "wait_fc":
                return
            status = d[0] & 0x0F
            if status == FlowStatus.CTS:
                self._tx_block_left = d[1] if len(d) > 1 else 0
                self._tx_st_min = decode_st_min(d[2]) if len(d) > 2 else 0.0
                self._tx_state, self._tx_next_t = "sending", now
            elif status == FlowStatus.WAIT:
                self._tx_wait_count += 1
                self._tx_deadline = now + self.cfg.n_bs_timeout_s
                if self._tx_wait_count > self.cfg.max_wait_frames:
                    self._fail_tx("too many FC.WAIT")
            elif status == FlowStatus.OVERFLOW:
                self._fail_tx("receiver overflow")
            else:
                self._fail_tx(f"invalid flow status {status}")


# ================================ UDS ======================================

class UdsService(enum.IntEnum):
    DIAGNOSTIC_SESSION_CONTROL = 0x10
    ECU_RESET = 0x11
    READ_DATA_BY_IDENTIFIER = 0x22
    WRITE_DATA_BY_IDENTIFIER = 0x2E
    TESTER_PRESENT = 0x3E


class Nrc(enum.IntEnum):
    GENERAL_REJECT = 0x10
    SERVICE_NOT_SUPPORTED = 0x11
    SUBFUNCTION_NOT_SUPPORTED = 0x12
    INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT = 0x13
    RESPONSE_TOO_LONG = 0x14
    CONDITIONS_NOT_CORRECT = 0x22
    REQUEST_OUT_OF_RANGE = 0x31
    SECURITY_ACCESS_DENIED = 0x33
    RESPONSE_PENDING = 0x78
    SUBFUNCTION_NOT_SUPPORTED_IN_ACTIVE_SESSION = 0x7E
    SERVICE_NOT_SUPPORTED_IN_ACTIVE_SESSION = 0x7F


class DiagnosticSession(enum.IntEnum):
    DEFAULT = 0x01
    PROGRAMMING = 0x02
    EXTENDED = 0x03


NEGATIVE_RESPONSE_SID = 0x7F
POSITIVE_OFFSET = 0x40
SUPPRESS_POS_RSP = 0x80

# Standard identifiers (ISO 14229-1 Annex C)
DID_VIN = 0xF190
DID_ECU_SERIAL = 0xF18C
DID_SW_VERSION = 0xF195
DID_ACTIVE_SESSION = 0xF186


class NegativeResponseError(Exception):
    def __init__(self, service: int, nrc: int) -> None:
        self.service, self.nrc = service, nrc
        try:
            name = Nrc(nrc).name
        except ValueError:
            name = f"0x{nrc:02X}"
        super().__init__(f"negative response to 0x{service:02X}: {name}")


def negative_response(service: int, nrc: Nrc) -> bytes:
    return bytes([NEGATIVE_RESPONSE_SID, service & 0xFF, int(nrc)])


def check_response(request: bytes, response: bytes) -> bytes:
    """Validate a response against its request; returns the payload after the SID.

    Raises NegativeResponseError on 0x7F, IsoTpError on mismatched SID.
    """
    if not response:
        raise IsoTpError("empty response")
    if response[0] == NEGATIVE_RESPONSE_SID:
        if len(response) < 3:
            raise IsoTpError("short negative response")
        raise NegativeResponseError(response[1], response[2])
    if response[0] != (request[0] + POSITIVE_OFFSET) & 0xFF:
        raise IsoTpError(f"unexpected response SID 0x{response[0]:02X} to 0x{request[0]:02X}")
    return response[1:]


DidValue = Union[bytes, Callable[[], bytes]]


@dataclass
class DidEntry:
    value: DidValue
    sessions: frozenset = frozenset(DiagnosticSession)  # sessions allowed to read
    writable_sessions: frozenset = frozenset()  # sessions allowed to write (empty = read-only)
    length: Optional[int] = None  # fixed length for writes

    def read(self) -> bytes:
        return self.value() if callable(self.value) else bytes(self.value)


@dataclass
class UdsServerConfig:
    p2_server_s: float = 0.050
    p2_star_server_s: float = 5.0
    s3_server_s: float = 5.0  # non-default session timeout without requests
    max_response_len: int = 4095


class UdsServer:
    """ECU-side UDS dispatcher. `handle()` is pure-ish (request -> response bytes);
    `step()` pumps an ISO-TP transport and answers requests."""

    def __init__(self, transport: Optional[IsoTpTransport] = None, dids: Optional[dict[int, DidEntry]] = None,
                 config: Optional[UdsServerConfig] = None, clock: Optional[Clock] = None) -> None:
        self.transport = transport
        self.config = config or UdsServerConfig()
        self.clock = clock or (transport.clock if transport else SimClock())
        self.session = DiagnosticSession.DEFAULT
        self.dids: dict[int, DidEntry] = dict(dids or {})
        self.dids.setdefault(DID_ACTIVE_SESSION, DidEntry(lambda: bytes([int(self.session)])))
        self._last_request_t = self.clock.now()
        self.reset_count = 0
        self.on_reset: Optional[Callable[[int], None]] = None

    def step(self) -> None:
        if self.transport is None:
            return
        self.transport.step()
        now = self.clock.now()
        if self.session != DiagnosticSession.DEFAULT and now - self._last_request_t > self.config.s3_server_s:
            self.session = DiagnosticSession.DEFAULT
        while (req := self.transport.recv()) is not None:
            resp = self.handle(req)
            if resp is not None:
                self.transport.send(resp)
                self.transport.step()

    def handle(self, req: bytes) -> Optional[bytes]:
        """Return the response, or None when the positive response is suppressed."""
        self._last_request_t = self.clock.now()
        if not req:
            return None
        sid = req[0]
        handler = {
            UdsService.DIAGNOSTIC_SESSION_CONTROL: self._session_control,
            UdsService.ECU_RESET: self._ecu_reset,
            UdsService.READ_DATA_BY_IDENTIFIER: self._read_did,
            UdsService.WRITE_DATA_BY_IDENTIFIER: self._write_did,
            UdsService.TESTER_PRESENT: self._tester_present,
        }.get(sid)
        if handler is None:
            return negative_response(sid, Nrc.SERVICE_NOT_SUPPORTED)
        try:
            return handler(req)
        except _Nrc as e:
            return negative_response(sid, e.nrc)

    # --- services ---
    @staticmethod
    def _subfunction(req: bytes) -> tuple[int, bool]:
        if len(req) < 2:
            raise _Nrc(Nrc.INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT)
        return req[1] & 0x7F, bool(req[1] & SUPPRESS_POS_RSP)

    def _session_control(self, req: bytes) -> Optional[bytes]:
        sub, suppress = self._subfunction(req)
        if len(req) != 2:
            raise _Nrc(Nrc.INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT)
        try:
            session = DiagnosticSession(sub)
        except ValueError:
            raise _Nrc(Nrc.SUBFUNCTION_NOT_SUPPORTED) from None
        if session == DiagnosticSession.PROGRAMMING and self.session == DiagnosticSession.DEFAULT:
            raise _Nrc(Nrc.SUBFUNCTION_NOT_SUPPORTED_IN_ACTIVE_SESSION)  # must pass via EXTENDED
        self.session = session
        if suppress:
            return None
        p2 = int(round(self.config.p2_server_s * 1000))
        p2s = int(round(self.config.p2_star_server_s * 100))  # 10 ms resolution
        return bytes([0x50, sub]) + p2.to_bytes(2, "big") + p2s.to_bytes(2, "big")

    def _ecu_reset(self, req: bytes) -> Optional[bytes]:
        sub, suppress = self._subfunction(req)
        if sub not in (0x01, 0x02, 0x03):  # hard, keyOffOn, soft
            raise _Nrc(Nrc.SUBFUNCTION_NOT_SUPPORTED)
        self.session = DiagnosticSession.DEFAULT
        self.reset_count += 1
        if self.on_reset:
            self.on_reset(sub)
        return None if suppress else bytes([0x51, sub])

    def _read_did(self, req: bytes) -> bytes:
        body = req[1:]
        if not body or len(body) % 2:
            raise _Nrc(Nrc.INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT)
        out = bytearray([0x62])
        for i in range(0, len(body), 2):
            did = int.from_bytes(body[i:i + 2], "big")
            entry = self.dids.get(did)
            if entry is None or self.session not in entry.sessions:
                raise _Nrc(Nrc.REQUEST_OUT_OF_RANGE)
            out += did.to_bytes(2, "big") + entry.read()
        if len(out) > self.config.max_response_len:
            raise _Nrc(Nrc.RESPONSE_TOO_LONG)
        return bytes(out)

    def _write_did(self, req: bytes) -> bytes:
        if len(req) < 4:
            raise _Nrc(Nrc.INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT)
        did = int.from_bytes(req[1:3], "big")
        entry = self.dids.get(did)
        if entry is None or not entry.writable_sessions:
            raise _Nrc(Nrc.REQUEST_OUT_OF_RANGE)
        if self.session not in entry.writable_sessions:
            raise _Nrc(Nrc.CONDITIONS_NOT_CORRECT)
        data = req[3:]
        if entry.length is not None and len(data) != entry.length:
            raise _Nrc(Nrc.INCORRECT_MESSAGE_LENGTH_OR_INVALID_FORMAT)
        entry.value = bytes(data)
        return bytes([0x6E]) + req[1:3]

    def _tester_present(self, req: bytes) -> Optional[bytes]:
        sub, suppress = self._subfunction(req)
        if sub != 0x00:
            raise _Nrc(Nrc.SUBFUNCTION_NOT_SUPPORTED)
        return None if suppress else bytes([0x7E, 0x00])


class _Nrc(Exception):
    def __init__(self, nrc: Nrc) -> None:
        self.nrc = nrc


@dataclass
class SessionTiming:
    p2_s: float
    p2_star_s: float


@dataclass
class UdsClientConfig:
    max_polls: int = 1000  # poll iterations before giving up on a response
    max_pending: int = 50  # tolerated 0x78 responsePending replies


class UdsClient:
    """Tester-side UDS client over an `IsoTpTransport`.

    Synchronous: `request()` sends, then repeatedly calls `transport.step()` and
    the optional `pump` (e.g. a simulated ECU's `step`, or advancing a SimClock)
    until a final response arrives.
    # TODO(comms): wall-clock P2/P2* timeouts on real hardware, async API.
    """

    def __init__(self, transport: IsoTpTransport, pump: Optional[Callable[[], None]] = None,
                 config: Optional[UdsClientConfig] = None) -> None:
        self.transport = transport
        self.pump = pump
        self.config = config or UdsClientConfig()
        self.timing: Optional[SessionTiming] = None

    def request(self, req: bytes) -> bytes:
        """Send a raw request; return the raw positive response (raises on NRC/timeout)."""
        self.transport.send(req)
        pending = 0
        for _ in range(self.config.max_polls):
            self.transport.step()
            if self.pump:
                self.pump()
            self.transport.step()
            resp = self.transport.recv()
            if resp is None:
                continue
            if resp[0] == NEGATIVE_RESPONSE_SID and len(resp) >= 3 and resp[2] == Nrc.RESPONSE_PENDING:
                pending += 1
                if pending > self.config.max_pending:
                    raise TimeoutError("too many responsePending")
                continue
            check_response(req, resp)
            return resp
        raise TimeoutError(f"no response to 0x{req[0]:02X}")

    def diagnostic_session_control(self, session: DiagnosticSession) -> SessionTiming:
        resp = self.request(bytes([UdsService.DIAGNOSTIC_SESSION_CONTROL, int(session)]))
        if len(resp) < 6 or resp[1] != int(session):
            raise IsoTpError("malformed DiagnosticSessionControl response")
        self.timing = SessionTiming(int.from_bytes(resp[2:4], "big") / 1000.0,
                                    int.from_bytes(resp[4:6], "big") / 100.0)
        return self.timing

    def read_data_by_identifier(self, did: int) -> bytes:
        resp = self.request(bytes([UdsService.READ_DATA_BY_IDENTIFIER]) + did.to_bytes(2, "big"))
        if int.from_bytes(resp[1:3], "big") != did:
            raise IsoTpError("DID mismatch in response")
        return resp[3:]

    def write_data_by_identifier(self, did: int, data: bytes) -> None:
        self.request(bytes([UdsService.WRITE_DATA_BY_IDENTIFIER]) + did.to_bytes(2, "big") + bytes(data))

    def tester_present(self) -> None:
        self.request(bytes([UdsService.TESTER_PRESENT, 0x00]))

    def ecu_reset(self, kind: int = 0x01) -> None:
        self.request(bytes([UdsService.ECU_RESET, kind]))
