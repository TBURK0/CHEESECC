"""Host side of the Jetson <-> UNO Q serial protocol. Contract: docs/serial_protocol.md."""
import math
import re
from dataclasses import dataclass
from typing import Optional

MAX_BODY = 120
MAX_LINE = 128

FAULT_BITS = {
    0x0001: 'ESTOP_INPUT',
    0x0002: 'ESTOP_LATCHED',
    0x0004: 'CMD_TIMEOUT',
    0x0008: 'BAD_FRAME',
    0x0010: 'INVALID_CMD',
    0x0020: 'ESTOP_PIN_UNASSIGNED',
    0x0040: 'OUTPUTS_DISABLED_BUILD',
    0x0080: 'HW_WATCHDOG_UNVERIFIED',
    0x0100: 'IMU_FAULT',
    0x0200: 'US_FAULT',
}
FAULT = {name: bit for bit, name in FAULT_BITS.items()}

US_IDS = {0: 'fl', 1: 'fr', 2: 'rl', 3: 'rr'}
US_STATUSES = {'OK', 'TOO_CLOSE', 'TOO_FAR', 'TIMEOUT', 'NO_ECHO', 'STUCK_HIGH', 'UNASSIGNED'}
IMU_SOURCES = {'MPU6050', 'ITG3200'}
IMU_STATUSES = {'OK', 'READ_FAIL', 'NOT_FOUND'}
STATES = {'DISARMED', 'ARMED', 'ESTOP'}


class ProtocolError(ValueError):
    pass


def fault_names(mask):
    return [name for bit, name in FAULT_BITS.items() if mask & bit]


def crc16(data: bytes) -> int:
    """CRC-16/CCITT-FALSE."""
    crc = 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


_BODY_OK = re.compile(r'^[\x20-\x7e]*$')


def encode(body: str) -> bytes:
    if len(body) > MAX_BODY or not _BODY_OK.match(body) or '$' in body or '*' in body:
        raise ProtocolError(f'invalid body: {body!r}')
    raw = body.encode('ascii')
    return b'$' + raw + b'*' + f'{crc16(raw):04X}'.encode() + b'\n'


class LineDecoder:
    """Feed raw bytes, get [('frame', body) | ('bad', reason)]."""

    def __init__(self):
        self._buf = bytearray()
        self._in_frame = False
        self._discarding = False

    def feed(self, data: bytes):
        out = []
        for b in data:
            if self._discarding:
                if b == 0x0A:
                    self._discarding = False
                continue
            if not self._in_frame:
                if b == 0x24:  # '$'
                    self._in_frame = True
                    self._buf = bytearray(b'$')
                continue
            if b == 0x0A:
                out.append(self._finish(bytes(self._buf)))
                self._in_frame = False
                continue
            if b == 0x0D:
                continue
            self._buf.append(b)
            if len(self._buf) > MAX_LINE:
                out.append(('bad', 'overlength'))
                self._in_frame = False
                self._discarding = True
        return out

    @staticmethod
    def _finish(line: bytes):
        star = line.rfind(b'*')
        if star < 0:
            return ('bad', 'missing *')
        raw, crc_txt = line[1:star], line[star + 1:]
        if len(raw) > MAX_BODY:
            return ('bad', 'overlength')
        if not re.fullmatch(rb'[0-9A-F]{4}', crc_txt):
            return ('bad', 'bad crc field')
        if int(crc_txt, 16) != crc16(raw):
            return ('bad', 'crc mismatch')
        try:
            body = raw.decode('ascii')
        except UnicodeDecodeError:
            return ('bad', 'non-ascii')
        if not _BODY_OK.match(body) or '$' in body:
            return ('bad', 'non-printable')
        return ('frame', body)


_INT = re.compile(r'^-?[0-9]+$')


def _int(s, optional=False):
    if optional and s == '-':
        return None
    if not _INT.match(s):
        raise ProtocolError(f'not an integer: {s!r}')
    return int(s)


def _enum(s, allowed):
    if s not in allowed:
        raise ProtocolError(f'unexpected value {s!r}')
    return s


@dataclass
class Boot:
    fw_version: str
    outputs_enabled: bool
    hw_watchdog_verified: bool


@dataclass
class Ack:
    seq: int
    ok: bool
    reason: Optional[str]


@dataclass
class Status:
    t_us: int
    state: str
    faults: int
    last_cmd_seq: Optional[int]
    cmd_age_ms: int
    throttle_pm: int
    brake_pm: int
    steer_pm: int
    bad_frames: int


@dataclass
class Ultrasonic:
    t_us: int
    id: int
    dist_mm: Optional[int]
    status: str

    @property
    def valid(self):
        return self.status == 'OK'


@dataclass
class Imu:
    t_us: int
    src: str
    accel_mm_s2: Optional[tuple]   # None for ITG3200 or when status != OK
    gyro_urad_s: Optional[tuple]
    temp_cc: Optional[int]
    status: str


@dataclass
class I2cIdent:
    addr: int
    ident: str


@dataclass
class I2cEnd:
    count: int


@dataclass
class Log:
    text: str


def _need(f, n):
    if len(f) != n:
        raise ProtocolError(f'{f[0]}: expected {n} fields, got {len(f)}')


def parse(body: str):
    f = body.split(',')
    t = f[0]
    if t == 'STA':
        _need(f, 10)
        if not re.fullmatch(r'[0-9A-F]{4}', f[3]):
            raise ProtocolError(f'bad fault mask {f[3]!r}')
        return Status(_int(f[1]), _enum(f[2], STATES), int(f[3], 16), _int(f[4], True),
                      _int(f[5]), _int(f[6]), _int(f[7]), _int(f[8]), _int(f[9]))
    if t == 'US':
        _need(f, 5)
        uid = _int(f[2])
        if uid not in US_IDS:
            raise ProtocolError(f'bad ultrasonic id {uid}')
        return Ultrasonic(_int(f[1]), uid, _int(f[3], True), _enum(f[4], US_STATUSES))
    if t == 'IMU':
        _need(f, 11)
        accel = [_int(x, True) for x in f[3:6]]
        gyro = [_int(x, True) for x in f[6:9]]
        return Imu(_int(f[1]), _enum(f[2], IMU_SOURCES),
                   None if None in accel else tuple(accel),
                   None if None in gyro else tuple(gyro),
                   _int(f[9], True), _enum(f[10], IMU_STATUSES))
    if t == 'ACK':
        _need(f, 4)
        ok = _enum(f[2], {'OK', 'ERR'}) == 'OK'
        return Ack(_int(f[1]), ok, None if f[3] == '-' else f[3])
    if t == 'BOOT':
        _need(f, 4)
        return Boot(f[1], _enum(f[2], {'OUTPUTS_ENABLED', 'OUTPUTS_DISABLED'}) == 'OUTPUTS_ENABLED',
                    _enum(f[3], {'HWWDT_ON', 'HWWDT_UNVERIFIED'}) == 'HWWDT_ON')
    if t == 'I2C':
        _need(f, 3)
        if f[1] == 'END':
            return I2cEnd(_int(f[2]))
        if not re.fullmatch(r'[0-9A-F]{2}', f[1]):
            raise ProtocolError(f'bad i2c addr {f[1]!r}')
        return I2cIdent(int(f[1], 16), f[2])
    if t == 'LOG':
        return Log(','.join(f[1:]))
    raise ProtocolError(f'unknown type {t!r}')


# ---- host -> MCU bodies ----

def _pm(x, lo, hi):
    if not math.isfinite(x):
        raise ProtocolError(f'non-finite value {x!r}')
    return int(round(min(max(x, lo), hi) * 1000))


def cmd_body(seq, throttle, brake, steering):
    """Normalized floats -> CMD body, clamped to the wire range. Raises ProtocolError on NaN/inf."""
    return f'CMD,{seq},{_pm(throttle, 0.0, 1.0)},{_pm(brake, 0.0, 1.0)},{_pm(steering, -1.0, 1.0)}'


def simple_body(kind, seq):
    if kind not in ('ARM', 'DISARM', 'ESTOP', 'RESET', 'SCAN'):
        raise ProtocolError(f'unknown command {kind!r}')
    return f'{kind},{seq}'
