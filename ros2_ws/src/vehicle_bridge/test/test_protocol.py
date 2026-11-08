import pytest

from vehicle_bridge import protocol as p
from vehicle_bridge.time_sync import McuClock


def frame(body):
    return p.encode(body)


def test_crc_check_value():
    assert p.crc16(b'123456789') == 0x29B1


def test_encode_decode_roundtrip_and_garbage():
    d = p.LineDecoder()
    data = b'noise' + frame('ARM,1') + frame('ACK,1,OK,-').replace(b'\n', b'\r\n')
    assert d.feed(data) == [('frame', 'ARM,1'), ('frame', 'ACK,1,OK,-')]
    raw = frame('STA,1,DISARMED,0040,-,65535,0,1000,0,0')
    assert d.feed(raw[:7]) == []
    assert d.feed(raw[7:]) == [('frame', 'STA,1,DISARMED,0040,-,65535,0,1000,0,0')]


def test_bad_frames():
    d = p.LineDecoder()
    good = frame('ARM,1')
    crc = good[-5:-1]
    wrong = b'0000' if crc != b'0000' else b'1111'
    assert d.feed(good[:-5] + wrong + b'\n') == [('bad', 'crc mismatch')]
    assert d.feed(b'$ARM,1\n') == [('bad', 'missing *')]
    lettered = next(f for f in (frame(f'ARM,{i}') for i in range(100))
                    if any(c in b'ABCDEF' for c in f[-5:-1]))
    assert d.feed(lettered[:-5] + lettered[-5:-1].lower() + b'\n') == [('bad', 'bad crc field')]
    out = d.feed(b'$' + b'A' * 200 + b'*0000\n' + good)
    assert out == [('bad', 'overlength'), ('frame', 'ARM,1')]


@pytest.mark.parametrize('body', ['A' * 121, 'a$b', 'a*b', 'tab\there'])
def test_encode_rejects_bad_body(body):
    with pytest.raises(p.ProtocolError):
        p.encode(body)


def test_parse_all_types():
    s = p.parse('STA,123,ARMED,0005,7,12,250,0,-300,3')
    assert (s.state, s.faults, s.last_cmd_seq, s.cmd_age_ms, s.throttle_pm, s.steer_pm,
            s.bad_frames) == ('ARMED', 5, 7, 12, 250, -300, 3)
    assert p.fault_names(s.faults) == ['ESTOP_INPUT', 'CMD_TIMEOUT']
    assert p.parse('STA,1,DISARMED,0000,-,65535,0,1000,0,0').last_cmd_seq is None
    u = p.parse('US,99,2,-,NO_ECHO')
    assert (u.id, u.dist_mm, u.status, u.valid) == (2, None, 'NO_ECHO', False)
    assert p.parse('US,99,0,1500,OK').valid
    i = p.parse('IMU,5,MPU6050,1,-2,9810,100,-200,300,2512,OK')
    assert i.accel_mm_s2 == (1, -2, 9810) and i.gyro_urad_s == (100, -200, 300)
    assert i.temp_cc == 2512
    itg = p.parse('IMU,5,ITG3200,-,-,-,1,2,3,3000,OK')
    assert itg.accel_mm_s2 is None and itg.gyro_urad_s == (1, 2, 3)
    fail = p.parse('IMU,5,MPU6050,-,-,-,-,-,-,-,READ_FAIL')
    assert fail.gyro_urad_s is None and fail.status == 'READ_FAIL'
    assert p.parse('ACK,4,ERR,NOT_ARMED') == p.Ack(4, False, 'NOT_ARMED')
    assert p.parse('ACK,4,OK,-') == p.Ack(4, True, None)
    b = p.parse('BOOT,0.1.0,OUTPUTS_DISABLED,HWWDT_UNVERIFIED')
    assert (b.fw_version, b.outputs_enabled, b.hw_watchdog_verified) == ('0.1.0', False, False)
    assert p.parse('I2C,68,MPU6050') == p.I2cIdent(0x68, 'MPU6050')
    assert p.parse('I2C,END,2') == p.I2cEnd(2)
    assert p.parse('LOG,hello world') == p.Log('hello world')


@pytest.mark.parametrize('body', [
    'STA,1.5,ARMED,0000,-,0,0,0,0,0',   # float
    'STA,1,ARMED,00,-,0,0,0,0,0',       # short mask
    'STA,1,FLYING,0000,-,0,0,0,0,0',    # bad state
    'STA,1,ARMED,0000,-,0,0,0,0',       # field count
    'US,1,4,100,OK',                    # bad id
    'US,1,0,1e3,OK',                    # not a strict int
    'US,1,0,100,MAYBE',
    'IMU,1,BNO055,0,0,0,0,0,0,0,OK',
    'ACK,1,MAYBE,-',
    'NOPE,1',
])
def test_parse_rejects(body):
    with pytest.raises(p.ProtocolError):
        p.parse(body)


def test_cmd_body_clamps_and_rejects_nonfinite():
    assert p.cmd_body(3, 0.5, 0.0, -0.25) == 'CMD,3,500,0,-250'
    assert p.cmd_body(4, 2.0, -1.0, 5.0) == 'CMD,4,1000,0,1000'
    for bad in (float('nan'), float('inf')):
        with pytest.raises(p.ProtocolError):
            p.cmd_body(1, bad, 0, 0)
    assert p.simple_body('ARM', 9) == 'ARM,9'
    with pytest.raises(p.ProtocolError):
        p.simple_body('FLY', 1)


def test_clock_unwrap():
    c = McuClock()
    assert c.unwrap(2**32 - 10) == 2**32 - 10
    assert c.unwrap(5) == 2**32 + 5             # wrapped
    assert c.unwrap(2**32 - 20) == 2**32 - 20   # late frame from before the wrap
    assert c.unwrap(3) == 2**32 + 3             # slightly older, same epoch
    assert c.unwrap(10) == 2**32 + 10


def test_clock_offset_uses_minimum_delay():
    c = McuClock()
    # host = mcu + 100 s + transport delay; smallest delay is 1 ms
    out = [c.observe(t, t * 1e-6 + 100 + d) for t, d in [(0, 0.005), (1000, 0.001), (2000, 0.003)]]
    assert out[-1] == pytest.approx(0.002 + 100.001)
