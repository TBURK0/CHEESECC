"""End-to-end against the firmware host simulator (no hardware, no rclpy).

Runs only when UNO_Q_SIM points at the built `uno_q_sim` executable, e.g.
    UNO_Q_SIM=/tmp/fwbuild/uno_q_sim colcon test --packages-select vehicle_bridge
"""
import os
import time

import pytest

from vehicle_bridge import protocol as p
from vehicle_bridge.link import Link

SIM = os.environ.get('UNO_Q_SIM', '')
pytestmark = pytest.mark.skipif(not (SIM and os.access(SIM, os.X_OK)),
                                reason='UNO_Q_SIM not set to the uno_q_sim executable')


class Sim:
    def __init__(self):
        self.link = Link('sim:' + SIM)
        assert self.link.open(), self.link.last_error
        self.dec = p.LineDecoder()
        self.seq = 0
        self.seen = []
        self.pending = []

    def raw(self, data):
        assert self.link.write(data)

    def send(self, kind, *args):
        if kind == 'SIM':
            self.raw(p.encode(','.join(['SIM', *map(str, args)])))
            return 0  # contract: SIM is acked with seq 0
        self.seq += 1
        self.raw(p.encode(','.join([kind, str(self.seq), *map(str, args)])))
        return self.seq

    def wait(self, pred, timeout=1.0, keepalive=None):
        """Read frames until pred(msg) is true; keepalive() is called every 20 ms."""
        end = time.monotonic() + timeout
        next_ka = 0.0
        while time.monotonic() < end:
            if keepalive and time.monotonic() >= next_ka:
                keepalive()
                next_ka = time.monotonic() + 0.02
            for kind, item in self.dec.feed(self.link.read()):
                assert kind == 'frame', f'bad frame from sim: {item}'
                self.pending.append(p.parse(item))
            while self.pending:
                m = self.pending.pop(0)
                self.seen.append(m)
                if pred(m):
                    return m
            time.sleep(0.002)
        raise AssertionError(f'timed out; last frames: {self.seen[-5:]}')

    def status(self, pred=lambda s: True, **kw):
        return self.wait(lambda m: isinstance(m, p.Status) and pred(m), **kw)

    def ack(self, seq, **kw):
        return self.wait(lambda m: isinstance(m, p.Ack) and m.seq == seq, **kw)

    def arm(self):
        assert self.ack(self.send('ARM')).ok
        return self.status(lambda s: s.state == 'ARMED')


@pytest.fixture
def sim():
    s = Sim()
    yield s
    s.link.close()


def test_boot_and_disarmed(sim):
    sim.wait(lambda m: isinstance(m, p.Boot))
    assert sim.status().state == 'DISARMED'


def test_cmd_clamped_then_expires(sim):
    sim.status()
    sim.arm()
    s = sim.status(lambda s: s.throttle_pm > 0, timeout=1.0,
                   keepalive=lambda: sim.send('CMD', 1000, 0, -1000))
    assert s.throttle_pm == 250 and s.steer_pm == -1000 and s.state == 'ARMED'
    t0 = time.monotonic()
    s = sim.status(lambda s: s.state == 'DISARMED', timeout=1.0)
    assert time.monotonic() - t0 < 0.5
    assert s.faults & p.FAULT['CMD_TIMEOUT'] and s.throttle_pm == 0


def test_cmd_rejected_when_disarmed(sim):
    sim.status()
    a = sim.ack(sim.send('CMD', 100, 0, 0))
    assert not a.ok and a.reason == 'NOT_ARMED'


def test_estop_latch_and_reset(sim):
    sim.status()
    sim.ack(sim.send('SIM', 'ESTOP', 1))
    sim.status(lambda s: s.state == 'ESTOP' and s.faults & p.FAULT['ESTOP_INPUT'])
    a = sim.ack(sim.send('RESET'))
    assert not a.ok and a.reason == 'ESTOP_ACTIVE'
    a = sim.ack(sim.send('ARM'))
    assert not a.ok
    sim.ack(sim.send('SIM', 'ESTOP', 0))
    assert sim.status(lambda s: True).state == 'ESTOP'   # still latched after release
    assert sim.ack(sim.send('RESET')).ok
    sim.status(lambda s: s.state == 'DISARMED')


def test_software_estop(sim):
    sim.status()
    sim.arm()
    assert sim.ack(sim.send('ESTOP')).ok
    s = sim.status(lambda s: s.state == 'ESTOP')
    assert s.throttle_pm == 0


def test_bad_frame_while_armed_disarms(sim):
    sim.status()
    sim.arm()
    sim.raw(b'$CMD,99,100,0,0*0000\n')
    s = sim.status(lambda s: s.state == 'DISARMED')
    assert s.faults & p.FAULT['BAD_FRAME'] and s.bad_frames >= 1


def test_invalid_cmd_while_armed_disarms(sim):
    sim.status()
    sim.arm()
    sim.send('CMD', 1001, 0, 0)
    s = sim.status(lambda s: s.state == 'DISARMED')
    assert s.faults & p.FAULT['INVALID_CMD']


def test_ultrasonics(sim):
    seen = set()
    sim.wait(lambda m: isinstance(m, p.Ultrasonic) and (seen.add(m.id) or seen == {0, 1, 2, 3}),
             timeout=2.0)
    sim.ack(sim.send('SIM', 'US', 2, -1))
    sim.ack(sim.send('SIM', 'US', 1, 0))
    sim.ack(sim.send('SIM', 'US', 0, 1500))
    sim.wait(lambda m: isinstance(m, p.Ultrasonic) and m.id == 2 and m.status == 'NO_ECHO', timeout=2.0)
    sim.wait(lambda m: isinstance(m, p.Ultrasonic) and m.id == 1 and m.status == 'TIMEOUT', timeout=2.0)
    u = sim.wait(lambda m: isinstance(m, p.Ultrasonic) and m.id == 0 and m.status == 'OK', timeout=2.0)
    assert abs(u.dist_mm - 1500) <= 5
    sim.status(lambda s: s.faults & p.FAULT['US_FAULT'], timeout=1.0)


def test_imus_and_failure(sim):
    sim.wait(lambda m: isinstance(m, p.Imu) and m.src == 'MPU6050' and m.status == 'OK')
    itg = sim.wait(lambda m: isinstance(m, p.Imu) and m.src == 'ITG3200' and m.status == 'OK')
    assert itg.accel_mm_s2 is None and itg.gyro_urad_s is not None
    sim.ack(sim.send('SIM', 'IMUFAIL', 'MPU6050', 1))
    sim.wait(lambda m: isinstance(m, p.Imu) and m.src == 'MPU6050' and m.status == 'READ_FAIL')
    sim.status(lambda s: s.faults & p.FAULT['IMU_FAULT'])


def test_i2c_scan(sim):
    sim.status()
    seq = sim.send('SCAN')
    sim.wait(lambda m: isinstance(m, p.I2cEnd))
    idents = {m.ident for m in sim.seen if isinstance(m, p.I2cIdent)}
    assert {'MPU6050', 'ITG3200'} <= idents
    assert sim.ack(seq).ok
