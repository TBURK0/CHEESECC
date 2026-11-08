#!/usr/bin/env python3
"""Hardware-in-the-loop checks for the UNO Q vehicle firmware. No ROS needed.

    python3 tools/hil/hil_check.py --port /dev/serial/by-id/<adapter> link
    python3 tools/hil/hil_check.py --port sim:/tmp/fwbuild/uno_q_sim all

Subcommands: link, scan, ultrasonic, imu, estop, watchdog, all.
With a `sim:` port, steps that need a human (cover a sensor, press the e-stop) are
driven through SIM messages instead, so the whole script runs in CI without hardware.
Exit code is non-zero if any check FAILs. See docs/hardware.md -> Hardware-in-the-loop tests.
"""
import argparse
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'ros2_ws', 'src', 'vehicle_bridge'))
from vehicle_bridge import protocol as P  # noqa: E402
from vehicle_bridge.link import Link  # noqa: E402

US_NAMES = {0: 'FL', 1: 'FR', 2: 'RL', 3: 'RR'}


class Mcu:
    def __init__(self, port, baud):
        self.link = Link(port, baud)
        self.sim = port.startswith('sim:')
        if not self.link.open():
            sys.exit(f'cannot open {port}: {self.link.last_error}')
        self.dec = P.LineDecoder()
        self.seq = 100
        self.rx_bad = 0
        self.boot = None
        self.status = None

    def send_raw(self, data: bytes):
        self.link.write(data)

    def send(self, body):
        self.send_raw(P.encode(body))

    def pump(self, seconds):
        """Receive for `seconds`; return parsed frames (Boot/Status are also remembered)."""
        out, end = [], time.monotonic() + seconds
        while True:
            for kind, item in self.dec.feed(self.link.read()):
                if kind == 'bad':
                    self.rx_bad += 1
                    continue
                try:
                    msg = P.parse(item)
                except P.ProtocolError:
                    self.rx_bad += 1
                    continue
                if isinstance(msg, P.Boot):
                    self.boot = msg
                if isinstance(msg, P.Status):
                    self.status = msg
                out.append(msg)
            if time.monotonic() >= end or not self.link.connected:
                return out
            time.sleep(0.002)

    def wait_for(self, pred, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            for m in self.pump(0.01):
                if pred(m):
                    return m
        return None

    def request(self, kind, timeout=1.0):
        """Send ARM/DISARM/ESTOP/RESET/SCAN and return its Ack (or None)."""
        self.seq += 1
        seq = self.seq
        self.send(P.simple_body(kind, seq))
        return self.wait_for(lambda m: isinstance(m, P.Ack) and m.seq == seq, timeout)

    def sim_cmd(self, body):
        assert self.sim
        self.send(body)
        self.wait_for(lambda m: isinstance(m, P.Ack), 1.0)

    def cmd(self, thr, brk, steer):
        self.seq += 1
        self.send(f'CMD,{self.seq},{thr},{brk},{steer}')

    def state(self, timeout=0.5):
        self.pump(timeout)
        return self.status


class Report:
    def __init__(self):
        self.failed = 0

    def check(self, ok, name, detail=''):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  — ' + detail if detail else ''}")
        self.failed += not ok
        return ok

    def info(self, text):
        print(f'  [INFO] {text}')


def human(mcu, prompt):
    if mcu.sim:
        return
    input(f'  >>> {prompt} — press Enter when done ')


# --------------------------------------------------------------------------- checks

def check_link(mcu, r):
    print('link')
    st = mcu.wait_for(lambda m: isinstance(m, P.Status), 2.0)
    if not r.check(st is not None, 'STA received within 2 s'):
        return
    mcu.pump(2.0)
    r.check(mcu.rx_bad == 0, 'no corrupted frames from MCU in 2 s', f'{mcu.rx_bad} bad')
    faults = P.fault_names(mcu.status.faults)
    r.info(f'state={mcu.status.state} faults={faults}')
    r.info('build: outputs ' + ('DISABLED' if 'OUTPUTS_DISABLED_BUILD' in faults else 'ENABLED')
           + ', hw watchdog ' + ('UNVERIFIED' if 'HW_WATCHDOG_UNVERIFIED' in faults else 'ON'))
    if mcu.boot:
        r.info(f'firmware {mcu.boot.fw_version}')
    before = mcu.status.bad_frames
    mcu.send_raw(b'$DISARM,1*0000\n')  # deliberately wrong CRC
    mcu.pump(0.5)
    r.check(mcu.status.bad_frames == before + 1, 'MCU counts a corrupted frame',
            f'bad_frames {before} -> {mcu.status.bad_frames}')


def check_scan(mcu, r):
    print('scan')
    found = []
    mcu.seq += 1
    seq = mcu.seq
    mcu.send(P.simple_body('SCAN', seq))
    end = time.monotonic() + 3.0
    done = False
    while time.monotonic() < end and not done:
        for m in mcu.pump(0.05):
            if isinstance(m, P.I2cIdent):
                found.append(m)
            if isinstance(m, P.I2cEnd):
                done = True
    if not r.check(done, 'scan completed'):
        return
    for m in found:
        r.info(f'0x{m.addr:02X}  {m.ident}')
    idents = {m.ident for m in found}
    r.check('MPU6050' in idents, 'MPU-6050 present')
    r.check('ITG3200' in idents, 'ITG-3200 present')
    addrs = [m.addr for m in found]
    r.check(len(addrs) == len(set(addrs)), 'no duplicate addresses')


def collect(mcu, seconds, cls):
    return [m for m in mcu.pump(seconds) if isinstance(m, cls)]


def check_ultrasonic(mcu, r):
    print('ultrasonic')
    frames = collect(mcu, 3.0, P.Ultrasonic)
    for uid, name in US_NAMES.items():
        mine = [f for f in frames if f.id == uid]
        last = mine[-1].status if mine else 'none'
        rate = len(mine) / 3.0
        r.check(len(mine) > 0, f'{name} reports', f'{rate:.1f} Hz, last status {last}')
        r.check(last not in ('NO_ECHO', 'STUCK_HIGH', 'UNASSIGNED', 'none'), f'{name} healthy', last)
    order = [f.id for f in frames[:8]]
    r.info(f'firing order seen: {[US_NAMES[i] for i in order]}')
    for uid, name in US_NAMES.items():
        if mcu.sim:
            mcu.sim_cmd(f'SIM,US,{uid},100')
        human(mcu, f'cover {name} with your hand at ~10 cm, keep the others clear')
        frames = collect(mcu, 1.5, P.Ultrasonic)
        if mcu.sim:
            mcu.sim_cmd(f'SIM,US,{uid},1500')

        def near(f):
            return f.status == 'TOO_CLOSE' or (f.status == 'OK' and f.dist_mm < 300)
        mine = [f for f in frames if f.id == uid]
        others = [f for f in frames if f.id != uid and near(f)]
        r.check(bool(mine) and near(mine[-1]), f'{name} sees the obstacle',
                f'{mine[-1].status} {mine[-1].dist_mm} mm' if mine else 'no frames')
        r.check(not others, f'only {name} changed (cross-talk / wiring swap check)',
                ', '.join(f'{US_NAMES[f.id]}={f.dist_mm}' for f in others))


def check_imu(mcu, r):
    print('imu (keep the car still)')
    frames = collect(mcu, 2.0, P.Imu)
    for src in ('MPU6050', 'ITG3200'):
        mine = [f for f in frames if f.src == src]
        ok = [f for f in mine if f.status == 'OK']
        r.check(70 <= len(mine) / 2.0 <= 130, f'{src} rate ~100 Hz', f'{len(mine) / 2.0:.0f} Hz')
        if not r.check(len(ok) > 10, f'{src} readings OK', mine[-1].status if mine else 'no frames'):
            continue
        if src == 'MPU6050':
            g = statistics.mean(math.dist((0, 0, 0), f.accel_mm_s2) for f in ok) / 1000
            r.check(abs(g - 9.80665) < 0.5, 'gravity magnitude', f'{g:.2f} m/s²')
            axis = max(range(3), key=lambda i: abs(ok[-1].accel_mm_s2[i]))
            r.info(f"gravity on chip axis {'xyz'[axis]} ({'+' if ok[-1].accel_mm_s2[axis] > 0 else '-'}) "
                   '— use for axis_map, docs/hardware.md')
        for i, ax in enumerate('xyz'):
            vals = [f.gyro_urad_s[i] / 1e6 for f in ok]
            r.check(statistics.pstdev(vals) < 0.05, f'{src} gyro {ax} noise at rest',
                    f'mean(bias) {statistics.mean(vals):+.4f} rad/s, std {statistics.pstdev(vals):.4f}')


def check_estop(mcu, r):
    print('estop')
    if mcu.sim:
        mcu.sim_cmd('SIM,ESTOP,1')
    human(mcu, 'PRESS the e-stop (or unplug its wire)')
    st = mcu.wait_for(lambda m: isinstance(m, P.Status) and m.state == 'ESTOP', 1.0)
    r.check(st is not None, 'ESTOP latched')
    ack = mcu.request('RESET')
    r.check(ack is not None and not ack.ok, 'RESET refused while pressed', ack and ack.reason)
    ack = mcu.request('ARM')
    r.check(ack is not None and not ack.ok, 'ARM refused while in ESTOP', ack and ack.reason)
    if mcu.sim:
        mcu.sim_cmd('SIM,ESTOP,0')
    human(mcu, 'RELEASE the e-stop')
    mcu.pump(0.2)
    r.check(mcu.status.state == 'ESTOP', 'still latched after release (needs RESET)', mcu.status.state)
    ack = mcu.request('RESET')
    r.check(ack is not None and ack.ok, 'RESET accepted after release', ack and ack.reason)
    st = mcu.state()
    r.check(st.state == 'DISARMED', 'RESET goes to DISARMED, not ARMED', st.state)


def arm(mcu, r):
    ack = mcu.request('ARM')
    if not r.check(ack is not None and ack.ok, 'ARM accepted', ack and ack.reason):
        return False
    return True


def check_watchdog(mcu, r, args):
    print('watchdog / command expiry')
    st = mcu.state()
    outputs = 'OUTPUTS_DISABLED_BUILD' not in P.fault_names(st.faults)
    if outputs and not mcu.sim and not args.wheels_off_ground:
        r.check(False, 'refusing: firmware drives actuators; re-run with --wheels-off-ground')
        return
    try:
        # 1. command expiry
        if not arm(mcu, r):
            return
        thr = 100  # 10 % — enough to see on the PWM output, small enough for a bench test
        for _ in range(25):
            mcu.cmd(thr, 0, 0)
            mcu.pump(0.02)
        r.check(mcu.status.state == 'ARMED' and mcu.status.throttle_pm == thr, 'CMD drives output',
                f'{mcu.status.state} thr={mcu.status.throttle_pm}')
        t0 = time.monotonic()
        st = mcu.wait_for(lambda m: isinstance(m, P.Status) and m.state != 'ARMED', 1.0)
        dt = time.monotonic() - t0
        r.check(st is not None and 'CMD_TIMEOUT' in P.fault_names(st.faults), 'stale commands → safe stop',
                f'after {dt * 1000:.0f} ms' if st else 'never')
        r.check(st is not None and dt < 0.35, 'within 200 ms timeout (+ link latency)', f'{dt * 1000:.0f} ms')
        r.check(st is not None and st.throttle_pm == 0 and st.steer_pm == 0, 'safe output after stop',
                st and f'thr={st.throttle_pm} brk={st.brake_pm} str={st.steer_pm}')
        # 2. limits
        if arm(mcu, r):
            for _ in range(10):
                mcu.cmd(1000, 0, 1000)
                mcu.pump(0.02)
            r.check(mcu.status.throttle_pm <= args.max_throttle_pm, 'throttle clamped to limit',
                    f'requested 1000, driven {mcu.status.throttle_pm} (limit {args.max_throttle_pm})')
            mcu.cmd(500, 300, 0)
            mcu.pump(0.15)
            r.check(mcu.status.throttle_pm == 0, 'brake forces throttle to 0', f'thr={mcu.status.throttle_pm}')
            mcu.request('DISARM')
        # 3. corrupted frame while armed
        if arm(mcu, r):
            mcu.cmd(0, 0, 0)
            mcu.send_raw(b'$CMD,1,0,0,0*0000\n')
            st = mcu.wait_for(lambda m: isinstance(m, P.Status) and m.state != 'ARMED', 0.5)
            r.check(st is not None and 'BAD_FRAME' in P.fault_names(st.faults), 'corrupted frame → safe stop')
        # 4. out-of-range command while armed
        if arm(mcu, r):
            mcu.cmd(1001, 0, 0)
            st = mcu.wait_for(lambda m: isinstance(m, P.Status) and m.state != 'ARMED', 0.5)
            r.check(st is not None and 'INVALID_CMD' in P.fault_names(st.faults), 'out-of-range CMD → safe stop')
    finally:
        mcu.request('DISARM')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', required=True, help='serial device, or sim:<path to uno_q_sim>')
    ap.add_argument('--baud', type=int, default=460800)
    ap.add_argument('--wheels-off-ground', action='store_true',
                    help='required for `watchdog` when the firmware has actuator outputs enabled')
    ap.add_argument('--max-throttle-pm', type=int, default=250, help='MAX_THROTTLE_PM in config.h')
    ap.add_argument('test', choices=['link', 'scan', 'ultrasonic', 'imu', 'estop', 'watchdog', 'all'])
    args = ap.parse_args()

    mcu, r = Mcu(args.port, args.baud), Report()
    try:
        mcu.pump(0.5)
        tests = ['link', 'scan', 'ultrasonic', 'imu', 'estop', 'watchdog'] if args.test == 'all' else [args.test]
        for t in tests:
            if t == 'watchdog':
                check_watchdog(mcu, r, args)
            else:
                globals()[f'check_{t}'](mcu, r)
    finally:
        mcu.link.close()
    print(f"\n{'ALL CHECKS PASSED' if not r.failed else f'{r.failed} CHECK(S) FAILED'}")
    sys.exit(1 if r.failed else 0)


if __name__ == '__main__':
    main()
