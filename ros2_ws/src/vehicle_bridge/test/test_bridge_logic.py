from vehicle_bridge.bridge_logic import BridgeLogic
from vehicle_bridge.protocol import Status


def sta(state):
    return Status(0, state, 0, None, 0, 0, 0, 0, 0)


def kinds(bodies):
    return [b.split(',')[0] for b in bodies]


def armed_logic(**kw):
    lg = BridgeLogic(allow_arm=True, **kw)
    lg.on_status(sta('ARMED'), 0.0)
    return lg


def test_no_cmd_unless_mcu_armed():
    lg = BridgeLogic(allow_arm=True)
    lg.on_input(0.1, 0, 0, 0.0)
    assert lg.tick(0.0) == []                 # no telemetry yet
    lg.on_status(sta('DISARMED'), 0.0)
    assert lg.tick(0.01) == []
    lg.on_status(sta('ARMED'), 0.02)
    lg.on_input(0.1, 0, 0, 0.02)
    assert kinds(lg.tick(0.02)) == ['CMD']


def test_cmd_rate_limited_and_clamped():
    lg = armed_logic(cmd_rate_hz=50, max_throttle=0.25)
    lg.on_input(1.0, 0.0, -3.0, 0.0)
    assert lg.tick(0.0)[0].split(',')[2:] == ['250', '0', '-1000']
    assert lg.tick(0.005) == []               # inside the 20 ms period
    lg.on_input(1.0, 0.0, 0.0, 0.02)
    lg.on_status(sta('ARMED'), 0.02)
    assert kinds(lg.tick(0.02)) == ['CMD']


def test_brake_forces_throttle_zero():
    lg = armed_logic()
    lg.on_input(0.2, 0.5, 0.0, 0.0)
    assert lg.tick(0.0)[0].split(',')[2:4] == ['0', '500']


def test_stale_input_sends_one_disarm():
    lg = armed_logic(cmd_input_timeout_s=0.2)
    lg.on_input(0.1, 0, 0, 0.0)
    assert kinds(lg.tick(0.0)) == ['CMD']
    lg.on_status(sta('ARMED'), 0.3)
    assert kinds(lg.tick(0.3)) == ['DISARM']
    lg.on_status(sta('ARMED'), 0.35)
    assert lg.tick(0.35) == []                # once per armed episode


def test_nonfinite_input_rejected_and_stops():
    lg = armed_logic()
    assert lg.on_input(float('nan'), 0, 0, 0.0)
    assert kinds(lg.tick(0.0)) == ['DISARM']


def test_telemetry_loss_disarms_and_never_resumes_without_rearm():
    lg = armed_logic(telemetry_timeout_s=0.5)
    lg.on_input(0.1, 0, 0, 0.0)
    lg.tick(0.0)
    lg.on_input(0.1, 0, 0, 0.6)
    assert kinds(lg.tick(0.6)) == ['DISARM'] and not lg.link_ok
    assert lg.tick(0.7) == []                 # one DISARM on the transition
    # link recovers and the MCU still claims ARMED: host must not resume CMD
    lg.on_status(sta('ARMED'), 0.8)
    lg.on_input(0.1, 0, 0, 0.8)
    assert kinds(lg.tick(0.8)) == ['DISARM']
    lg.on_status(sta('DISARMED'), 0.9)
    assert lg.tick(0.9) == []                 # nothing auto-sent, in particular no ARM
    ok, _, bodies = lg.request('ARM')
    assert ok and kinds(bodies) == ['ARM']
    lg.on_status(sta('ARMED'), 1.0)
    lg.on_input(0.1, 0, 0, 1.0)
    assert kinds(lg.tick(1.0)) == ['CMD']


def test_allow_arm_false_blocks_arm():
    lg = BridgeLogic()                        # default allow_arm=False
    lg.on_status(sta('DISARMED'), 0.0)
    ok, reason, bodies = lg.request('ARM')
    assert not ok and bodies == [] and 'allow_arm' in reason
    ok, _, bodies = lg.request('ESTOP')
    assert ok and kinds(bodies) == ['ESTOP']


def test_arm_needs_link():
    ok, _, bodies = BridgeLogic(allow_arm=True).request('ARM')
    assert not ok and bodies == []
