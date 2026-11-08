from cheesestack.core import Fault, Severity
from cheesestack.control import ControllerMode, ControllerStateMachine, DriverInputs


def engaged():
    sm = ControllerStateMachine()
    sm.enable(0.0)
    assert sm.request_engage(0.1)
    return sm


def test_basic_lifecycle():
    sm = ControllerStateMachine()
    assert sm.mode == ControllerMode.OFF
    assert not sm.request_engage(0.0)
    sm = engaged()
    assert sm.is_commanding
    sm.request_disengage(1.0)
    assert sm.mode == ControllerMode.STANDBY
    sm.disable(2.0)
    assert sm.mode == ControllerMode.OFF


def test_override_debounce_and_resume():
    sm = engaged()
    torque = DriverInputs(steer_torque_nm=4.0)
    assert sm.update(1.0, torque) == ControllerMode.ENGAGED  # debounce
    assert sm.update(1.25, torque) == ControllerMode.OVERRIDE
    assert sm.update(1.5, DriverInputs()) == ControllerMode.OVERRIDE
    assert sm.update(2.6, DriverInputs()) == ControllerMode.ENGAGED


def test_override_by_steer_divergence_and_brake():
    sm = engaged()
    d = DriverInputs(measured_steer_wheel_deg=90.0, commanded_steer_wheel_deg=10.0)
    sm.update(1.0, d)
    assert sm.update(1.3, d) == ControllerMode.OVERRIDE
    assert sm.update(1.4, DriverInputs(brake_pressed=True)) == ControllerMode.STANDBY


def test_fault_latches_until_cleared():
    sm = engaged()
    sm.report_fault(Fault("X", Severity.WARNING, "t", 1.0))
    assert sm.mode == ControllerMode.ENGAGED
    sm.report_fault(Fault("ACTUATOR_STUCK", Severity.CRITICAL, "t", 1.0))
    assert sm.mode == ControllerMode.FAULT
    assert not sm.clear_fault(2.0)
    sm.resolve_fault("ACTUATOR_STUCK")
    assert sm.clear_fault(2.0) and sm.mode == ControllerMode.STANDBY
    assert [tr.reason for tr in sm.history][-1] == "fault_cleared"
