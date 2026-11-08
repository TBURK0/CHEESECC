import pytest

from cheesestack.control import PID, PIDConfig, RateLimiter, RateLimiterConfig, clamp


def test_clamp_and_rate_limiter():
    assert clamp(5, 0, 1) == 1 and clamp(-5, 0, 1) == 0
    rl = RateLimiter(RateLimiterConfig(max_rate=2.0, max_fall_rate=4.0, lo=-1.0, hi=1.0))
    assert rl.update(10.0, 0.1) == pytest.approx(0.2)
    assert rl.saturated
    for _ in range(20):
        rl.update(10.0, 0.1)
    assert rl.value == 1.0
    assert rl.update(-1.0, 0.1) == pytest.approx(0.6)


def _windup_run(mode):
    pid = PID(PIDConfig(kp=0.5, ki=2.0, out_min=-1, out_max=1, anti_windup=mode))
    y, dt = 0.0, 0.01
    # plant y' = u with large setpoint (saturates), then setpoint drops
    for i in range(600):
        sp = 5.0 if i < 300 else 0.0
        u = pid.update(sp - y, dt)
        y += u * 0.2 * dt
    return pid, y


def test_antiwindup_limits_integrator():
    none, _ = _windup_run("none")
    clamp_, _ = _windup_run("clamp")
    bc, _ = _windup_run("back_calculation")
    assert abs(clamp_.integral) < abs(none.integral)
    assert abs(bc.integral) < abs(none.integral)


def test_pid_converges_unsaturated():
    pid = PID(PIDConfig(kp=2.0, ki=1.0, kd=0.05))
    y, dt = 0.0, 0.01
    for _ in range(2000):
        y += pid.update(1.0 - y, dt, measurement=y) * dt
    assert y == pytest.approx(1.0, abs=1e-3)


def test_pid_rejects_bad_dt():
    with pytest.raises(ValueError):
        PID().update(1.0, 0.0)
