import pytest

from cheesestack.vehicle import VehicleParams, WheelSpeedConfig, WheelSpeedValidator, wheel_ground_speeds

P = VehicleParams()
R = P.tire_radius


def omegas(v, r=0.0, delta=0.0):
    import math
    beta = math.atan(P.lr * math.tan(delta) / P.wheelbase)
    g = wheel_ground_speeds(P, v, v * math.tan(beta), r, delta)
    return [g[w] / R for w in ("fl", "fr", "rl", "rr")]


def test_consistent_wheels_in_turn_pass():
    val = WheelSpeedValidator(P)
    v, delta = 10.0, 0.1
    r = v * __import__("math").tan(delta) / P.wheelbase
    rep = val.update(0.0, omegas(v, r, delta), yaw_rate=r, road_wheel=delta, gps_speed=10.0)
    assert rep.outliers == () and rep.valid and rep.gps_consistent
    assert rep.reference_speed == pytest.approx(10.0, rel=0.02)


def test_faulty_wheel_debounced():
    cfg = WheelSpeedConfig(debounce_ticks=3)
    val = WheelSpeedValidator(P, cfg)
    w = omegas(15.0)
    w[2] = 0.0  # rl sensor dead
    reps = [val.update(i * 0.1, w) for i in range(3)]
    assert reps[0].outliers == ("rl",) and reps[0].faulted_wheels == ()
    assert reps[-1].faulted_wheels == ("rl",)
    assert any(f.code == "WSS_RL_IMPLAUSIBLE" for f in reps[-1].faults)
    assert reps[-1].reference_speed == pytest.approx(15.0, rel=1e-3)


def test_gps_mismatch_and_scale_estimation():
    val = WheelSpeedValidator(P, WheelSpeedConfig(debounce_ticks=2))
    for i in range(3):
        rep = val.update(i * 0.1, omegas(10.0), gps_speed=20.0)
    assert rep.gps_consistent is False and any(f.code == "WSS_GPS_MISMATCH" for f in rep.faults)
    val = WheelSpeedValidator(P, WheelSpeedConfig(scale_ema_alpha=0.05))
    for i in range(500):
        val.update(i * 0.1, omegas(10.0), gps_speed=10.3)  # tires 3% larger than catalog
    assert val.scale_factor == pytest.approx(1.03, abs=0.003)
