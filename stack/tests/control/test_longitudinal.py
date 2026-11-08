import pytest

from cheesestack.control import (
    LongitudinalController, VehicleParams, accel_to_pedals, simple_pedals, target_speed_for_curvature,
)
from cheesestack.control.longitudinal import resistive_force


def app_accel(p, speed, throttle, brake):
    """Port of LongitudinalModel.acceleration (the plant)."""
    m = max(400, p.mass_kg)
    t = min(1, max(0, throttle))
    watts = p.horsepower * 745.7 * t
    torque_force = p.torque_nm * 9.0 * t / max(0.2, p.tire_radius_m)
    power_force = watts / max(speed, 3)
    traction = min(m * 6.0, torque_force, power_force)
    drag = 0.5 * 1.225 * p.drag_coefficient * p.frontal_area_m2 * speed * speed
    roll = 0.013 * m * 9.81
    drive = (traction - drag - roll) / m if t > 0.02 else -(drag + roll) / m
    return drive - min(1, max(0, brake)) * 6.5


def test_target_speed_port():
    p = VehicleParams()
    assert target_speed_for_curvature(p, 0.0) == pytest.approx(min(13.5, 8 + 192 / 1475 * 40))
    assert target_speed_for_curvature(p, 0.1) < 6.0
    t, b = simple_pedals(p, 5.0, 8.5)
    assert b == 0 and 0 < t <= 1


@pytest.mark.parametrize("a,v", [(1.0, 5.0), (0.3, 12.0), (-2.0, 10.0), (-0.05, 3.0)])
def test_accel_to_pedals_inverts_plant(a, v):
    p = VehicleParams()
    thr, brk = accel_to_pedals(p, a, v)
    assert app_accel(p, v, thr, brk) == pytest.approx(a, abs=0.05)


def test_speed_tracking_closed_loop_with_app_plant():
    p = VehicleParams()
    ctl = LongitudinalController(p)
    v, dt = 0.0, 0.05
    for _ in range(600):
        out = ctl.update(v, 10.0, 0.0, dt)
        v = max(0.0, v + app_accel(p, v, out.throttle, out.brake) * dt)
    assert v == pytest.approx(10.0, abs=0.15)
    for _ in range(400):
        out = ctl.update(v, 4.0, 0.0, dt)
        v = max(0.0, v + app_accel(p, v, out.throttle, out.brake) * dt)
    assert v == pytest.approx(4.0, abs=0.15)


def test_standstill_holds_brake():
    out = LongitudinalController().update(0.0, 0.0, 0.0, 0.05)
    assert out.brake > 0 and out.throttle == 0
