import math

import pytest

from cheesestack.core import Header, Pose2D
from cheesestack.control import (
    CurvatureController, CurvatureControlConfig, LagCompensator, LagCompensatorConfig, SteerSolution, VehicleParams,
    curvature_to_steer, make_control_command, predict_pose, theoretical_gain, understeer_feedforward,
)


def test_curvature_to_steer_matches_bicycle_model_swift():
    p = VehicleParams()
    k = 0.02
    sol = curvature_to_steer(k, p)
    assert sol.road_wheel_deg == pytest.approx(math.degrees(math.atan(k * p.wheelbase_m)))
    assert sol.steer_wheel_deg == pytest.approx(math.atan(k * p.wheelbase_m) * p.steering_ratio * 180 / math.pi)
    assert sol.steer_wheel_deg == pytest.approx(theoretical_gain(p) * k, rel=3e-3)
    big = curvature_to_steer(1.0, p)
    assert big.steer_wheel_deg == p.max_steer_wheel_deg and big.saturated


def test_curvature_rate_limited():
    cc = CurvatureController(VehicleParams(), CurvatureControlConfig(max_curvature_rate=0.1))
    s = cc.update(0.1, 0.1)
    assert s.curvature == pytest.approx(0.01) and s.saturated


def test_command_sign_convention():
    sol = SteerSolution(30.0, 0.01, 2.0)  # stack: left turn
    app = make_control_command(Header(1.0), sol, 0.3, 0.0, 5.0)
    assert app.steer_wheel_deg == -30.0 and app.curvature == -0.01  # app: +right
    stack = make_control_command(Header(1.0), sol, 1.5, -1.0, 5.0, app_sign_convention=False)
    assert stack.steer_wheel_deg == 30.0 and stack.throttle == 1.0 and stack.brake == 0.0


def test_understeer_feedforward():
    p = VehicleParams()
    assert p.understeer_gradient > 0  # default sedan understeers
    assert understeer_feedforward(0.02, 0.0, p) == 0.0
    ff = understeer_feedforward(0.02, 20.0, p)
    assert ff == pytest.approx(p.understeer_gradient * 400 * 0.02)


def test_lag_compensator_improves_tracking_of_first_order_actuator():
    tau, dt = 0.15, 0.01
    comp = LagCompensator(LagCompensatorConfig(tau_s=tau, deriv_filter_s=0.02))
    y_raw = y_comp = 0.0
    err_raw = err_comp = 0.0
    for i in range(400):
        r = math.sin(2 * math.pi * 0.5 * i * dt)
        u = comp.update(r, dt)
        y_raw += (r - y_raw) * dt / tau
        y_comp += (u - y_comp) * dt / tau
        if i > 100:
            err_raw += abs(r - y_raw)
            err_comp += abs(r - y_comp)
    assert err_comp < 0.3 * err_raw


def test_predict_pose_arc():
    p = predict_pose(Pose2D(0, 0, 0), 10.0, 0.1, math.pi / 2 * 1.0, steps=200)  # quarter circle R=10
    assert p.x == pytest.approx(10.0, abs=0.05) and p.y == pytest.approx(10.0, abs=0.05)
