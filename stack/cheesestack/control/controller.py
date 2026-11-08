"""Top-level trajectory controller: VehicleState + Trajectory -> core.ControlCommand.

Per tick:
 1. (optional) predict pose forward by actuator delay (dynamics_comp.predict_pose)
 2. lateral law -> desired curvature  (pure_pursuit | stanley | rear_wheel | lqr)
 3. stability limit (lateral accel / yaw rate / full lock)
 4. curvature rate + magnitude limit, understeer feedforward -> handwheel deg
 5. longitudinal PID + accel feedforward -> throttle / brake; speed reference is
    the trajectory speed, or the app's curvature-based target if the trajectory
    carries no speed, capped by the stability curve-speed limit
 6. gate by ControllerStateMachine (only ENGAGED emits non-zero actuation)

`ControlNode` wires this to a MessageBus: subscribes to /localization/state and
/planning/trajectory, publishes /control/command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from cheesestack.core import ControlCommand, Header, MessageBus, Node, Trajectory, VehicleState, Clock

from .curvature_control import CurvatureControlConfig, CurvatureController, SteerSolution, make_control_command
from .dynamics_comp import predict_pose, understeer_feedforward
from .longitudinal import LongitudinalConfig, LongitudinalController, target_speed_for_curvature
from .lqr_lateral import LQRConfig, LQRLateralController
from .path_tracking import PurePursuit, PurePursuitConfig, RearWheelFeedback, Stanley, StanleyConfig
from .reference_path import ReferencePath
from .stability import StabilityConfig, StabilityLimiter
from .state_machine import ControllerMode, ControllerStateMachine, DriverInputs
from .vehicle_params import VehicleParams


@dataclass
class TrajectoryControllerConfig:
    lateral: str = "pure_pursuit"  # pure_pursuit | stanley | rear_wheel | lqr
    actuator_delay_s: float = 0.0
    use_understeer_ff: bool = True
    app_sign_convention: bool = True
    pure_pursuit: PurePursuitConfig = field(default_factory=PurePursuitConfig)
    stanley: StanleyConfig = field(default_factory=StanleyConfig)
    lqr: LQRConfig = field(default_factory=LQRConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    curvature: CurvatureControlConfig = field(default_factory=CurvatureControlConfig)
    longitudinal: LongitudinalConfig = field(default_factory=LongitudinalConfig)


class TrajectoryController:
    def __init__(self, params: VehicleParams | None = None, config: TrajectoryControllerConfig | None = None,
                 state_machine: ControllerStateMachine | None = None) -> None:
        self.params = params or VehicleParams()
        c = self.config = config or TrajectoryControllerConfig()
        c.stanley.wheelbase_m = self.params.wheelbase_m
        self.lat_pp = PurePursuit(c.pure_pursuit)
        self.lat_stanley = Stanley(c.stanley)
        self.lat_rwf = RearWheelFeedback()
        self.lat_lqr = LQRLateralController(self.params, c.lqr)
        self.stability = StabilityLimiter(self.params, c.stability)
        self.curv = CurvatureController(self.params, c.curvature)
        self.lon = LongitudinalController(self.params, c.longitudinal)
        self.sm = state_machine or ControllerStateMachine()
        self._path: Optional[ReferencePath] = None
        self.last_debug: dict = {}

    def set_trajectory(self, traj: Trajectory | ReferencePath) -> None:
        self._path = traj if isinstance(traj, ReferencePath) else ReferencePath.from_trajectory(traj)

    def reset(self) -> None:
        self.curv.reset()
        self.lon.reset()
        self.lat_lqr.reset()

    def _lateral(self, state: VehicleState, dt: float) -> float:
        c, path = self.config, self._path
        pose = state.pose
        if c.actuator_delay_s > 0:
            pose = predict_pose(pose, state.v, state.curvature, c.actuator_delay_s)
        if c.lateral == "pure_pursuit":
            return self.lat_pp.update(pose, state.v, path).curvature
        if c.lateral == "stanley":
            return self.lat_stanley.update(pose, state.v, path).curvature
        if c.lateral == "rear_wheel":
            return self.lat_rwf.update(pose, state.v, path).curvature
        if c.lateral == "lqr":
            return self.lat_lqr.update(pose, state.v, state.yaw_rate, path, dt).curvature
        raise ValueError(f"unknown lateral controller {c.lateral!r}")

    def compute(self, state: VehicleState, dt: float, driver: DriverInputs | None = None) -> ControlCommand:
        c = self.config
        t = state.header.stamp
        mode = self.sm.update(t, driver)
        header = Header(stamp=t, frame_id="base_link", source="control")
        if self._path is None or mode != ControllerMode.ENGAGED:
            self.reset()
            self.curv.reset(state.curvature)
            brake = 0.0 if mode in (ControllerMode.OVERRIDE, ControllerMode.OFF) else c.longitudinal.standstill_brake * (state.v < 0.2)
            return make_control_command(header, SteerSolution(0.0, 0.0, 0.0), 0.0, brake, 0.0,
                                        c.app_sign_convention, source=f"control:{mode.value}")
        kappa_des = self._lateral(state, dt)
        stab = self.stability.limit(kappa_des, state.v)
        ff = understeer_feedforward(stab.curvature, state.v, self.params) if c.use_understeer_ff else 0.0
        if c.lateral == "lqr":
            ff = 0.0  # LQR feedforward already includes the understeer term
        steer = self.curv.update(stab.curvature, dt, ff)

        proj = self._path.project(state.pose)
        v_ref = proj.speed if proj.speed > 0 or self._path.speed.any() else target_speed_for_curvature(self.params, proj.curvature)
        v_ref = min(v_ref, self.stability.max_speed_for_curvature(proj.curvature))
        a_ref = proj.accel
        if proj.s >= self._path.length - 0.5 and not self._path.speed.any():
            v_ref = 0.0
        lon = self.lon.update(state.v, v_ref, a_ref, dt)
        self.last_debug = {"kappa_des": kappa_des, "limited_by": stab.limited_by, "v_ref": v_ref,
                           "cross_track": proj.cross_track, "heading_error": proj.heading_error,
                           "accel_cmd": lon.accel_cmd}
        return make_control_command(header, steer, lon.throttle, lon.brake, v_ref, c.app_sign_convention)


class ControlNode(Node):
    rate_hz = 20.0
    STATE_TOPIC = "/localization/state"
    TRAJ_TOPIC = "/planning/trajectory"
    CMD_TOPIC = "/control/command"

    def __init__(self, controller: TrajectoryController | None = None, bus: MessageBus | None = None,
                 clock: Clock | None = None, name: str = "control") -> None:
        super().__init__(name, bus, clock)
        self.controller = controller or TrajectoryController()
        self.bus.subscribe(self.TRAJ_TOPIC, lambda _t, m: self.controller.set_trajectory(m))

    def step(self, t: float) -> None:
        state = self.bus.latest(self.STATE_TOPIC)
        if state is None:
            return
        cmd = self.controller.compute(state, 1.0 / self.rate_hz, self.bus.latest("/vehicle/driver_inputs"))
        self.bus.publish(self.CMD_TOPIC, cmd)
