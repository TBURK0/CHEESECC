"""Control layer: planned motion to vehicle requests."""

from .vehicle_params import VehicleParams
from .saturation import clamp, RateLimiter, RateLimiterConfig, PID, PIDConfig
from .reference_path import ReferencePath, PathProjection
from .path_tracking import (
    PurePursuit, PurePursuitConfig, Stanley, StanleyConfig,
    RearWheelFeedback, RearWheelFeedbackConfig, TrackingResult, pure_pursuit_curvature,
)
from .longitudinal import (
    LongitudinalController, LongitudinalConfig, LongitudinalOutput,
    accel_to_pedals, target_speed_for_curvature, simple_pedals,
)
from .lqr_lateral import LQRLateralController, LQRConfig, solve_dare, dlqr, c2d_zoh, expm, error_dynamics
from .dynamics_comp import LagCompensator, LagCompensatorConfig, understeer_feedforward, predict_pose
from .curvature_control import (
    CurvatureController, CurvatureControlConfig, SteerSolution, curvature_to_steer,
    make_control_command, theoretical_gain,
)
from .state_machine import ControllerMode, ControllerStateMachine, StateMachineConfig, DriverInputs
from .tuning import GainSchedule, StepMetrics, step_response_metrics
from .stability import StabilityLimiter, StabilityConfig
from .oscillation import OscillationDetector, OscillationConfig, OscillationReport, dominant_frequency
from .actuator_monitor import ActuatorMonitor, ActuatorMonitorConfig, ActuatorStatus, estimate_lag
from .controller import TrajectoryController, TrajectoryControllerConfig, ControlNode

import types as _types

__all__ = [n for n, v in dict(globals()).items() if not n.startswith("_") and not isinstance(v, _types.ModuleType)]
