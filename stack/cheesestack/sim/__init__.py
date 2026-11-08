"""Simulation layer: digital twin, replay, scenarios, closed-loop testing."""

from .closed_loop import ClosedLoopConfig, ClosedLoopRunner, PurePursuitDriver, SimResult, run_scenario
from .physics import ActorTruth, World
from .roads import Road
from .scenarios import (
    Scenario,
    curve,
    cut_in,
    hard_brake_lead,
    make_scenario,
    pedestrian_crossing,
    straight_follow,
)
from .sensors import default_sensor_suite

__all__ = [
    "ActorTruth",
    "ClosedLoopConfig",
    "ClosedLoopRunner",
    "PurePursuitDriver",
    "Road",
    "Scenario",
    "SimResult",
    "World",
    "curve",
    "cut_in",
    "default_sensor_suite",
    "hard_brake_lead",
    "make_scenario",
    "pedestrian_crossing",
    "run_scenario",
    "straight_follow",
]
