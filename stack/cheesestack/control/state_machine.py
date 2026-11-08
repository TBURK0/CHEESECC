"""Controller mode state machine with driver-override detection.

Modes: OFF -> STANDBY -> ENGAGED <-> OVERRIDE, any -> FAULT.

    OFF       system disabled; outputs nothing
    STANDBY   armed, ready to engage once preconditions hold
    ENGAGED   controller commands are sent to actuators
    OVERRIDE  driver input detected; controller yields (outputs zeroed / passthrough).
              Returns to ENGAGED after `override_release_s` of no driver input if
              `resume_after_override`, else drops to STANDBY.
    FAULT     latched until `clear_fault()` with no active faults, then STANDBY.

Override detection (any of):
    |driver steering torque| > torque threshold (if a torque sensor is present)
    |measured handwheel - commanded handwheel| > steer threshold for > debounce
    brake pedal pressed (brake disengages -> STANDBY, like ACC)
    throttle pedal pressed above threshold (override, longitudinal only)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

from cheesestack.core import Fault, Severity


class ControllerMode(enum.Enum):
    OFF = "off"
    STANDBY = "standby"
    ENGAGED = "engaged"
    OVERRIDE = "override"
    FAULT = "fault"


@dataclass
class DriverInputs:
    steer_torque_nm: Optional[float] = None
    measured_steer_wheel_deg: Optional[float] = None
    commanded_steer_wheel_deg: Optional[float] = None
    brake_pressed: bool = False
    throttle_pedal: float = 0.0  # [0, 1]


@dataclass
class StateMachineConfig:
    override_torque_nm: float = 2.5
    override_steer_error_deg: float = 25.0
    override_debounce_s: float = 0.2
    override_release_s: float = 1.0
    throttle_override: float = 0.15
    resume_after_override: bool = True
    min_engage_speed_mps: float = 0.0
    fault_severity: Severity = Severity.CRITICAL  # faults at or above this latch FAULT


@dataclass
class Transition:
    stamp: float
    src: ControllerMode
    dst: ControllerMode
    reason: str


class ControllerStateMachine:
    def __init__(self, config: StateMachineConfig | None = None) -> None:
        self.config = config or StateMachineConfig()
        self.mode = ControllerMode.OFF
        self.history: list[Transition] = []
        self.active_faults: dict[str, Fault] = {}
        self._override_since: Optional[float] = None
        self._quiet_since: Optional[float] = None

    # -- transitions ---------------------------------------------------------
    def _go(self, dst: ControllerMode, t: float, reason: str) -> None:
        if dst != self.mode:
            self.history.append(Transition(t, self.mode, dst, reason))
            self.mode = dst
            self._override_since = None
            self._quiet_since = None

    def enable(self, t: float) -> None:
        if self.mode == ControllerMode.OFF:
            self._go(ControllerMode.STANDBY, t, "enable")

    def disable(self, t: float) -> None:
        self._go(ControllerMode.OFF, t, "disable")

    def request_engage(self, t: float, speed_mps: float = 0.0) -> bool:
        if self.mode == ControllerMode.STANDBY and not self.active_faults and speed_mps >= self.config.min_engage_speed_mps:
            self._go(ControllerMode.ENGAGED, t, "engage")
            return True
        return False

    def request_disengage(self, t: float) -> None:
        if self.mode in (ControllerMode.ENGAGED, ControllerMode.OVERRIDE):
            self._go(ControllerMode.STANDBY, t, "disengage")

    def report_fault(self, fault: Fault) -> None:
        if fault.severity >= self.config.fault_severity:
            self.active_faults[fault.code] = fault
            if self.mode != ControllerMode.OFF:
                self._go(ControllerMode.FAULT, fault.stamp, f"fault:{fault.code}")

    def resolve_fault(self, code: str) -> None:
        self.active_faults.pop(code, None)

    def clear_fault(self, t: float) -> bool:
        if self.mode == ControllerMode.FAULT and not self.active_faults:
            self._go(ControllerMode.STANDBY, t, "fault_cleared")
            return True
        return False

    # -- override detection --------------------------------------------------
    def driver_active(self, d: DriverInputs) -> bool:
        c = self.config
        if d.steer_torque_nm is not None and abs(d.steer_torque_nm) > c.override_torque_nm:
            return True
        if d.measured_steer_wheel_deg is not None and d.commanded_steer_wheel_deg is not None:
            if abs(d.measured_steer_wheel_deg - d.commanded_steer_wheel_deg) > c.override_steer_error_deg:
                return True
        return d.throttle_pedal > c.throttle_override

    def update(self, t: float, driver: DriverInputs | None = None) -> ControllerMode:
        """Call every tick with current driver inputs."""
        c = self.config
        d = driver or DriverInputs()
        if self.mode in (ControllerMode.ENGAGED, ControllerMode.OVERRIDE) and d.brake_pressed:
            self._go(ControllerMode.STANDBY, t, "brake_pressed")
            return self.mode
        active = self.driver_active(d)
        if self.mode == ControllerMode.ENGAGED:
            if active:
                if self._override_since is None:
                    self._override_since = t
                if t - self._override_since >= c.override_debounce_s:
                    self._go(ControllerMode.OVERRIDE, t, "driver_override")
            else:
                self._override_since = None
        elif self.mode == ControllerMode.OVERRIDE:
            if active:
                self._quiet_since = None
            else:
                if self._quiet_since is None:
                    self._quiet_since = t
                if t - self._quiet_since >= c.override_release_s:
                    dst = ControllerMode.ENGAGED if c.resume_after_override else ControllerMode.STANDBY
                    self._go(dst, t, "override_released")
        return self.mode

    @property
    def is_commanding(self) -> bool:
        return self.mode == ControllerMode.ENGAGED

    # TODO(control): engage preconditions from safety layer (localization OK, trajectory fresh),
    # hands-on-wheel monitoring, separate lateral/longitudinal override modes.
