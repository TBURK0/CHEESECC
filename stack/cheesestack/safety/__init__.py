"""Safety layer: watchdogs, validation, fault handling and fallback."""

from .assertions import AssertMode, SafetyAssertionError, SafetyAssertions
from .driver_monitoring import DriverAvailability, DriverMonitor, DriverState, TakeoverManager
from .fallback import MinimalRiskManeuver, ModeManager, SafetyMode, severity_to_mode, stopping_distance
from .faults import FaultClassifier, FaultRegistry, FaultTable, make_fault
from .health import HealthReport, HealthScorer
from .heartbeat import HeartbeatMonitor, Liveness
from .input_validation import ControlLimits, InputValidator, StateLimits, ValidationResult
from .integrity import AliveCounterMonitor, E2EChecker, E2EProtector, crc8, crc32
from .model_sanity import DetectionSanityChecker, ImageStatsOOD, TrackJumpChecker
from .recovery import RecoveryManager, RecoveryStrategy
from .redundancy import DualChannelComparator, TripleVoter, vote_2oo3
from .sensor_validation import SensorValidator
from .watchdog import Watchdog, WatchdogState

__all__ = [
    "AliveCounterMonitor",
    "AssertMode",
    "ControlLimits",
    "DetectionSanityChecker",
    "DriverAvailability",
    "DriverMonitor",
    "DriverState",
    "DualChannelComparator",
    "E2EChecker",
    "E2EProtector",
    "FaultClassifier",
    "FaultRegistry",
    "FaultTable",
    "HealthReport",
    "HealthScorer",
    "HeartbeatMonitor",
    "ImageStatsOOD",
    "InputValidator",
    "Liveness",
    "MinimalRiskManeuver",
    "ModeManager",
    "RecoveryManager",
    "RecoveryStrategy",
    "SafetyAssertionError",
    "SafetyAssertions",
    "SafetyMode",
    "SensorValidator",
    "StateLimits",
    "TakeoverManager",
    "TrackJumpChecker",
    "TripleVoter",
    "ValidationResult",
    "Watchdog",
    "WatchdogState",
    "crc32",
    "crc8",
    "make_fault",
    "severity_to_mode",
    "stopping_distance",
    "vote_2oo3",
]
