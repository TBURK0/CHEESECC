"""RC telemetry signal contract — what the phone dashboard reads.

One JSON object per frame, newline-terminated on the wire. Units/ranges are in
DASHBOARD_SIGNALS.md. `None` means "not available yet" (sensor or fusion not wired) — the
dashboard should render those blank, NOT as 0, so a missing speedo doesn't read as "stopped".
"""
import json
from dataclasses import dataclass, asdict, field
from typing import Optional, List

SCHEMA_VERSION = 1


@dataclass
class Telemetry:
    ts: float                                  # epoch seconds (sender clock)
    schema: int = SCHEMA_VERSION

    # --- drive state (live, from VehicleStatus) ---
    state: str = "UNKNOWN"                      # DISARMED | ARMED | ESTOP | UNKNOWN
    armed: bool = False
    link_ok: bool = False
    throttle: float = 0.0                       # 0..1 driven
    brake: float = 0.0                          # 0..1 driven
    steering: float = 0.0                       # -1..1, + = left
    faults: List[str] = field(default_factory=list)
    firmware: str = ""
    outputs_enabled: bool = False
    hw_watchdog_verified: bool = False

    # --- proximity (live, meters; None if that sensor's reading is invalid) ---
    us_fl: Optional[float] = None
    us_fr: Optional[float] = None
    us_rl: Optional[float] = None
    us_rr: Optional[float] = None

    # --- motion (partial) ---
    yaw_rate: Optional[float] = None            # rad/s, live from IMU gz
    heading_deg: Optional[float] = None         # TODO(build): needs fusion/odometry

    # --- pending hardware (documented None, not faked) ---
    speed_mps: Optional[float] = None           # TODO(build): wheel/motor encoder
    battery_v: Optional[float] = None           # TODO(build): battery ADC on the UNO Q
    battery_pct: Optional[float] = None         # TODO(build)
    detections: Optional[int] = None            # TODO(build): perception model

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    def validate(self) -> "Telemetry":
        assert self.state in ("DISARMED", "ARMED", "ESTOP", "UNKNOWN"), self.state
        assert 0.0 <= self.throttle <= 1.0, self.throttle
        assert 0.0 <= self.brake <= 1.0, self.brake
        assert -1.0 <= self.steering <= 1.0, self.steering
        for d in (self.us_fl, self.us_fr, self.us_rl, self.us_rr):
            assert d is None or d >= 0.0, d
        assert self.battery_pct is None or 0.0 <= self.battery_pct <= 100.0, self.battery_pct
        return self


if __name__ == "__main__":
    # runnable self-check
    t = Telemetry(ts=1.0, state="ARMED", throttle=0.25, steering=-0.5, us_fl=1.2).validate()
    back = json.loads(t.to_json())
    assert back["state"] == "ARMED" and back["us_fl"] == 1.2 and back["us_rr"] is None
    assert back["speed_mps"] is None  # pending hardware stays null, not 0
    bad = Telemetry(ts=1.0, steering=2.0)
    try:
        bad.validate()
    except AssertionError:
        pass
    else:
        raise SystemExit("validate() failed to reject out-of-range steering")
    print("schema OK:", t.to_json())
