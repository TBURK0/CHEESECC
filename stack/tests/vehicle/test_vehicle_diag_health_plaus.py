import math

import pytest

from cheesestack.core import NodeStatus, Severity
from cheesestack.vehicle import (CheckType, CrossCheck, DiagnosticManager, DTCConfig, DTCStatus, HealthMonitor,
                                 PlausibilityChecker, SignalSpec, SubsystemSpec, default_vehicle_specs)


def test_dtc_lifecycle():
    dm = DiagnosticManager(DTCConfig(confirm_threshold=6, heal_threshold=-4, fail_step=2, pass_step=1, aging_cycles=2))
    tr = dm.report("X", True, 0.0)
    assert tr.new == DTCStatus.PENDING
    dm.report("X", True, 0.1)
    tr = dm.report("X", True, 0.2, {"speed": 3.0})
    assert tr.new == DTCStatus.CONFIRMED and dm.dtcs["X"].freeze_frame["speed"] == 3.0
    assert dm.active_faults(0.2)[0].severity == Severity.WARNING
    statuses = [dm.report("X", False, 0.3 + i * 0.1) for i in range(20)]
    assert dm.status("X") == DTCStatus.HEALED
    assert sum(1 for s in statuses if s) == 1
    dm.end_cycle()  # failed during this cycle: no aging
    dm.end_cycle()
    assert dm.status("X") == DTCStatus.HEALED and dm.dtcs["X"].healed_cycles == 1
    tr = dm.end_cycle()
    assert dm.status("X") == DTCStatus.NO_FAULT and tr[0].new == DTCStatus.NO_FAULT
    assert dm.dtcs["X"].occurrences == 1


def test_dtc_pending_heals_without_confirm():
    dm = DiagnosticManager(DTCConfig(confirm_threshold=10, heal_threshold=-3))
    dm.report("Y", True, 0)
    for i in range(5):
        dm.report("Y", False, i)
    assert dm.status("Y") == DTCStatus.NO_FAULT and dm.confirmed() == []


def test_plausibility_range_rate_stuck_timeout_cross():
    pc = PlausibilityChecker(default_vehicle_specs())
    pc.add_signal("x", SignalSpec(0, 10, max_rate=5, stuck_window_s=1.0, stuck_tolerance=1e-3, stuck_min_abs=0.1))
    assert all(r.passed for r in pc.check("x", 0.0, 1.0))
    bad = {r.check: r.passed for r in pc.check("x", 0.1, 5.0)}
    assert bad[CheckType.RATE] is False
    assert {r.check: r.passed for r in pc.check("x", 0.2, 50.0)}[CheckType.RANGE] is False
    assert pc.check("x", 0.3, float("nan"))[0].check == CheckType.NAN
    res = []
    for i in range(15):
        res = pc.check("x", 0.4 + 0.1 * i, 5.0)
    stuck = [r for r in res if r.check == CheckType.STUCK]
    assert stuck and not stuck[0].passed and stuck[0].code == "PLAUS_X_STUCK"

    to = {r.signal: r.passed for r in pc.check_timeouts(10.0)}
    assert to["speed"] is False

    pc.add_cross_check(CrossCheck("ay_vs_yaw", "yaw_rate", "lat_accel", lambda r: 10.0 * r, tol=0.5))
    pc.check("yaw_rate", 11.0, 0.1)
    pc.check("lat_accel", 11.0, 1.1)
    assert pc.check_cross(11.0)[0].passed
    pc.check("lat_accel", 11.05, 4.0)
    assert not pc.check_cross(11.05)[0].passed
    assert pc.check_cross(20.0) == []  # stale: no verdict


def test_health_aggregation():
    hm = HealthMonitor([SubsystemSpec("steering", weight=2.0, timeout_s=0.5),
                        SubsystemSpec("gps", critical=False, timeout_s=1.0)])
    hm.update("steering", 0.0, NodeStatus.OK)
    hm.update("gps", 0.0, Severity.INFO)
    r = hm.report(0.1)
    assert r.ok_to_drive and not r.degraded and r.score == pytest.approx(1.0)
    hm.update("gps", 0.2, NodeStatus.FAILED)  # non-critical: capped at DEGRADED
    r = hm.report(0.3)
    assert r.ok_to_drive and r.degraded and r.overall == Severity.DEGRADED and r.worst == "gps"
    r = hm.report(1.0)  # steering stale
    assert not r.ok_to_drive and r.subsystems["steering"].stale
    assert any(f.code == "HEALTH_STEERING_STALE" for f in r.faults)
