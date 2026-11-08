from cheesestack.planning import BehaviorConfig, BehaviorPlanner, BehaviorState, WorldSummary


def run(bp, w, t0, t1, dt=0.1):
    t = t0
    d = None
    while t <= t1 + 1e-9:
        d = bp.update(w, t)
        t += dt
    return d


def test_follow_vehicle_with_hysteresis():
    bp = BehaviorPlanner()
    d = run(bp, WorldSummary(20, 25, lead_distance=40, lead_speed=18), 0, 1)
    assert d.state == BehaviorState.FOLLOW_VEHICLE
    assert d.target_speed < 25
    # just beyond entry range (60 m) but inside exit range (78 m): stays following
    d = run(bp, WorldSummary(20, 25, lead_distance=70, lead_speed=18), 1.1, 2)
    assert d.state == BehaviorState.FOLLOW_VEHICLE
    d = run(bp, WorldSummary(20, 25, lead_distance=100, lead_speed=18), 2.1, 3)
    assert d.state == BehaviorState.LANE_FOLLOW and d.target_speed == 25


def test_lane_change_sequence():
    bp = BehaviorPlanner()
    run(bp, WorldSummary(20, 25), 0, 0.5)
    d = run(bp, WorldSummary(20, 25, lane_change_request=1), 0.6, 1.0)
    assert d.state == BehaviorState.PREPARE_LANE_CHANGE and d.lane_change_direction == 1
    d = run(bp, WorldSummary(20, 25, lane_change_request=1, target_gap_acceptable=True), 1.1, 1.6)
    assert d.state == BehaviorState.LANE_CHANGE
    d = run(bp, WorldSummary(20, 25, lane_change_request=1, target_gap_acceptable=True, lane_change_progress=0.5), 1.7, 2.0)
    assert d.state == BehaviorState.LANE_CHANGE
    d = run(bp, WorldSummary(20, 25, lane_change_request=0, target_gap_acceptable=True, lane_change_progress=1.0), 2.1, 2.5)
    assert d.state == BehaviorState.LANE_FOLLOW and d.lane_change_direction == 0


def test_lane_change_abort_and_prepare_timeout():
    bp = BehaviorPlanner()
    run(bp, WorldSummary(20, 25, lane_change_request=-1), 0, 0.5)
    run(bp, WorldSummary(20, 25, lane_change_request=-1, target_gap_acceptable=True), 0.6, 1.0)
    assert bp.state == BehaviorState.LANE_CHANGE
    d = run(bp, WorldSummary(20, 25, lane_change_request=-1, target_gap_acceptable=False, lane_change_progress=0.1), 1.1, 1.5)
    assert d.state in (BehaviorState.LANE_FOLLOW, BehaviorState.PREPARE_LANE_CHANGE)
    assert any("aborted" in h[2] for h in bp.history)
    bp2 = BehaviorPlanner(BehaviorConfig(prepare_timeout_s=2.0))
    run(bp2, WorldSummary(20, 25, lane_change_request=1), 0, 0.5)
    run(bp2, WorldSummary(20, 25, lane_change_request=1), 0.6, 3.5)
    assert any(h[2] == "prepare timeout" for h in bp2.history)


def test_stop_and_release():
    bp = BehaviorPlanner()
    d = bp.update(WorldSummary(10, 15, stop_distance=20), 0.0)
    assert d.state == BehaviorState.STOP and d.target_speed == 0.0
    d = run(bp, WorldSummary(0.0, 15, stop_distance=None), 0.1, 0.5)
    assert d.state == BehaviorState.STOP  # dwell not elapsed
    d = run(bp, WorldSummary(0.0, 15, stop_distance=None), 0.6, 1.5)
    assert d.state == BehaviorState.LANE_FOLLOW


def test_far_stop_limits_speed_without_stopping_state():
    bp = BehaviorPlanner()
    d = bp.update(WorldSummary(10, 15, stop_distance=200), 0.0)
    assert d.state == BehaviorState.LANE_FOLLOW and d.target_speed == 15


def test_emergency_has_priority():
    bp = BehaviorPlanner()
    run(bp, WorldSummary(20, 25, lane_change_request=1), 0, 0.5)
    d = bp.update(WorldSummary(20, 25, lane_change_request=1, min_ttc=0.8), 0.6)
    assert d.state == BehaviorState.EMERGENCY_STOP and d.target_speed == 0 and d.lane_change_direction == 0
    d = bp.update(WorldSummary(5, 25), 0.7)
    assert d.state == BehaviorState.EMERGENCY_STOP  # still moving
    d = bp.update(WorldSummary(0.0, 25), 0.8)
    assert d.state == BehaviorState.STOP
    d = run(bp, WorldSummary(0.0, 25), 0.9, 2.0)
    assert d.state == BehaviorState.LANE_FOLLOW
