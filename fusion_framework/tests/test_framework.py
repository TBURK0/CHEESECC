"""Tests for the parts of the framework that are implemented (Step 0 tooling,
node dispatch, config guard, safety math). Add a test file per layer as you
implement it, ideally one that replays a short real log you saved.
"""
import math

import pytest

from fusion.config import NotMeasuredError, load_config, require, unmeasured
from fusion.io import ReplaySource, RunLogger
from fusion.messages import Event, ImuSample, UltrasonicReading, VoPose, from_dict, to_dict
from fusion.node import FusionNode, Layer
from fusion.safety import max_safe_speed, stopping_distance


def test_message_roundtrip():
    m = ImuSample(t=1.5, gyro=(0.1, 0.2, 0.3), accel=(0.0, 0.0, 9.81), temp_c=31.0)
    assert from_dict(to_dict(m)) == m
    u = UltrasonicReading(t=2.0, sensor="front", range_m=None)
    assert from_dict(to_dict(u)) == u


def test_logger_replay_roundtrip_and_capture_order(tmp_path):
    with RunLogger(tmp_path, notes="unit test", config={"a": 1}) as log:
        log.write(ImuSample(t=0.00, gyro=(0, 0, 0), accel=(0, 0, 9.8)))
        log.write(ImuSample(t=0.10, gyro=(0, 0, 0), accel=(0, 0, 9.8)))
        # VO for the frame captured at 0.02 arrives (is logged) after t=0.10
        log.write(VoPose(t=0.02, x=1, y=2, yaw=0.1, t_result=0.11))
        log.write(Event(t=0.12, text="tap"))
    src = ReplaySource(log.path)
    assert src.header["notes"] == "unit test" and src.header["config"] == {"a": 1}
    msgs = list(src)
    assert [m.t for m in msgs] == [0.0, 0.02, 0.10, 0.12]          # capture order
    late = list(ReplaySource(log.path, deliver_at_arrival=True))
    assert [type(m).__name__ for m in late] == ["ImuSample", "ImuSample", "VoPose", "Event"]
    raw = list(ReplaySource(log.path, reorder_window_s=None))
    assert [m.t for m in raw] == [0.0, 0.10, 0.02, 0.12]           # file order


def test_replay_type_filter(tmp_path):
    with RunLogger(tmp_path) as log:
        log.write(ImuSample(t=0.0, gyro=(0, 0, 0), accel=(0, 0, 9.8)))
        log.write(Event(t=0.1, text="x"))
    assert [type(m).__name__ for m in ReplaySource(log.path, types={"Event"})] == ["Event"]


def test_config_guard():
    cfg = load_config()
    assert "imu.gyro_noise" in unmeasured(cfg)
    with pytest.raises(NotMeasuredError):
        require(cfg, "imu.gyro_noise")
    cfg["imu"]["gyro_noise"] = 0.004
    assert require(cfg, "imu.gyro_noise") == [0.004]


def test_layer_refuses_to_start_unmeasured():
    from fusion.layers import AttitudeLayer
    with pytest.raises(NotMeasuredError):
        AttitudeLayer(load_config())


def test_node_dispatch_and_state_merge():
    class Counter(Layer):
        name = "counter"

        def __init__(self, cfg):
            super().__init__(cfg)
            self.n = 0

        def on_ImuSample(self, msg):
            self.n += 1

        def state(self):
            return {"imu_count": self.n}

    class Reader(Layer):
        def on_Event(self, msg):
            self.seen = self.node.state.get("imu_count")

    c, r = Counter({}), Reader({})
    node = FusionNode({}, [c, r])
    for i in range(3):
        node.process(ImuSample(t=i * 0.002, gyro=(0, 0, 0), accel=(0, 0, 9.8)))
    node.process(Event(t=0.01, text="check"))
    assert node.state["imu_count"] == 3 and r.seen == 3


def test_safety_math_is_consistent():
    v = max_safe_speed(2.0, latency_s=0.15, decel=3.0, margin_m=0.3)
    assert math.isclose(stopping_distance(v, 0.15, 3.0) + 0.3, 2.0, rel_tol=1e-9)
    assert max_safe_speed(0.2, 0.15, 3.0, 0.3) == 0.0
    assert stopping_distance(-1.0, 0.1, 3.0) == 0.0
