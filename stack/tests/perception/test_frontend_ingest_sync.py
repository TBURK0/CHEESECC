import numpy as np
import pytest

from cheesestack.core import MessageBus, SimClock
from cheesestack.perception.camera_ingest import (
    ArraySource, CameraIngest, CameraIngestConfig, FrameRingBuffer, NpyDirectorySource, SyntheticSource, frames_from,
)
from cheesestack.perception.sensor_sync import (
    ApproximateTimeConfig, ApproximateTimeSynchronizer, StreamBuffer, lerp_payload, match_interpolated, match_nearest,
)


def test_synthetic_source_rate_and_shape():
    clock = SimClock()
    ing = CameraIngest(SyntheticSource(width=32, height=24, fps=10.0), CameraIngestConfig(expected_fps=10.0), clock=clock)
    for _ in range(10):
        clock.advance(0.1)
        ing.tick()
    assert ing.stats.received == 11  # t=0..1.0 inclusive
    f = ing.latest()
    assert f.image.shape == (24, 32, 3) and f.image.dtype == np.float32
    assert 0.0 <= f.image.min() and f.image.max() <= 1.0
    assert abs(ing.stats.fps_estimate - 10.0) < 1e-6
    assert ing.stats.dropped_seq == 0


def test_drop_and_latency_stats():
    clock = SimClock()
    src = SyntheticSource(fps=10.0, drop_every=4, latency=0.05)
    ing = CameraIngest(src, CameraIngestConfig(expected_fps=10.0), clock=clock)
    for _ in range(20):
        clock.advance(0.1)
        ing.tick()
    s = ing.stats
    assert s.dropped_seq >= 4
    assert s.dropped_time == s.dropped_seq
    assert 0.0 < s.drop_rate < 0.5
    assert s.latency_mean >= 0.05 - 1e-9 and s.latency_max < 0.1 + 1e-9


def test_ring_buffer_overflow_and_closest():
    rb = FrameRingBuffer(3)
    for f in frames_from([np.zeros((2, 2))] * 5, fps=10):
        rb.push(f)
    assert len(rb) == 3 and rb.overflow_drops == 2
    assert rb.closest(0.31).header.seq == 3
    assert rb.pop().header.seq == 2


def test_array_and_npy_sources_publish(tmp_path):
    imgs = [np.full((4, 4), i, dtype=np.float32) for i in range(3)]
    for i, im in enumerate(imgs):
        np.save(tmp_path / f"{i:03d}.npy", im)
    (tmp_path / "stamps.txt").write_text("0.0\n0.05\n0.1\n")
    bus = MessageBus()
    got = []
    bus.subscribe("/cam", lambda t, m: got.append(m))
    ing = CameraIngest(NpyDirectorySource(str(tmp_path)), CameraIngestConfig(topic="/cam", buffer_capacity=2), bus=bus, publish=True)
    assert len(ing.poll(1.0)) == 3
    assert [f.header.seq for f in got] == [0, 1, 2]
    assert ing.source.exhausted() and ing.stats.overflow == 1
    with pytest.raises(ValueError):
        ArraySource(imgs, [0.0])


def test_stream_buffer_nearest_interp():
    b = StreamBuffer("imu")
    for t in [0.0, 0.2, 0.1, 0.3]:  # out of order insert
        b.add(t, np.array([t * 10, 1.0]))
    assert b.stamps == [0.0, 0.1, 0.2, 0.3]
    assert b.nearest(0.14)[0] == pytest.approx(0.1)
    assert b.nearest(0.5, tolerance=0.1) is None
    np.testing.assert_allclose(b.interpolate(0.15), [1.5, 1.0])
    assert b.interpolate(0.4) is None


def test_lerp_angle_wrap():
    out = lerp_payload([3.0, 0.0], [-3.0, 1.0], 0.5, angle_indices=(0,))
    assert abs(abs(out[0]) - np.pi) < 0.2
    assert out[1] == pytest.approx(0.5)


def test_match_helpers():
    a, b = StreamBuffer("a"), StreamBuffer("b")
    for i in range(5):
        a.add(i * 0.1, float(i))
        b.add(i * 0.1 + 0.01, float(i) * 2)
    m = match_nearest({"a": a, "b": b}, 0.2, tolerance=0.02)
    assert m["a"][1] == 2.0 and m["b"][1] == 4.0
    assert match_nearest({"a": a, "b": b}, 0.25, tolerance=0.02) is None
    mi = match_interpolated({"a": a, "b": b}, 0.21)
    assert mi["a"] == pytest.approx(2.1) and mi["b"] == pytest.approx(4.0)


def test_approximate_time_sync():
    out = []
    sync = ApproximateTimeSynchronizer(["cam", "imu", "gps"], ApproximateTimeConfig(slop=0.01), callback=out.append)
    rng = np.random.default_rng(0)
    for k in range(10):
        t = k * 0.1
        sync.add("cam", t + rng.uniform(-0.003, 0.003), f"cam{k}")
        for j in range(5):  # imu at 50 Hz
            sync.add("imu", t + j * 0.02, f"imu{k}.{j}")
        if k % 2 == 0:  # gps at 5 Hz
            sync.add("gps", t + 0.002, f"gps{k}")
    assert len(out) >= 4
    for s in out:
        assert s.spread <= 0.01 + 1e-9
        assert s["cam"][3:] == s["gps"][3:]
    stamps = [s.stamp for s in out]
    assert stamps == sorted(stamps)
