import numpy as np
import pytest

from cheesestack.core import Pose2D
from cheesestack.infra import (ArrayStore, CompressionConfig, RecordStore, RecordStoreConfig,
                               TimeseriesCodecConfig, compress, decode_timeseries, decompress,
                               dequantize, encode_timeseries, quantize, varint_decode, varint_encode)
from cheesestack.infra.compression import zigzag_decode, zigzag_encode


def test_record_store_segments_and_query(tmp_path):
    with RecordStore(tmp_path, RecordStoreConfig(segment_max_records=10)) as st:
        for i in range(35):
            st.append(i * 0.1, "/a" if i % 2 else "/b", {"i": i, "arr": np.arange(2)})
        assert len(st.segments) == 4
        r = st.query(1.0, 1.5)
        assert [x.data["i"] for x in r] == [10, 11, 12, 13, 14, 15]
    st2 = RecordStore(tmp_path)
    assert len(st2) == 35
    assert [x.data["i"] for x in st2.query(0, 0.5, topics=["/a"])] == [1, 3, 5]
    assert st2.time_range() == (0.0, pytest.approx(3.4))
    st2.append(10.0, "/c", Pose2D(1.0, 2.0, 0.5))
    assert st2.query(9, 11)[0].data == {"x": 1.0, "y": 2.0, "yaw": 0.5}
    st2.close()


def test_record_store_out_of_order_and_reindex(tmp_path):
    st = RecordStore(tmp_path)
    for t in [3.0, 1.0, 2.0]:
        st.append(t, "/x", t)
    st.flush()
    assert [r.t for r in st.query()] == [1.0, 2.0, 3.0]
    st.close()
    (tmp_path / "index.json").unlink()
    st2 = RecordStore(tmp_path)  # orphan segments are recovered
    assert len(st2) == 3
    st2.reindex()
    assert st2.segments[0].t_min == 1.0


def test_array_store(tmp_path):
    from cheesestack.infra import ArrayStoreConfig
    st = ArrayStore(tmp_path, ArrayStoreConfig(chunk_size=8))
    for i in range(20):
        st.append("/imu", i * 0.01, [i, -i, 0.0])
    t, d = st.query("/imu", 0.05, 0.12)
    assert np.allclose(t, np.arange(5, 13) * 0.01) and d.shape == (8, 3)
    st.flush()
    st2 = ArrayStore(tmp_path)
    t, d = st2.query("/imu")
    assert len(t) == 20 and d[-1, 1] == -19 and st2.topics() == ["/imu"]


@pytest.mark.parametrize("codec", ["none", "zlib", "lzma"])
def test_compress_roundtrip(codec):
    data = b"cheese" * 1000
    blob = compress(data, CompressionConfig(codec=codec))
    assert decompress(blob) == data
    if codec != "none":
        assert len(blob) < len(data) / 10


def test_varint_zigzag():
    x = np.array([0, -1, 1, -64, 63, 2**40, -(2**40)], dtype=np.int64)
    z = zigzag_encode(x)
    assert list(z[:3]) == [0, 1, 2]
    buf = varint_encode(z)
    dec, n = varint_decode(buf)
    assert n == len(buf) and np.array_equal(zigzag_decode(dec), x)
    assert len(varint_encode([127])) == 1 and len(varint_encode([128])) == 2
    with pytest.raises(ValueError):
        varint_decode(b"\x80")


@pytest.mark.parametrize("order", [0, 1, 2])
def test_timeseries_codec(order):
    rng = np.random.default_rng(0)
    t = np.arange(1000) * 0.01 + 100.0
    v = np.cumsum(rng.normal(0, 0.01, 1000))
    for x, step in [(t, 1e-4), (v, 1e-3)]:
        cfg = TimeseriesCodecConfig(step=step, order=order, zlib_level=6 if order == 2 else 0)
        blob = encode_timeseries(x, cfg)
        y = decode_timeseries(blob)
        assert np.max(np.abs(y - x)) <= step / 2 + 1e-9
    blob = encode_timeseries(t, TimeseriesCodecConfig(step=1e-4, order=2))
    assert len(blob) < 1100  # ~1 byte/sample vs 8 raw


def test_quantize():
    x = np.linspace(-1, 1, 11)
    q = quantize(x, -1, 1, bits=8)
    assert q.dtype == np.uint8 and q[0] == 0 and q[-1] == 255
    assert np.max(np.abs(dequantize(q, -1, 1) - x)) <= 1 / 255 + 1e-12
    assert quantize([5.0], 0, 1, bits=16)[0] == 65535
