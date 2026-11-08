import numpy as np
import pytest

from cheesestack.core import BoundingBox3D, Detection, Header, ObjectClass, Pose2D
from cheesestack.perception.fusion import (DetectionFuser, FusionConfig, SourceInfo, SourceRegistry,
                                           covariance_intersection, information_merge)


def det(src, x, y, cls=ObjectClass.CAR, score=0.8, cov=None, t=0.0, **attrs):
    a = dict(attrs)
    if cov is not None:
        a["covariance"] = np.asarray(cov, float)
    return Detection(Header(t, "base_link", source=src), cls, score,
                     box3d=BoundingBox3D(Pose2D(x, y, 0.0), 4.5, 1.8), attributes=a)


def registry():
    r = SourceRegistry()
    r.register(SourceInfo("camera", position_std=1.0, range_std_gain=0.0, reliability=0.9))
    r.register(SourceInfo("radar", position_std=0.3, range_std_gain=0.0, reliability=0.7, provides_class=False))
    return r


def test_merge_rules():
    x, P = information_merge([np.array([0.0, 0]), np.array([2.0, 0])], [np.eye(2), np.eye(2)])
    np.testing.assert_allclose(x, [1, 0])
    np.testing.assert_allclose(P, 0.5 * np.eye(2))
    xc, Pc = covariance_intersection([np.array([0.0, 0]), np.array([2.0, 0])], [np.eye(2), np.eye(2)])
    assert np.trace(Pc) >= np.trace(P) - 1e-9  # CI is conservative
    x3, _ = covariance_intersection([np.zeros(2)] * 3, [np.eye(2)] * 3)
    np.testing.assert_allclose(x3, 0)


def test_fuses_camera_and_radar():
    f = DetectionFuser(registry())
    dets = [det("camera", 20.0, 1.0), det("camera", 40.0, -3.0, ObjectClass.PEDESTRIAN),
            det("radar", 20.4, 1.1, ObjectClass.UNKNOWN, 0.9), det("radar", 70.0, 0.0, ObjectClass.UNKNOWN, 0.6)]
    out = f.fuse(dets, 0.0)
    assert len(out) == 3
    fused = [d for d in out if d.attributes["n_sources"] == 2]
    assert len(fused) == 1
    d = fused[0]
    assert d.cls == ObjectClass.CAR  # radar does not vote on class
    # radar is more precise, so fused x lies closer to radar
    assert 20.3 < d.box3d.pose.x < 20.4
    assert np.trace(d.attributes["covariance"]) < 2 * 0.3 ** 2
    assert abs(d.score - (1 - 0.28 * 0.37)) < 1e-9  # noisy-OR of 0.72 and 0.63
    radar_only = [o for o in out if o.attributes["sources"] == ["radar"]][0]
    assert radar_only.cls == ObjectClass.UNKNOWN


def test_one_measurement_per_source_and_gating():
    f = DetectionFuser(registry(), FusionConfig(use_covariance_intersection=True))
    # two radar returns near one camera object: only one may join the cluster
    out = f.fuse([det("camera", 10, 0), det("radar", 10.1, 0), det("radar", 10.3, 0.2)], 0.0)
    assert sorted(o.attributes["n_sources"] for o in out) == [1, 2]
    # explicit covariance + latency compensation
    r = registry()
    r.register(SourceInfo("v2x", latency_s=0.5))
    f2 = DetectionFuser(r)
    m = f2.to_measurement(det("v2x", 10, 0, cov=np.eye(2) * 0.01, velocity=[10.0, 0.0]), stamp=0.0)
    np.testing.assert_allclose(m.z, [15.0, 0.0])
    np.testing.assert_allclose(m.R, np.eye(2) * 0.01)
    with pytest.raises(KeyError):
        f2.fuse([det("lidar", 0, 0)], 0.0)
