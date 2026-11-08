import numpy as np

from cheesestack.core import Header
from cheesestack.perception.drivable_space import (RegionGrowingSegmenter, boundary_to_ground,
                                                   free_space_boundary)
from cheesestack.perception.road_edge import RoadEdgeConfig, RoadEdgeDetector, ransac_polyfit

H = Header(0.0, "camera_front")


def road_scene(h=80, w=100, seed=0):
    rng = np.random.default_rng(seed)
    img = np.empty((h, w, 3), np.float32)
    img[:] = [0.2, 0.6, 0.2]  # grass
    img[:20] = [0.6, 0.7, 0.9]  # sky
    for v in range(20, h):
        half = 10 + (v - 20) * 0.6
        img[v, int(50 - half):int(50 + half)] = [0.4, 0.4, 0.4]
    img[40:55, 45:60] = [0.8, 0.1, 0.1]  # obstacle on the road
    img += rng.normal(0, 0.01, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def test_free_space_boundary():
    m = np.ones((5, 3), bool)
    m[1, 0] = False
    m[4, 2] = False
    assert free_space_boundary(m).tolist() == [1, -1, 4]


def test_region_growing_segments_road_and_boundary():
    img = road_scene()
    ds = RegionGrowingSegmenter().segment(img, H)
    assert ds.confidence > 0.9
    assert ds.mask[70, 50] and not ds.mask[70, 2] and not ds.mask[10, 50]
    assert not ds.mask[45, 50]  # obstacle
    # columns behind the obstacle stop at the obstacle bottom (row 54)
    assert ds.boundary_v[50] == 54
    # free column beside obstacle reaches higher up
    assert ds.boundary_v[40] < 40
    g = boundary_to_ground(ds.boundary_v, np.eye(3), img.shape[0])
    assert g.shape == (100, 2) and g[50, 1] == 55


def test_ransac_polyfit_rejects_outliers():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 50, 60)
    y = 0.01 * x ** 2 + 0.5 * x + 3 + rng.normal(0, 0.2, x.size)
    y[::7] += 30
    c, inl = ransac_polyfit(x, y, 2, tol=1.0, rng=np.random.default_rng(1))
    np.testing.assert_allclose(c, [0.01, 0.5, 3], atol=0.1)
    assert (~inl[::7]).all()


def test_road_edges_from_mask_and_metric():
    h, w = 100, 100
    mask = np.zeros((h, w), bool)
    mask[:, 30:71] = True  # straight road, cols 30..70
    cfg = RoadEdgeConfig(m_per_px_x=0.1, m_per_px_y=0.1)
    edges = RoadEdgeDetector(cfg).detect(H, mask=mask)
    assert edges.left is not None and edges.right is not None
    assert abs(np.polyval(edges.left.coeffs_px, 50) - 30) < 0.5
    assert abs(np.polyval(edges.right.coeffs_px, 50) - 70) < 0.5
    assert abs(edges.road_width_m - 4.0) < 0.15
    # crude mask path on a synthetic image
    e2 = RoadEdgeDetector().detect(H, image=road_scene())
    assert e2.left is not None and e2.right is not None
