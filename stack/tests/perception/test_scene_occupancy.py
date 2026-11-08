import numpy as np

from cheesestack.core import BoundingBox3D, Detection, Header, ObjectClass, Pose2D
from cheesestack.perception.occupancy_grid import OccupancyGrid, OccupancyGridConfig


def test_ray_casting_free_and_occupied():
    g = OccupancyGrid(OccupancyGridConfig(size_m=20, resolution=0.2))
    angles = np.linspace(-0.3, 0.3, 31)
    for _ in range(3):
        g.insert_scan((0.0, 0.0, 0.0), angles, np.full(31, 6.0))
    assert g.query(6.0, 0.0) > 0.65
    assert g.query(3.0, 0.0) < 0.35
    assert abs(g.query(-5.0, 0.0) - 0.5) < 1e-6  # unobserved
    # max-range returns clear the ray but mark nothing occupied
    g2 = OccupancyGrid(OccupancyGridConfig(size_m=20, resolution=0.2, max_range=5.0))
    g2.insert_scan((0, 0, np.pi / 2), [0.0], [np.inf])  # inf filtered
    g2.insert_scan((0, 0, np.pi / 2), [0.0], [10.0])
    assert g2.query(0.0, 3.0) < 0.5 and not g2.occupied_mask().any()


def test_detections_decay_and_shift():
    g = OccupancyGrid(OccupancyGridConfig(size_m=20, resolution=0.25))
    d = Detection(Header(0.0), ObjectClass.CAR, 1.0, box3d=BoundingBox3D(Pose2D(5.0, 2.0, 0.0), 4.0, 2.0))
    for _ in range(4):
        g.insert_detections([d])
    assert g.query(5.0, 2.0) > 0.9 and g.query(6.8, 2.8) > 0.9 and g.query(5.0, -1.0) == 0.5
    l0 = g.logodds.max()
    g.step(0.0)
    g.step(2.0)  # one time constant
    np.testing.assert_allclose(g.logodds.max(), l0 * np.exp(-1), rtol=1e-4)
    # ego drives 2 m forward: obstacle now at x=3
    g.shift(Pose2D(2.0, 0.0, 0.0))
    assert g.query(3.0, 2.0) > 0.7 and g.query(6.5, 2.0) == 0.5
    # ego turns left 90 deg: obstacle (3,2) now at (2,-3)
    g.shift(Pose2D(0.0, 0.0, np.pi / 2))
    assert g.query(2.0, -3.0) > 0.7


def test_points_and_clamping():
    cfg = OccupancyGridConfig(size_m=10, resolution=0.5, l_max=2.0)
    g = OccupancyGrid(cfg)
    for _ in range(20):
        g.insert_points(np.array([[1.0, 1.0], [100.0, 0.0]]))
    assert g.logodds.max() == np.float32(2.0)
    assert g.occupied_mask().sum() == 1
