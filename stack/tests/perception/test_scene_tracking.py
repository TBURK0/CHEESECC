import itertools

import numpy as np

from cheesestack.core import BoundingBox3D, Detection, Header, ObjectClass, Pose2D
from cheesestack.perception.tracking import MultiObjectTracker, TrackState, TrackerConfig, hungarian


def _brute(c):
    n, m = c.shape
    best = np.inf
    if n <= m:
        for perm in itertools.permutations(range(m), n):
            best = min(best, sum(c[i, perm[i]] for i in range(n)))
    else:
        for perm in itertools.permutations(range(n), m):
            best = min(best, sum(c[perm[j], j] for j in range(m)))
    return best


def test_hungarian_matches_bruteforce():
    rng = np.random.default_rng(1)
    for shape in [(3, 3), (4, 6), (6, 4), (5, 5), (1, 4)]:
        for _ in range(5):
            c = rng.uniform(0, 10, shape)
            pairs = hungarian(c)
            assert len(pairs) == min(shape)
            assert len({p[0] for p in pairs}) == len(pairs) and len({p[1] for p in pairs}) == len(pairs)
            assert abs(sum(c[i, j] for i, j in pairs) - _brute(c)) < 1e-9
    assert hungarian(np.zeros((0, 3))) == []


def _det(x, y, t, cls=ObjectClass.CAR):
    return Detection(Header(t, "odom"), cls, 0.9, box3d=BoundingBox3D(Pose2D(x, y, 0.0), 4.5, 1.8))


def test_tracks_two_crossing_objects_and_estimates_velocity():
    rng = np.random.default_rng(2)
    trk = MultiObjectTracker(TrackerConfig(meas_std=0.2))
    dt = 0.1
    for k in range(40):
        t = k * dt
        dets = [_det(10 + 5 * t + rng.normal(0, .1), 2.0 + rng.normal(0, .1), t),
                _det(30 - 3 * t + rng.normal(0, .1), -2.0 + rng.normal(0, .1), t, ObjectClass.PEDESTRIAN)]
        out = trk.update(dets, t)
    assert len(out) == 2
    by_cls = {o.cls: o for o in out}
    car, ped = by_cls[ObjectClass.CAR], by_cls[ObjectClass.PEDESTRIAN]
    np.testing.assert_allclose(car.velocity, [5, 0], atol=0.6)
    np.testing.assert_allclose(ped.velocity, [-3, 0], atol=0.6)
    assert car.track_id != ped.track_id and car.covariance.shape == (4, 4)


def test_lifecycle_tentative_confirmed_lost_deleted():
    trk = MultiObjectTracker(TrackerConfig(n_init=3, max_misses=2))
    assert trk.update([_det(0, 0, 0.0)], 0.0) == []  # tentative -> not output
    assert trk.tracks[0].state == TrackState.TENTATIVE
    trk.update([_det(0, 0, 0.1)], 0.1)
    out = trk.update([_det(0, 0, 0.2)], 0.2)
    assert len(out) == 1 and trk.tracks[0].state == TrackState.CONFIRMED
    tid = out[0].track_id
    trk.update([], 0.3)
    assert trk.tracks[0].state == TrackState.LOST
    out = trk.update([_det(0, 0, 0.4)], 0.4)
    assert out[0].track_id == tid and trk.tracks[0].state == TrackState.CONFIRMED
    for k in range(3):
        trk.update([], 0.5 + 0.1 * k)
    assert trk.tracks == []
    # a tentative track dies on its first miss
    trk.update([_det(5, 5, 1.0)], 1.0)
    trk.update([], 1.1)
    assert trk.tracks == []


def test_gating_spawns_new_track_for_far_measurement():
    trk = MultiObjectTracker()
    trk.update([_det(0, 0, 0.0)], 0.0)
    trk.update([_det(50, 50, 0.1)], 0.1)
    assert len(trk.tracks) == 1 and trk.tracks[0].x[0] == 50  # old tentative deleted, new spawned
