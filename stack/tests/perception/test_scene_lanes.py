import numpy as np

from cheesestack.core import Header
from cheesestack.perception.lanes import ClassicalLaneDetector, LaneDetectorConfig, warp_perspective

H = Header(0.0, "camera_front")
CFG = dict(m_per_px_x=0.1, m_per_px_y=0.05, x_offset_m=0.0)


def synth_bev(offset_m=0.0, curvature=0.0, h=200, w=200, width_m=3.5, yellow_left=False):
    """BEV image with two lane lines y = +-width/2 - offset + k/2 x^2 (ego at column center)."""
    img = np.full((h, w, 3), 0.25, np.float32)
    ego_u = (w - 1) / 2
    for v in range(h):
        x = (h - 1 - v) * CFG["m_per_px_x"]
        for side, sgn in (("l", 1), ("r", -1)):
            y = sgn * width_m / 2 - offset_m + 0.5 * curvature * x * x
            u = int(round(ego_u - y / CFG["m_per_px_y"]))
            color = [0.9, 0.8, 0.1] if (yellow_left and side == "l") else [1.0, 1.0, 1.0]
            img[v, max(0, u - 2):u + 3] = color
    return img


def test_straight_lane_width_and_offset():
    det = ClassicalLaneDetector(LaneDetectorConfig(**CFG))
    est = det.detect(synth_bev(offset_m=0.3), H)
    assert est.valid
    assert abs(est.lane_width_m - 3.5) < 0.1
    # lane centre is 0.3 m to the right of ego -> ego is 0.3 m left of centre
    assert abs(est.center_offset_m - 0.3) < 0.05
    assert abs(est.curvature) < 2e-3 and abs(est.heading_error_rad) < 0.02
    assert est.confidence > 0.5


def test_curved_lane_curvature_and_yellow_marking():
    det = ClassicalLaneDetector(LaneDetectorConfig(**CFG, use_prior=False))
    est = det.detect(synth_bev(curvature=0.01, yellow_left=True), H)
    assert est.valid
    assert abs(est.curvature - 0.01) < 2.5e-3


def test_prior_search_tracks_and_reset_on_loss():
    det = ClassicalLaneDetector(LaneDetectorConfig(**CFG))
    for off in (0.0, 0.05, 0.1, 0.15):
        est = det.detect(synth_bev(offset_m=off), H)
        assert est.valid
    assert 0.05 < est.center_offset_m < 0.2  # EMA-smoothed
    blank = np.full((200, 200, 3), 0.25, np.float32)
    assert not det.detect(blank, H).valid
    assert det._prev is None


def test_warp_perspective_identity_and_translation():
    img = np.arange(48, dtype=np.float32).reshape(6, 8)
    np.testing.assert_array_equal(warp_perspective(img, np.eye(3), (6, 8)), img)
    T = np.array([[1, 0, 2], [0, 1, 1], [0, 0, 1]], float)
    out = warp_perspective(img, T, (6, 8))
    assert out[1, 2] == img[0, 0] and out[0, 0] == 0
    # detector accepts a homography and warps internally
    det = ClassicalLaneDetector(LaneDetectorConfig(**CFG, homography=np.eye(3), bev_shape=(200, 200)))
    assert det.detect(synth_bev(), H).valid
