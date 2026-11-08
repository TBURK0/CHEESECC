import numpy as np

from cheesestack.core import BoundingBox2D, Detection, Header, ObjectClass
from cheesestack.perception.detection import (BlobDetector, ModelDetector, connected_components, fill_holes,
                                              nms, nms_detections, rgb_to_hsv, sobel_xy)

H = Header(stamp=1.0, frame_id="camera_front")


def test_connected_components_counts_blobs():
    m = np.zeros((30, 30), bool)
    m[2:6, 2:6] = True
    m[10:20, 10:12] = True
    m[25, 25] = True
    m[26, 26] = True  # diagonal neighbour
    lab4, n4 = connected_components(m, 4)
    lab8, n8 = connected_components(m, 8)
    assert n4 == 4 and n8 == 3
    assert (lab4[m] > 0).all() and (lab4[~m] == 0).all()
    # snake-shaped component converges to a single label
    s = np.zeros((20, 20), bool)
    s[::4, :] = True
    s[:, 19] = True
    assert connected_components(s)[1] == 1


def test_fill_holes_and_hsv_and_sobel():
    m = np.zeros((10, 10), bool)
    m[2:8, 2:8] = True
    m[4:6, 4:6] = False
    assert fill_holes(m)[4, 4]
    hsv = rgb_to_hsv(np.array([[[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]], np.float32))
    np.testing.assert_allclose(hsv[0, :, 0], [0, 1 / 3, 2 / 3], atol=1e-6)
    img = np.zeros((5, 10), np.float32)
    img[:, 5:] = 1.0
    gx, gy = sobel_xy(img)
    assert gx[2, 5] > 0.4 and np.abs(gy).max() < 1e-6


def test_nms_suppresses_overlaps():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)
    assert keep.tolist() == [0, 2]
    dets = [Detection(H, ObjectClass.CAR, 0.9, BoundingBox2D(0, 0, 10, 10)),
            Detection(H, ObjectClass.PEDESTRIAN, 0.8, BoundingBox2D(1, 1, 10, 10)),
            Detection(H, ObjectClass.CAR, 0.5, BoundingBox2D(1, 1, 10, 10))]
    assert len(nms_detections(dets, class_aware=True)) == 2
    assert len(nms_detections(dets, class_aware=False)) == 1


def test_blob_detector_finds_objects():
    img = np.full((60, 80, 3), 0.5, np.float32)
    img[10:30, 10:20] = [0.9, 0.1, 0.1]
    img[40:50, 50:75] = [0.1, 0.1, 0.9]
    dets = BlobDetector().detect(img, H)
    assert len(dets) == 2
    boxes = sorted((d.box2d.u, d.box2d.v, d.box2d.w, d.box2d.h) for d in dets)
    assert boxes[0] == (10, 10, 10, 20) and boxes[1] == (50, 40, 25, 10)
    assert all(0 < d.score <= 1 for d in dets)
    # background differencing
    bd = BlobDetector()
    bg = np.full((60, 80, 3), 0.5, np.float32)
    bd.set_background(bg)
    assert len(bd.detect(bg, H)) == 0


def test_model_detector_adapter():
    def infer(_img):
        return np.array([[0, 0, 10, 10], [1, 1, 10, 10], [30, 30, 40, 40]]), np.array([0.9, 0.8, 0.1]), \
            ["car", "car", "person"]
    dets = ModelDetector(infer, lambda s: ObjectClass.CAR if s == "car" else ObjectClass.PEDESTRIAN).detect(None, H)
    assert len(dets) == 1 and dets[0].cls == ObjectClass.CAR
