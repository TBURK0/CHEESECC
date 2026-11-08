import numpy as np

from cheesestack.core import BoundingBox2D, Detection, Header, ObjectClass
from cheesestack.perception.classification import FeatureClassifier, crop_features, map_label
from cheesestack.perception.detection import BlobDetector


def test_map_label():
    assert map_label("Person") == ObjectClass.PEDESTRIAN
    assert map_label("traffic_light") == ObjectClass.TRAFFIC_LIGHT
    assert map_label("vehicle.car.sedan") == ObjectClass.CAR
    assert map_label("truck") == ObjectClass.TRUCK
    assert map_label("zeppelin") == ObjectClass.UNKNOWN


def test_default_priors_shape_rules():
    img = np.zeros((200, 200, 3), np.float32)
    clf = FeatureClassifier()
    ped, _, _ = clf.classify(img, BoundingBox2D(0, 0, 40, 100), metric_height_m=1.75)
    car, _, _ = clf.classify(img, BoundingBox2D(0, 0, 160, 80), metric_height_m=1.5)
    assert ped == ObjectClass.PEDESTRIAN
    assert car == ObjectClass.CAR


def test_fit_and_classify_detections():
    rng = np.random.default_rng(0)
    X, y = [], []
    for _ in range(30):
        X.append([2.5 + rng.normal(0, .2), 0.5, 0.5, 0.2, 0.5]); y.append(ObjectClass.PEDESTRIAN)
        X.append([0.5 + rng.normal(0, .05), 0.4, 0.9, 0.2, 0.5]); y.append(ObjectClass.CAR)
    clf = FeatureClassifier().fit(np.array(X), y)
    probs = clf.predict_proba(np.array([2.4, 0.5, 0.5, 0.2, 0.5]))
    assert probs[ObjectClass.PEDESTRIAN] > 0.9
    img = np.full((100, 100, 3), 0.5, np.float32)
    img[10:60, 10:30] = 0.0  # tall thin blob, h/w 2.5
    dets = BlobDetector(classifier=clf).detect(img, Header(0.0))
    assert len(dets) == 1
    d = Detection(Header(0.0), ObjectClass.UNKNOWN, 1.0, BoundingBox2D(10, 10, 20, 50))
    out = clf.classify_detections(img, [d, Detection(Header(0.0), ObjectClass.CAR, 0.5)])
    assert out[0].cls == ObjectClass.PEDESTRIAN and "class_probs" in out[0].attributes
    assert out[1].cls == ObjectClass.CAR
    f = crop_features(img, BoundingBox2D(10, 10, 20, 50))
    assert f.shape == (5,) and abs(f[0] - 2.5) < 1e-9
