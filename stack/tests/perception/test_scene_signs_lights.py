import numpy as np

from cheesestack.core import BoundingBox2D, Header, ObjectClass
from cheesestack.perception.traffic_lights import (HSVTrafficLightClassifier, LightState, TrafficLightFilter,
                                                   TrafficLightObservation)
from cheesestack.perception.traffic_signs import ColorShapeSignRecognizer, SignType, SignVoter

H = Header(0.0, "camera_front")
RED, BLUE, YELLOW, WHITE = [0.85, 0.05, 0.05], [0.05, 0.2, 0.85], [0.95, 0.8, 0.05], [1.0, 1.0, 1.0]


def _grid(n=40):
    v, u = np.mgrid[0:n, 0:n]
    return u - (n - 1) / 2, v - (n - 1) / 2


def put(img, mask, top, left, color):
    h, w = mask.shape
    img[top:top + h, left:left + w][mask] = color


def circle(n=40):
    x, y = _grid(n)
    return x * x + y * y <= (n / 2) ** 2


def octagon(n=40):
    x, y = _grid(n)
    r = n / 2
    return (np.abs(x) <= r) & (np.abs(y) <= r) & (np.abs(x) + np.abs(y) <= r * 1.414)


def triangle(n=40, up=True):
    x, y = _grid(n)
    yy = y if up else -y
    return (yy <= n / 2) & (np.abs(x) <= (yy + n / 2) / 2)


def scene(*items):
    img = np.full((120, 240, 3), 0.5, np.float32)
    for mask, top, left, color in items:
        put(img, mask, top, left, color)
    return img


def sign_types(img):
    return {d.attributes["sign_type"] for d in ColorShapeSignRecognizer().recognize(img, H)}


def test_shapes_to_sign_types():
    ring_in = circle(40) & ~np.pad(circle(28), 6)
    img = scene((octagon(), 10, 10, RED), (triangle(up=False), 10, 70, RED), (circle(), 10, 130, BLUE),
                (ring_in, 70, 10, RED), (triangle(up=True), 70, 70, RED))
    put(img, np.pad(circle(28), 6), 70, 10, WHITE)
    assert sign_types(img) == {SignType.STOP, SignType.YIELD, SignType.MANDATORY, SignType.SPEED_LIMIT,
                               SignType.WARNING}
    dets = ColorShapeSignRecognizer().recognize(img, H)
    assert all(d.cls == ObjectClass.TRAFFIC_SIGN and 0 < d.score <= 1 for d in dets)


def test_no_entry_and_info_sign():
    ne = circle(40).copy()
    bar = np.zeros((40, 40), bool)
    bar[17:23, 6:34] = True
    img = scene((ne & ~bar, 10, 10, RED), (np.ones((30, 40), bool), 10, 100, BLUE))
    put(img, bar, 10, 10, WHITE)
    assert sign_types(img) == {SignType.NO_ENTRY, SignType.INFORMATION}


def test_voter_requires_stable_majority():
    rec, voter = ColorShapeSignRecognizer(), SignVoter()
    img = scene((octagon(), 10, 10, RED))
    first = voter.update(rec.recognize(img, H))
    assert first == []  # one frame is not enough
    for _ in range(3):
        out = voter.update(rec.recognize(img, H))
    assert len(out) == 1 and out[0].attributes["sign_type"] == SignType.STOP
    # a single misclassified frame doesn't flip it
    d = rec.recognize(img, H)
    d[0].attributes["sign_type"] = SignType.YIELD
    out = voter.update(d)
    assert out[0].attributes["sign_type"] == SignType.STOP
    for _ in range(7):
        voter.update([])
    assert voter.hyps == []


def housing(lit):
    img = np.full((60, 20, 3), 0.1, np.float32)
    pos = {"red": 10, "yellow": 30, "green": 50}
    col = {"red": [1, 0.1, 0.1], "yellow": [1, 0.75, 0.05], "green": [0.1, 1, 0.4]}
    if lit:
        y0 = pos[lit]
        v, u = np.mgrid[0:60, 0:20]
        img[(v - y0) ** 2 + (u - 10) ** 2 <= 36] = col[lit]
    return img


def test_hsv_light_states():
    clf = HSVTrafficLightClassifier()
    assert clf.classify(housing("red"), H).state == LightState.RED
    assert clf.classify(housing("yellow"), H).state == LightState.YELLOW
    assert clf.classify(housing("green"), H).state == LightState.GREEN
    assert clf.classify(housing(None), H).state == LightState.UNKNOWN
    big = np.zeros((100, 100, 3), np.float32)
    big[20:80, 40:60] = housing("green")
    assert clf.classify(big, H, BoundingBox2D(40, 20, 20, 60)).state == LightState.GREEN


def test_light_filter_suppresses_flicker():
    clf, filt = HSVTrafficLightClassifier(), TrafficLightFilter()
    states = [filt.update(clf.classify(housing("green"), H)).state for _ in range(5)]
    assert states[-1] == LightState.GREEN
    # single red glitch does not flip the output
    assert filt.update(clf.classify(housing("red"), H)).state != LightState.RED
    assert filt.update(clf.classify(housing("green"), H)).state == LightState.GREEN
    # dropouts (unknown) keep the belief for a while
    assert filt.update(TrafficLightObservation(H, LightState.UNKNOWN, 1.0, {LightState.UNKNOWN: 1.0})).state \
        == LightState.GREEN
    # sustained yellow switches
    for _ in range(4):
        out = filt.update(clf.classify(housing("yellow"), H))
    assert out.state == LightState.YELLOW
