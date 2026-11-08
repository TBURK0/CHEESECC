"""RGB-D helpers; optional dependencies imported only by camera processing."""
import math


def signal_state(candidates, threshold=0.8):
    valid = [(name, confidence) for name, confidence in candidates
             if name in ('RED', 'YELLOW', 'GREEN') and math.isfinite(confidence) and threshold <= confidence <= 1]
    if len(valid) != 1:
        return 'UNKNOWN'
    return valid[0][0]


def classical_path(bgr):
    """White lane markings baseline; returns confidence, normalized center offset."""
    import cv2
    import numpy as np
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        return 0.0, 0.0
    roi = bgr[bgr.shape[0]//2:]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([179, 70, 255]))
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return 0.0, 0.0
    return min(1.0, len(xs)/(mask.size*0.1)), float(xs.mean()/mask.shape[1]*2-1)


def classify_light_crop(bgr):
    """Conservative color classifier inside a pretrained traffic-light detection."""
    import cv2
    import numpy as np
    if bgr.size == 0:
        return 'UNKNOWN', 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    bright = (sat > 120) & (val > 180)
    count = int(bright.sum())
    if count < 8:
        return 'UNKNOWN', 0.0
    scores = [('RED', int((bright & ((hue < 10) | (hue > 170))).sum())/count),
              ('YELLOW', int((bright & (hue >= 15) & (hue <= 35)).sum())/count),
              ('GREEN', int((bright & (hue >= 40) & (hue <= 95)).sum())/count)]
    state = signal_state(scores)
    return state, max(c for _, c in scores)


class PretrainedDetector:
    def __init__(self, weights):
        # No implicit model downloads; operator supplies reviewed local weights.
        from pathlib import Path
        if not Path(weights).is_file():
            raise ValueError('local reviewed weights required')
        from ultralytics import YOLO
        self.model = YOLO(weights)

    def detect(self, image, stamp):
        objects = []
        for result in self.model.predict(image, verbose=False, conf=0.5):
            for box in result.boxes:
                name = result.names[int(box.cls.item())]
                if name not in ('stop sign', 'traffic light'):
                    continue
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                x1, y1 = max(0, x1), max(0, y1)
                state, confidence = ('NONE', 1.0)
                if name == 'traffic light':
                    state, confidence = classify_light_crop(image[y1:max(y1,y2), x1:max(x1,x2)])
                objects.append(dict(class_name=name, confidence=float(box.conf.item()),
                                    stamp=stamp, signal=state, signal_confidence=confidence,
                                    box=[x1, y1, x2, y2]))
        return objects
