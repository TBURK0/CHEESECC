"""Image-space class-aware tracking and robust aligned depth localization."""
import math
import statistics


def iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / area if area > 0 else 0.0


class BoxTracker:
    """Greedy IoU association with one-to-one matches and expiry; no identity guarantee."""
    def __init__(self, ttl=0.5, threshold=0.25):
        self.ttl, self.threshold = ttl, threshold
        self.tracks, self.next_id, self.last_time = {}, 1, None

    def update(self, detections, now):
        if not math.isfinite(now):
            raise ValueError('invalid frame time')
        if self.last_time is not None and now < self.last_time:
            self.tracks.clear()
        self.last_time = now
        self.tracks = {i: t for i, t in self.tracks.items() if 0 <= now - t['stamp'] <= self.ttl}
        unused, result = set(self.tracks), []
        for detection in sorted(detections, key=lambda d: d['confidence'], reverse=True):
            box = detection['xyxy']
            if (len(box) != 4 or not all(math.isfinite(v) for v in box) or
                    box[2] <= box[0] or box[3] <= box[1]):
                raise ValueError('invalid detection box')
            matches = [(iou(box, self.tracks[i]['xyxy']), i) for i in unused
                       if detection['name'] == self.tracks[i]['name']]
            score, identifier = max(matches, default=(0, None))
            if score < self.threshold or identifier is None:
                identifier, self.next_id = self.next_id, self.next_id + 1
            else:
                unused.remove(identifier)
            track = dict(detection, track_id=identifier, stamp=now)
            self.tracks[identifier] = track
            result.append(track)
        return result


def detection_position(box, rows, scale, intrinsics, transform):
    """Median central box depth, requiring >=3 valid aligned-depth pixels."""
    if not rows or not rows[0]:
        return None
    fx, fy, cx, cy = intrinsics
    if not all(math.isfinite(v) for v in (fx, fy, cx, cy, scale)) or min(fx, fy, scale) <= 0:
        return None
    x1, y1, x2, y2 = box
    u, v = (x1 + x2) / 2, (y1 + y2) / 2
    rx, ry = max(1, (x2 - x1) / 8), max(1, (y2 - y1) / 8)
    samples = []
    for y in range(max(0, int(v - ry)), min(len(rows), int(v + ry) + 1)):
        for x in range(max(0, int(u - rx)), min(len(rows[y]), int(u + rx) + 1)):
            z = rows[y][x] * scale
            if math.isfinite(z) and 0.1 <= z <= 4.0:
                samples.append(z)
    if len(samples) < 3:
        return None
    z = statistics.median(samples)
    return transform.apply(((u - cx) * z / fx, (v - cy) * z / fy, z))
