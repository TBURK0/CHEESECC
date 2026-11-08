"""YOLO stop-sign/traffic-light video annotation and reviewable YOLO label export."""
import argparse
import json
import math
from pathlib import Path

CLASSES = ('stop sign', 'traffic light')


def class_ids(names):
    items = names.items() if isinstance(names, dict) else enumerate(names)
    found = {int(i): CLASSES.index(name) for i, name in items if name in CLASSES}
    if set(found.values()) != {0, 1}:
        raise ValueError('model must contain both stop sign and traffic light classes')
    return found


def normalize_box(box, width, height):
    if width <= 0 or height <= 0 or len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise ValueError('invalid bounding box or frame dimensions')
    x1, y1, x2, y2 = box
    x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
    y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError('empty bounding box')
    return ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height), (x1, y1, x2, y2)


class Detector:
    def __init__(self, weights, confidence, image_size, device):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.mapping = class_ids(self.model.names)
        self.options = dict(conf=confidence, imgsz=image_size,
                            classes=list(self.mapping), verbose=False)
        if device:
            self.options['device'] = device

    def __call__(self, frame):
        result = self.model.predict(source=frame, **self.options)[0]
        return [(self.mapping[int(cls)], float(conf), box)
                for box, conf, cls in zip(result.boxes.xyxy.cpu().tolist(),
                                          result.boxes.conf.cpu().tolist(),
                                          result.boxes.cls.cpu().tolist())]


def annotate_video(source, output, detector, export_frames=False, show=False,
                   max_frames=None, fps_override=None):
    """Process every decoded frame, preserving source FPS; output directory must be new.

    detector(frame) yields (export_class_id, confidence, xyxy). No audio is copied.
    Raw images, never box overlays, accompany optional YOLO training labels.
    """
    import cv2
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_file():
        raise ValueError('input video does not exist: %s' % source)
    if max_frames is not None and max_frames < 1:
        raise ValueError('max_frames must be positive')
    capture = cv2.VideoCapture(str(source))
    writer = None
    count, total, interrupted = 0, 0, False
    try:
        if not capture.isOpened():
            raise ValueError('OpenCV could not open input video')
        fps = fps_override if fps_override is not None else capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('invalid source FPS; specify --fps')
        ok, frame = capture.read()
        if not ok:
            raise ValueError('input contains no decodable video frames')
        height, width = frame.shape[:2]
        output.mkdir(parents=True, exist_ok=False)
        if export_frames:
            (output / 'images').mkdir()
            (output / 'labels').mkdir()
            (output / 'dataset.yaml').write_text(
                '# Review predictions and split by source video before adding train/val paths.\n'
                'path: .\nnames:\n  0: stop sign\n  1: traffic light\n', encoding='utf-8')
        writer = cv2.VideoWriter(str(output / 'annotated.mp4'),
                                 cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            raise ValueError('OpenCV could not create MP4 writer (mp4v codec)')
        with (output / 'detections.jsonl').open('w', encoding='utf-8') as log:
            while ok and (max_frames is None or count < max_frames):
                if frame.shape[:2] != (height, width):
                    raise ValueError('video frame dimensions changed')
                annotated, labels, detections = frame.copy(), [], []
                for cls, confidence, box in detector(frame):
                    if cls not in (0, 1) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                        raise ValueError('invalid detector result')
                    normalized, clipped = normalize_box(box, width, height)
                    labels.append('%d %s' % (cls, ' '.join('%.6f' % v for v in normalized)))
                    detections.append(dict(class_id=cls, name=CLASSES[cls],
                                           confidence=confidence, xyxy=list(clipped)))
                    x1, y1, x2, y2 = [int(v) for v in clipped]
                    color = (0, 0, 255) if cls == 0 else (0, 220, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, '%s %.2f' % (CLASSES[cls], confidence),
                                (x1, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                writer.write(annotated)
                log.write(json.dumps(dict(frame=count, time_s=count / fps,
                                          detections=detections), allow_nan=False) + '\n')
                if export_frames:
                    stem = 'frame_%08d' % count
                    if not cv2.imwrite(str(output / 'images' / (stem + '.jpg')), frame):
                        raise ValueError('could not save frame image')
                    (output / 'labels' / (stem + '.txt')).write_text(
                        '\n'.join(labels) + ('\n' if labels else ''), encoding='utf-8')
                count, total = count + 1, total + len(detections)
                if show:
                    cv2.imshow('Stop signs and traffic lights (q to stop)', annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        interrupted = True
                        break
                if max_frames is not None and count >= max_frames:
                    break
                ok, frame = capture.read()
        summary = dict(source=str(source), frames=count, detections=total, fps=fps,
                       width=width, height=height, stopped_by_user=interrupted,
                       class_names=list(CLASSES), labels_are_predictions=True,
                       timestamps='frame index / nominal FPS; variable-frame-rate timing not preserved')
        (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        return summary
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--output', type=Path, default=Path('runs/traffic_labels'))
    parser.add_argument('--model', default='yolo11n.pt', help='COCO or custom weights with both class names')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--device', help='cpu, 0, etc.; default is Ultralytics auto selection')
    parser.add_argument('--export-yolo', action='store_true', help='save raw frames and normalized YOLO labels')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--fps', type=float, help='override missing/incorrect source FPS')
    args = parser.parse_args()
    if not 0 < args.conf <= 1 or args.imgsz < 32:
        parser.error('confidence must be in (0, 1]; imgsz must be at least 32')
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('max-frames must be positive')
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error('fps must be finite and positive')
    if not args.video.is_file() or args.output.exists():
        parser.error('input must exist and output directory must be new')
    try:
        detector = Detector(args.model, args.conf, args.imgsz, args.device)
        summary = annotate_video(args.video, args.output, detector, args.export_yolo,
                                 args.show, args.max_frames, args.fps)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, 'Video labeling failed: %s\n' % exc)
    print(json.dumps(summary, indent=2))
    print('Saved to %s' % args.output.resolve())


if __name__ == '__main__':
    main()
