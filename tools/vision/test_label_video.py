import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from label_video import Detector, annotate_video, class_ids, normalize_box


class VideoTests(unittest.TestCase):
    def test_yolo_adapter_filters_and_remaps_classes(self):
        class Tensor:
            def __init__(self, data):
                self.data = data
            def cpu(self):
                return self
            def tolist(self):
                return self.data
        boxes = SimpleNamespace(xyxy=Tensor([[1, 2, 20, 30]]), conf=Tensor([0.9]), cls=Tensor([11]))
        result = SimpleNamespace(boxes=boxes)
        model = SimpleNamespace(names={9: 'traffic light', 11: 'stop sign'})
        seen = {}
        def predict(**kwargs):
            seen.update(kwargs)
            return [result]
        model.predict = predict
        with patch.dict('sys.modules', {'ultralytics': SimpleNamespace(YOLO=lambda _: model)}):
            detector = Detector('test-only.pt', 0.4, 640, 'cpu')
            self.assertEqual(detector('frame'), [(0, 0.9, [1, 2, 20, 30])])
        self.assertEqual(seen['classes'], [9, 11])
        self.assertEqual(seen['device'], 'cpu')
        self.assertEqual(seen['conf'], 0.4)

    def test_name_mapping(self):
        self.assertEqual(class_ids({9: 'traffic light', 11: 'stop sign'}), {9: 1, 11: 0})
        with self.assertRaises(ValueError):
            class_ids(['car'])

    def test_clipping_and_normalization(self):
        normalized, box = normalize_box([-10, 10, 110, 50], 100, 100)
        self.assertEqual(normalized, (0.5, 0.3, 1.0, 0.4))
        self.assertEqual(box, (0, 10, 100, 50))
        with self.assertRaises(ValueError):
            normalize_box([3, 2, 1, 4], 100, 100)

    def test_real_opencv_io_with_injected_detections(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'input.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10, (160, 120))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
            writer.release()
            result = annotate_video(source, root / 'out',
                                    lambda _: [(0, 0.9, [20, 30, 60, 70]),
                                               (1, 0.8, [100, 10, 120, 50])], True)
            self.assertEqual(result['frames'], 3)
            self.assertEqual(result['detections'], 6)
            records = [json.loads(s) for s in (root / 'out/detections.jsonl').read_text().splitlines()]
            self.assertEqual(records[-1]['time_s'], 0.2)
            labels = (root / 'out/labels/frame_00000000.txt').read_text().splitlines()
            self.assertEqual(labels[0], '0 0.250000 0.416667 0.250000 0.333333')
            cap = cv2.VideoCapture(str(root / 'out/annotated.mp4'))
            ok, annotated = cap.read()
            cap.release()
            self.assertTrue(ok)
            self.assertGreater(int(annotated[30, 20].sum()), 30)
            with self.assertRaises(FileExistsError):
                annotate_video(source, root / 'out', lambda _: [])

    def test_empty_frame_labels(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            writer = cv2.VideoWriter(str(root / 'in.avi'), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 64))
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
            writer.release()
            annotate_video(root / 'in.avi', root / 'out', lambda _: [], True)
            self.assertEqual((root / 'out/labels/frame_00000000.txt').read_text(), '')


if __name__ == '__main__':
    unittest.main()
