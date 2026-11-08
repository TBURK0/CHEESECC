# Stop sign and traffic light video labeling

`tools/vision/label_video.py` reads a local video, runs Ultralytics YOLO on every
frame, and uses OpenCV to draw boxes and confidence labels. It accepts pretrained
COCO weights or custom weights with `stop sign` and `traffic light` class names.

## Install and run

Use Python 3.10+ in a virtual environment. This development host only has Python
3.8; install a newer Python first for the optional detector environment. From the
repo root, with the newer Python selected:

```powershell
python -m venv .venv-vision
.\.venv-vision\Scripts\python.exe -m pip install -r tools/vision/requirements.txt
.\.venv-vision\Scripts\python.exe tools/vision/label_video.py "C:\videos\drive.mp4" --output "runs\drive01" --export-yolo
```

Add `--show` for a live preview (`q` stops), `--device cpu` to force CPU inference,
or `--device 0` for a configured CUDA GPU. `--conf 0.4`, `--imgsz 1280` and
`--max-frames 300` control threshold, resolution and processing length.
`--model C:\models\best.pt` uses local weights. The default `yolo11n.pt` is
downloaded by Ultralytics on first use unless cached. Offline use requires
dependencies and weights provided beforehand. Existing output runs are refused.

## Outputs

| File | Contents |
|---|---|
| `annotated.mp4` | Boxes, class names and confidence; no audio |
| `detections.jsonl` | One record per frame: nominal timestamp and pixel boxes, including empty detections |
| `summary.json` | Source, dimensions, FPS, frame and detection counts |
| `images/frame_00000000.jpg` | Original unannotated frames; with `--export-yolo` |
| `labels/frame_00000000.txt` | Paired normalized YOLO labels; with `--export-yolo` |
| `dataset.yaml` | Class mapping; add reviewed train/validation split paths before training |

Export IDs are **0 = stop sign, 1 = traffic light**, remapped from model class
names. Label lines are `class_id center_x center_y width height`, coordinates
normalized to 0–1. Confidence is in JSON, not an extra training-label field.
Empty detections produce empty label files.

These are automatic predictions for human review, not ground truth. Correct false
positives, missed objects and box boundaries before training. Split by recording
or scene so adjacent frames do not leak into both training and validation.
The traffic light label identifies the object; red/yellow/green state recognition
is not included. This tool makes no driving decisions.

Output uses nominal FPS, or `--fps` to override metadata. Audio and variable frame
timing are not preserved. OpenCV may report a decode failure as end-of-file;
compare frame counts for damaged recordings. Failed runs may leave partial output
without `summary.json`.

## Verification

```powershell
python -B -m unittest discover -s tools/vision -p test_label_video.py -v
```

Five tests cover class remapping, the YOLO API adapter, box normalization, empty labels, and actual
OpenCV video read/write with injected detector boxes. Real YOLO inference has not
run on this host because Ultralytics and model weights are unavailable.

API references: [Ultralytics prediction](https://docs.ultralytics.com/modes/predict/),
[COCO classes](https://docs.ultralytics.com/datasets/detect/coco/),
[OpenCV VideoWriter](https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html).
