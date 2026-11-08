# cheesestack.ml

ML infrastructure for the team's camera models (lanes, objects, signs, lights). numpy only. A real framework (torch / ONNX / CoreML) plugs in behind `training.Model` and `annotation.PreLabeler`.

| Module | Capability | Works now | Key TODOs |
|---|---|---|---|
| `dataset.py` | Dataset collection | `SampleRecord` (image + sensors + labels), JSONL manifests, triggers (low-confidence, model disagreement by IoU, rare-class online counts, periodic), `DatasetCollector` with a rate limit, a sample budget and `.npy` image sink | pre/post-trigger ring buffer, upload queue, PII blurring |
| `annotation.py` | Annotation tools | Box/polyline/mask/tag labels, `LabelSchema.validate`, COCO JSON (RLE masks, polylines) and YOLO txt round-trip, `ColorBlobPreLabeler` (traffic-light blobs), `auto_label` hook | polygon/compressed RLE, detector-backed pre-labeler |
| `versioning.py` | Dataset versioning | SHA-256 content hash (labels + pixels), content-addressed snapshots, `DatasetRegistry` (commit/tag/load/diff), hash-based train/val/test splits that can be grouped (e.g. by drive) | object-store backend, dedup image store, locking |
| `augmentation.py` | Augmentation | Seeded `Compose`; HFlip (with class swap), RandomCrop, BrightnessContrast, GaussianNoise, separable GaussianBlur, Cutout. Boxes, polylines and masks stay consistent with the pixels; dtype is preserved | affine/perspective, colour jitter, mosaic, weather |
| `training.py` | Training | `Model` ABC, `SoftmaxRegression` and `MLPClassifier` (SGD + momentum, L2; gradient checked), `Trainer` (minibatches, LR decay), `EarlyStopping` (restores best), `ModelCheckpoint` (.npz), `History` | data loaders, framework backend, distributed training, experiment tracking |
| `tuning.py` | Hyperparameter tuning | Grid search, random search (Uniform/LogUniform/Int/Choice), successive halving | Hyperband/ASHA, checkpoint resume |
| `metrics.py` | Model validation | Accuracy, confusion matrix, P/R/F1 (per class, macro, micro), IoU matrix, AP (COCO-101 / VOC), mAP over classes × IoU thresholds, TuSimple-style lane metrics | COCO area ranges/maxDets, crowd regions |
| `benchmark.py` | Benchmarking | Warmup + timed iterations, mean/std/p50/p90/p99/max, throughput, tracemalloc peak, parameter count / size (including a quantised dtype estimate), injectable timer | on-device (ANE/GPU) runs |
| `drift.py` | Drift detection | PSI (reference-quantile bins), two-sample KS with Kolmogorov p-value, embedding mean shift, `DriftDetector` per-feature `core.Fault` alerts | ADWIN/Page-Hinkley, RBF MMD, standard image features |
| `monitoring.py` | Online monitoring | Rolling window: accuracy on labelled events, mean confidence, low-confidence rate, class mix, confidence histogram, ECE, latency percentiles; threshold `Fault`s; `MonitorNode` on the bus | per-condition slices, label-delay joins, persistence |
