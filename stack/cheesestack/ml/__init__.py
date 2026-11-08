"""ML infrastructure: datasets, training, validation, drift."""

from .annotation import (
    Annotation, BoxLabel, PolylineLabel, MaskLabel, LabelSchema, ColorBlobPreLabeler, PreLabeler,
    to_coco, from_coco, to_yolo, from_yolo, mask_to_rle, rle_to_mask, auto_label,
)
from .dataset import (
    SampleRecord, TriggerContext, LowConfidenceTrigger, DisagreementTrigger, RareClassTrigger,
    PeriodicTrigger, CollectorConfig, DatasetCollector, read_manifest, write_manifest,
)
from .versioning import (
    DatasetRegistry, DatasetSnapshot, DatasetDiff, SplitConfig, content_hash, diff_snapshots,
    split_for, split_records,
)
from .augmentation import (
    Compose, HorizontalFlip, RandomCrop, BrightnessContrast, GaussianNoise, GaussianBlur, Cutout,
)
from .training import (
    Model, SoftmaxRegression, MLPClassifier, Trainer, TrainerConfig, Callback, History,
    EarlyStopping, ModelCheckpoint, save_checkpoint, load_checkpoint,
)
from .tuning import (
    grid_search, random_search, successive_halving, Uniform, LogUniform, IntUniform, Choice, TuningResult,
)
from .metrics import (
    accuracy, confusion_matrix, precision_recall_f1, iou_matrix, average_precision,
    mean_average_precision, lane_point_metrics, evaluate_classifier,
)
from .benchmark import BenchmarkConfig, BenchmarkResult, benchmark, model_size
from .drift import DriftConfig, DriftDetector, DriftReport, psi, ks_2samp, embedding_mean_shift
from .monitoring import MonitorConfig, OnlineMonitor, PredictionEvent, MonitorNode

__all__ = [n for n in dir() if not n.startswith("_")]
