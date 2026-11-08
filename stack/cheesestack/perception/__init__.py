"""Perception layer: sensors to an understanding of the world.

Front end: see README_frontend.md. Scene understanding: see README_scene.md.
"""
from .camera_ingest import CameraIngest, SyntheticSource
from .sensor_sync import ApproximateTimeSynchronizer
from .preprocessing import Preprocessor
from .calibration import PinholeCamera, InversePerspectiveMapper, calibrate_zhang
from .distortion import BrownConrady, UndistortMap
from .environment import EnvironmentEstimator, Condition
from .confidence import ConfidenceScorer
from .detection import BlobDetector, ModelDetector, nms
from .classification import FeatureClassifier, map_label
from .tracking import MultiObjectTracker, hungarian
from .lanes import ClassicalLaneDetector
from .road_edge import RoadEdgeDetector
from .drivable_space import RegionGrowingSegmenter
from .traffic_signs import ColorShapeSignRecognizer, SignVoter
from .traffic_lights import HSVTrafficLightClassifier, TrafficLightFilter
from .occupancy_grid import OccupancyGrid
from .fusion import DetectionFuser, SourceRegistry

__all__ = [
    "CameraIngest",
    "SyntheticSource",
    "ApproximateTimeSynchronizer",
    "Preprocessor",
    "PinholeCamera",
    "InversePerspectiveMapper",
    "calibrate_zhang",
    "BrownConrady",
    "UndistortMap",
    "EnvironmentEstimator",
    "Condition",
    "ConfidenceScorer",
    "BlobDetector",
    "ModelDetector",
    "nms",
    "FeatureClassifier",
    "map_label",
    "MultiObjectTracker",
    "hungarian",
    "ClassicalLaneDetector",
    "RoadEdgeDetector",
    "RegionGrowingSegmenter",
    "ColorShapeSignRecognizer",
    "SignVoter",
    "HSVTrafficLightClassifier",
    "TrafficLightFilter",
    "OccupancyGrid",
    "DetectionFuser",
    "SourceRegistry",
]
