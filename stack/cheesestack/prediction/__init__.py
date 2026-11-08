"""Prediction layer: future motion of other agents and the ego."""

from .motion_models import (
    MotionModel, MotionModelConfig, MotionModelType, PredictedTrajectory,
    EgoPredictor, EgoPredictorConfig,
)
from .object_prediction import (
    HypothesisSpec, ObjectPredictor, ObjectPredictorConfig, ObjectPrediction, tracked_object_state,
)
from .lane_change import (
    Lane, LaneProjection, project_to_lane, LaneChangeFeatures, extract_lane_change_features,
    LaneChangeModel, LogisticLaneChangeModel, LogisticLaneChangeConfig,
    LaneChangePredictor, LaneChangePredictorConfig, LaneChangePrediction,
)
from .collision import (
    OrientedBox, boxes_overlap, CollisionChecker, CollisionCheckerConfig, CollisionResult,
)
from .ttc import (
    ttc_longitudinal, ttc_constant_accel, closest_point_of_approach, ttc_2d_circles,
    TTCEstimator, TTCEstimatorConfig, TTCResult,
)
from .risk import RiskScorer, RiskScorerConfig, RiskAssessment, uncertainty_from_cov
from .behavior import (
    Intent, BehaviorFeatures, features_from_history, IntentModel, GaussianIntentModel,
    GaussianIntentConfig, BehaviorForecaster, BehaviorForecasterConfig, IntentForecast,
)

__all__ = [
    "MotionModel", "MotionModelConfig", "MotionModelType", "PredictedTrajectory",
    "EgoPredictor", "EgoPredictorConfig",
    "HypothesisSpec", "ObjectPredictor", "ObjectPredictorConfig", "ObjectPrediction", "tracked_object_state",
    "Lane", "LaneProjection", "project_to_lane", "LaneChangeFeatures", "extract_lane_change_features",
    "LaneChangeModel", "LogisticLaneChangeModel", "LogisticLaneChangeConfig",
    "LaneChangePredictor", "LaneChangePredictorConfig", "LaneChangePrediction",
    "OrientedBox", "boxes_overlap", "CollisionChecker", "CollisionCheckerConfig", "CollisionResult",
    "ttc_longitudinal", "ttc_constant_accel", "closest_point_of_approach", "ttc_2d_circles",
    "TTCEstimator", "TTCEstimatorConfig", "TTCResult",
    "RiskScorer", "RiskScorerConfig", "RiskAssessment", "uncertainty_from_cov",
    "Intent", "BehaviorFeatures", "features_from_history", "IntentModel", "GaussianIntentModel",
    "GaussianIntentConfig", "BehaviorForecaster", "BehaviorForecasterConfig", "IntentForecast",
]
