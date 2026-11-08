"""Mapping layer: roadway structure."""

from .topology import (
    BoundaryType, Lane, LaneBoundary, LaneMap, LaneMapConfig, Polyline, Projection,
)
from .road_graph import Edge, EdgeType, PathResult, RoadGraph, RoadGraphConfig
from .ingestion import (
    MapFormatError, OSMConfig, SAMPLE_MAP_PATH, lane_map_from_dict, lane_map_to_dict,
    latlon_to_local, load_map_json, load_sample_map, parse_osm_xml, save_map_json,
)
from .map_matching import HMMMapMatcher, HMMMatcherConfig, MapMatcher, MapMatcherConfig, MatchResult
from .route import Route, RouteConfig, RouteSegment
from .curvature import (
    ClothoidSegment, CubicSpline1D, CubicSpline2D, CurvatureConfig, curvature_profile,
    discrete_curvature, fit_clothoid_g1, fit_linear_curvature, heading_rate_curvature,
    smooth_polyline, smooth_signal,
)
from .boundaries import (
    BoundaryConfig, Corridor, build_corridor, corridor_from_route, generate_lane_boundaries,
    lane_boundary_polylines, offset_polyline, point_in_polygon, polygon_area,
)
from .dynamic_updates import ChangeRecord, MapOverlay, MapUpdate, OverlayConfig, UpdateKind

__all__ = [
    "BoundaryType", "Lane", "LaneBoundary", "LaneMap", "LaneMapConfig", "Polyline", "Projection",
    "Edge", "EdgeType", "PathResult", "RoadGraph", "RoadGraphConfig",
    "MapFormatError", "OSMConfig", "SAMPLE_MAP_PATH", "lane_map_from_dict", "lane_map_to_dict",
    "latlon_to_local", "load_map_json", "load_sample_map", "parse_osm_xml", "save_map_json",
    "HMMMapMatcher", "HMMMatcherConfig", "MapMatcher", "MapMatcherConfig", "MatchResult",
    "Route", "RouteConfig", "RouteSegment",
    "ClothoidSegment", "CubicSpline1D", "CubicSpline2D", "CurvatureConfig", "curvature_profile",
    "discrete_curvature", "fit_clothoid_g1", "fit_linear_curvature", "heading_rate_curvature",
    "smooth_polyline", "smooth_signal",
    "BoundaryConfig", "Corridor", "build_corridor", "corridor_from_route", "generate_lane_boundaries",
    "lane_boundary_polylines", "offset_polyline", "point_in_polygon", "polygon_area",
    "ChangeRecord", "MapOverlay", "MapUpdate", "OverlayConfig", "UpdateKind",
]
