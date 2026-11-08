# mapping

Lane-level road model in the `map` frame (ENU, metres, lateral + = left). Pure numpy/stdlib.

| Module | Capability | Works now | Key TODOs |
|---|---|---|---|
| `topology.py` | Lane topology | `Polyline` (arc length, interpolate, signed projection w/ window + extrapolation, resample, sub), `Lane`, `LaneBoundary`/`BoundaryType`, `LaneMap` (succ/pred/left/right, leftmost/rightmost, nearest lanes, `validate()`) | spatial index, per-vertex width, opposing neighbors, stop lines/signals |
| `road_graph.py` | Road graph + routing | Directed lane graph (successor + boundary-aware lane-change edges), A*/Dijkstra via `heapq`, distance/time costs, blocked lanes, speed overrides, cost scaling | turn costs, traffic-aware costs, contraction hierarchies, mid-lane lane-change landing |
| `ingestion.py` | Map ingestion | `cheesestack-map` v1 JSON load/save/roundtrip, minimal OSM XML (nodes/ways, `lanes`/`oneway`/`maxspeed`/`width`) -> per-direction lanes + junction links, `load_sample_map()` (`data/sample_map.json`) | lanes:forward/backward, turn:lanes, restrictions, proper ENU/UTM, Lanelet2/OpenDRIVE |
| `map_matching.py` | Map matching | `MapMatcher` (distance + heading gating, Gaussian cost, softmax confidence w/ off-map hypothesis), `HMMMapMatcher` (Viterbi; route-vs-euclid transitions, lane-change penalty, chain breaks) | online fixed-lag, pose covariance, perception lane evidence |
| `route.py` | Route representation | `Route` from lane path (lane changes spliced at a fraction), Frenet `to_frenet`/`from_frenet`, `lane_at`, speed limit, remaining distance, distance to next lane change | smooth lane-change blends, corridor routes, re-route on overlay version |
| `curvature.py` | Curvature modeling | Menger + heading-rate curvature, profiles, Laplacian/moving-average smoothing, `ClothoidSegment` (Gauss-Legendre), G1 clothoid fit (Newton), linear-curvature LSQ, natural `CubicSpline1D/2D` | Taubin/QP smoothing, robust Bertolazzi-Frego G1, banded spline solve |
| `boundaries.py` | Road boundaries | Miter `offset_polyline`, lane boundary polylines, `generate_lane_boundaries` (dashed/solid/curb typing), `Corridor` polygons along lanes/routes (optionally full carriageway), point-in-polygon, area | self-intersection cleanup on tight curves, intersection polygons |
| `dynamic_updates.py` | Dynamic map updates | Versioned `MapOverlay`: closures (full/partial), speed limits (min wins), fused observed lane shifts (EMA, TTL refresh), lane-change blocks, expiry, change history, `graph_constraints()`, `materialize()` | fleet/V2X sync, persistence, source conflict resolution, polygon closures |

Tests: `python3 -m pytest tests/mapping`.
