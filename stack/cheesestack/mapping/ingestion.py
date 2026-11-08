"""Map ingestion: a simple JSON lane-map format and a minimal OSM-XML node/way parser.

JSON format ("cheesestack-map", version 1)::

    {
      "format": "cheesestack-map", "version": 1, "name": "...", "frame": "map",
      "origin": {"lat": 37.0, "lon": -122.0},          # optional geodetic anchor
      "lanes": [
        {"id": "a", "centerline": [[x, y], ...], "width": 3.5, "speed_limit": 13.89,
         "successors": ["b"], "left_neighbor": null, "right_neighbor": null,
         "left_boundary": "dashed", "right_boundary": "solid",   # optional types
         "road_id": "r1", "attributes": {}}
      ]
    }

Coordinates are metres in the local `map` frame. Predecessors are derived.

OSM: `parse_osm_xml` reads <node id lat lon> and <way> with <nd ref> and tags
`highway`, `lanes`, `oneway`, `maxspeed`, `width`. Each way becomes per-direction
lanes offset from the way centerline (right-hand traffic) and ways sharing
endpoint nodes are linked.
"""

from __future__ import annotations

import json
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from .boundaries import offset_polyline
from .topology import BoundaryType, Lane, LaneBoundary, LaneMap, Polyline

FORMAT_NAME = "cheesestack-map"
FORMAT_VERSION = 1
DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLE_MAP_PATH = DATA_DIR / "sample_map.json"

EARTH_RADIUS_M = 6378137.0


class MapFormatError(ValueError):
    pass


# --------------------------------------------------------------------- JSON


def lane_map_from_dict(doc: dict) -> LaneMap:
    if doc.get("format") != FORMAT_NAME:
        raise MapFormatError(f"not a {FORMAT_NAME} document")
    if int(doc.get("version", 0)) > FORMAT_VERSION:
        raise MapFormatError(f"unsupported version {doc.get('version')}")
    lanes = []
    for i, ld in enumerate(doc.get("lanes", [])):
        try:
            cl = Polyline(ld["centerline"])
            lane = Lane(
                id=str(ld["id"]),
                centerline=cl,
                width=float(ld.get("width", 3.5)),
                speed_limit=float(ld.get("speed_limit", 13.89)),
                successors=[str(s) for s in ld.get("successors", [])],
                left_neighbor=ld.get("left_neighbor"),
                right_neighbor=ld.get("right_neighbor"),
                road_id=str(ld.get("road_id", "")),
                attributes=dict(ld.get("attributes", {})),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise MapFormatError(f"lane #{i}: {e}") from e
        for side, sign in (("left", 1.0), ("right", -1.0)):
            bt = ld.get(f"{side}_boundary")
            if bt is None:
                continue
            if isinstance(bt, dict):
                btype = BoundaryType(bt.get("type", "solid"))
                pts = bt.get("points") or offset_polyline(cl.points, sign * lane.width / 2)
            else:
                btype = BoundaryType(bt)
                pts = offset_polyline(cl.points, sign * lane.width / 2)
            setattr(lane, f"{side}_boundary", LaneBoundary(Polyline(pts), btype))
        lanes.append(lane)
    lm = LaneMap(lanes, name=str(doc.get("name", "")))
    unknown = [f"{l.id}->{s}" for l in lm for s in l.successors if s not in lm]
    if unknown:
        raise MapFormatError(f"dangling successor references: {unknown}")
    return lm


def lane_map_to_dict(lane_map: LaneMap, origin: Optional[dict] = None, ndigits: int = 4) -> dict:
    lanes = []
    for lane in lane_map:
        d: dict[str, Any] = {
            "id": lane.id,
            "centerline": lane.centerline.to_list(ndigits),
            "width": lane.width,
            "speed_limit": lane.speed_limit,
            "successors": list(lane.successors),
            "left_neighbor": lane.left_neighbor,
            "right_neighbor": lane.right_neighbor,
            "road_id": lane.road_id,
        }
        if lane.attributes:
            d["attributes"] = lane.attributes
        for side in ("left", "right"):
            b = getattr(lane, f"{side}_boundary")
            if b is not None:
                d[f"{side}_boundary"] = {"type": b.type.value, "points": b.polyline.to_list(ndigits)}
        lanes.append(d)
    doc: dict[str, Any] = {"format": FORMAT_NAME, "version": FORMAT_VERSION, "name": lane_map.name,
                           "frame": "map", "lanes": lanes}
    if origin:
        doc["origin"] = origin
    return doc


def load_map_json(source: Union[str, os.PathLike, dict]) -> LaneMap:
    """Load from a path, a JSON string, or an already-parsed dict."""
    if isinstance(source, dict):
        return lane_map_from_dict(source)
    s = str(source)
    if s.lstrip().startswith("{"):
        return lane_map_from_dict(json.loads(s))
    with open(s, "r", encoding="utf-8") as f:
        return lane_map_from_dict(json.load(f))


def save_map_json(lane_map: LaneMap, path: Union[str, os.PathLike], origin: Optional[dict] = None) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lane_map_to_dict(lane_map, origin), f, indent=1)


def load_sample_map() -> LaneMap:
    """Small demo map shipped in cheesestack/mapping/data/sample_map.json."""
    return load_map_json(SAMPLE_MAP_PATH)


# ---------------------------------------------------------------------- OSM


def latlon_to_local(lat, lon, lat0: float, lon0: float) -> np.ndarray:
    """Equirectangular projection to local ENU metres (x east, y north).
    Fine for a few km. # TODO(mapping): proper ENU via ECEF / UTM for larger areas."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    x = np.radians(lon - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_RADIUS_M
    return np.stack([x, y], axis=-1)


@dataclass
class OSMConfig:
    default_lane_width_m: float = 3.5
    default_speed_mps: float = 13.89
    default_lanes_two_way: int = 2
    default_lanes_one_way: int = 1
    highway_types: tuple[str, ...] = ("motorway", "trunk", "primary", "secondary", "tertiary",
                                      "unclassified", "residential", "service", "living_street",
                                      "motorway_link", "trunk_link", "primary_link", "secondary_link",
                                      "tertiary_link")
    origin: Optional[tuple[float, float]] = None  # (lat, lon); default = mean of nodes


def parse_maxspeed(value: Optional[str], default: float) -> float:
    """'50' -> km/h, '30 mph' -> mph, '13 m/s' -> m/s. Returns m/s."""
    if not value:
        return default
    m = re.match(r"\s*([\d.]+)\s*(mph|km/h|kmh|kph|m/s)?", value)
    if not m:
        return default
    v = float(m.group(1))
    unit = m.group(2) or "km/h"
    if unit == "mph":
        return v * 0.44704
    if unit == "m/s":
        return v
    return v / 3.6


@dataclass
class _WayLanes:
    first_node: str
    last_node: str
    forward: list[str] = field(default_factory=list)  # index 0 = leftmost (innermost)
    backward: list[str] = field(default_factory=list)


def parse_osm_xml(xml_text: str, config: Optional[OSMConfig] = None) -> tuple[LaneMap, tuple[float, float]]:
    """Parse OSM XML into a LaneMap. Returns (lane_map, (lat0, lon0) origin).

    Lane layout per way (right-hand traffic): two-way roads split `lanes` into
    forward/backward halves (each >= 1) on the right/left of the way; one-way
    roads center their lanes on the way. Connectivity: at a shared end node, every
    lane arriving there links to the incoming lanes of other ways, matched by
    lane index (clamped).
    # TODO(mapping): turn:lanes, lanes:forward/backward tags, relations
    # (restrictions), intersection lane synthesis, mid-way node junctions.
    """
    cfg = config or OSMConfig()
    root = ET.fromstring(xml_text)
    nodes: dict[str, tuple[float, float]] = {}
    for n in root.iter("node"):
        nodes[n.get("id")] = (float(n.get("lat")), float(n.get("lon")))
    if not nodes:
        raise MapFormatError("OSM document has no nodes")
    if cfg.origin is not None:
        lat0, lon0 = cfg.origin
    else:
        arr = np.array(list(nodes.values()))
        lat0, lon0 = float(arr[:, 0].mean()), float(arr[:, 1].mean())

    lane_map = LaneMap(name="osm")
    ways: list[_WayLanes] = []
    for w in root.iter("way"):
        tags = {t.get("k"): t.get("v") for t in w.findall("tag")}
        if tags.get("highway") not in cfg.highway_types:
            continue
        refs = [nd.get("ref") for nd in w.findall("nd") if nd.get("ref") in nodes]
        if len(refs) < 2:
            continue
        ll = np.array([nodes[r] for r in refs])
        xy = latlon_to_local(ll[:, 0], ll[:, 1], lat0, lon0)
        oneway = tags.get("oneway", "no") in ("yes", "true", "1", "-1")
        if tags.get("oneway") == "-1":
            xy, refs = xy[::-1], refs[::-1]
        try:
            n_lanes = int(tags.get("lanes", ""))
        except ValueError:
            n_lanes = cfg.default_lanes_one_way if oneway else cfg.default_lanes_two_way
        n_lanes = max(1, n_lanes)
        width = cfg.default_lane_width_m
        if "width" in tags:
            try:
                width = float(tags["width"]) / n_lanes
            except ValueError:
                pass
        speed = parse_maxspeed(tags.get("maxspeed"), cfg.default_speed_mps)
        wid = w.get("id")
        wl = _WayLanes(refs[0], refs[-1])
        if oneway:
            n_f, n_b = n_lanes, 0
            offsets_f = [((n_lanes - 1) / 2 - i) * width for i in range(n_f)]  # i=0 leftmost
        else:
            n_f = max(1, n_lanes // 2)
            n_b = max(1, n_lanes - n_f)
            offsets_f = [-(i + 0.5) * width for i in range(n_f)]
        for i, off in enumerate(offsets_f):
            lid = f"w{wid}_f{i}"
            lane_map.add_lane(Lane(lid, Polyline(offset_polyline(xy, off)), width, speed, road_id=f"w{wid}",
                                   attributes={"osm_way": wid, "highway": tags.get("highway")}))
            wl.forward.append(lid)
        rxy = xy[::-1]
        for i in range(n_b):
            lid = f"w{wid}_b{i}"
            lane_map.add_lane(Lane(lid, Polyline(offset_polyline(rxy, -(i + 0.5) * width)), width, speed,
                                   road_id=f"w{wid}", attributes={"osm_way": wid, "highway": tags.get("highway")}))
            wl.backward.append(lid)
        for group in (wl.forward, wl.backward):
            for a, b in zip(group[:-1], group[1:]):
                lane_map.set_neighbors(a, b)  # a is left of b
        ways.append(wl)

    # connectivity at shared end nodes
    arriving: dict[str, list[list[str]]] = {}
    departing: dict[str, list[list[str]]] = {}
    for wl in ways:
        if wl.forward:
            arriving.setdefault(wl.last_node, []).append(wl.forward)
            departing.setdefault(wl.first_node, []).append(wl.forward)
        if wl.backward:
            arriving.setdefault(wl.first_node, []).append(wl.backward)
            departing.setdefault(wl.last_node, []).append(wl.backward)
    for node_id, in_groups in arriving.items():
        for gin in in_groups:
            for gout in departing.get(node_id, []):
                if _is_u_turn(lane_map, gin, gout):
                    continue
                for i, lid in enumerate(gin):
                    lane_map.link(lid, gout[min(i, len(gout) - 1)])
    return lane_map, (lat0, lon0)


def _is_u_turn(lane_map: LaneMap, gin: list[str], gout: list[str]) -> bool:
    a, b = lane_map[gin[0]], lane_map[gout[0]]
    return a.road_id == b.road_id
