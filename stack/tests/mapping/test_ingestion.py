import json

import numpy as np
import pytest

from cheesestack.mapping import (
    BoundaryType, MapFormatError, OSMConfig, SAMPLE_MAP_PATH, lane_map_to_dict, latlon_to_local,
    load_map_json, parse_osm_xml, save_map_json,
)
from cheesestack.mapping.ingestion import parse_maxspeed


def test_sample_map_loads(sample_map):
    assert SAMPLE_MAP_PATH.exists()
    assert len(sample_map) == 9
    assert sample_map["r1_1"].left_boundary.type == BoundaryType.DOUBLE_SOLID
    assert sample_map["r2_0"].speed_limit == pytest.approx(8.33)


def test_json_roundtrip(sample_map, tmp_path):
    p = tmp_path / "m.json"
    save_map_json(sample_map, p, origin={"lat": 1.0, "lon": 2.0})
    m2 = load_map_json(p)
    assert set(m2.ids()) == set(sample_map.ids())
    for lid in m2.ids():
        np.testing.assert_allclose(m2[lid].centerline.points, sample_map[lid].centerline.points, atol=1e-3)
        assert m2[lid].successors == sample_map[lid].successors
        assert m2[lid].left_boundary.type == sample_map[lid].left_boundary.type
    m3 = load_map_json(json.dumps(lane_map_to_dict(sample_map)))
    assert len(m3) == len(sample_map)


def test_json_errors():
    with pytest.raises(MapFormatError):
        load_map_json({"format": "other"})
    with pytest.raises(MapFormatError):
        load_map_json({"format": "cheesestack-map", "lanes": [{"id": "a", "centerline": [[0, 0], [1, 0]], "successors": ["x"]}]})
    with pytest.raises(MapFormatError):
        load_map_json({"format": "cheesestack-map", "lanes": [{"id": "a"}]})


OSM = """<?xml version="1.0"?>
<osm version="0.6">
  <node id="1" lat="37.0000" lon="-122.0000"/>
  <node id="2" lat="37.0000" lon="-121.9990"/>
  <node id="3" lat="37.0009" lon="-121.9990"/>
  <node id="4" lat="37.0018" lon="-121.9990"/>
  <way id="10"><nd ref="1"/><nd ref="2"/><tag k="highway" v="residential"/><tag k="maxspeed" v="30 mph"/></way>
  <way id="11"><nd ref="2"/><nd ref="3"/><nd ref="4"/><tag k="highway" v="primary"/><tag k="oneway" v="yes"/><tag k="lanes" v="2"/></way>
  <way id="12"><nd ref="1"/><nd ref="3"/><tag k="building" v="yes"/></way>
</osm>"""


def test_osm_parse():
    m, (lat0, lon0) = parse_osm_xml(OSM, OSMConfig(origin=(37.0, -122.0)))
    assert lat0 == 37.0
    assert set(m.ids()) == {"w10_f0", "w10_b0", "w11_f0", "w11_f1"}
    f = m["w10_f0"]
    assert f.speed_limit == pytest.approx(30 * 0.44704)
    # forward lane of an eastbound two-way road sits right (south) of the way
    assert f.centerline.points[:, 1].mean() == pytest.approx(-1.75, abs=1e-6)
    assert f.centerline.length == pytest.approx(88.8, rel=0.01)
    assert m["w11_f0"].right_neighbor == "w11_f1" and m["w11_f1"].left_neighbor == "w11_f0"
    assert set(f.successors) == {"w11_f0"}
    assert m["w10_b0"].successors == []  # nothing arrives into the backward lane's start besides itself
    assert "w10_f0" in m["w11_f0"].predecessors


def test_latlon_and_maxspeed():
    xy = latlon_to_local(37.001, -122.0, 37.0, -122.0)
    assert xy[0] == pytest.approx(0) and xy[1] == pytest.approx(111.3, rel=0.01)
    assert parse_maxspeed("50", 1.0) == pytest.approx(50 / 3.6)
    assert parse_maxspeed("10 m/s", 1.0) == 10.0
    assert parse_maxspeed(None, 7.0) == 7.0
    assert parse_maxspeed("signals", 7.0) == 7.0
