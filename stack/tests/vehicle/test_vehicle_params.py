import json
import math

import pytest

from cheesestack.vehicle import DEFAULT_CATALOG_PATH, ParamsConfig, VehicleParams, load_catalog, load_vehicle_params

ENTRY = {
    "id": "test-car", "make": "Test", "model": "Car", "year": 2020, "bodyStyle": "sedan",
    "lengthMM": 4700, "widthMM": 1800, "heightMM": 1450, "wheelbaseMM": 2800,
    "frontTrackMM": 1560, "rearTrackMM": 1570, "curbWeightKG": 1420, "dragCoefficient": 0.3,
    "frontalAreaM2": 2.2, "engineType": "I4", "displacementL": 2.0, "horsepower": 180,
    "torqueNm": 250, "drivetrain": "FWD", "steeringRatio": 14.0, "maxSteerWheelDeg": 490, "tireRadiusM": 0.32,
}


def test_from_entry_units_and_geometry():
    p = VehicleParams.from_catalog_entry(ENTRY)
    assert p.wheelbase == pytest.approx(2.8)
    assert p.track_rear == pytest.approx(1.57)
    assert p.mass == pytest.approx(1420 + ParamsConfig().payload_kg)
    assert p.lf + p.lr == pytest.approx(p.wheelbase)
    assert p.lf < p.lr  # FWD is nose heavy
    assert p.max_road_wheel_angle == pytest.approx(math.radians(490 / 14))
    assert p.min_turn_radius == pytest.approx(2.8 / math.tan(math.radians(35)))
    assert p.understeer_gradient > 0
    assert p.handwheel_gain_deg_per_curvature() == pytest.approx(2.8 * 14 * 180 / math.pi)


def test_load_catalog_path_argument(tmp_path):
    f = tmp_path / "cars.json"
    f.write_text(json.dumps([ENTRY, dict(ENTRY, id="other", drivetrain="RWD")]))
    cat = load_catalog(f)
    assert set(cat) == {"test-car", "other"}
    assert cat["other"].front_weight_fraction < cat["test-car"].front_weight_fraction
    assert load_vehicle_params("other", f).drivetrain == "RWD"
    with pytest.raises(KeyError):
        load_vehicle_params("nope", f)


@pytest.mark.skipif(not DEFAULT_CATALOG_PATH.exists(), reason="app catalog not checked out")
def test_real_catalog_loads():
    cat = load_catalog()
    assert len(cat) > 0
    for p in cat.values():
        assert 2.0 < p.wheelbase < 4.5 and 800 < p.mass < 4000
        assert 2.0 < p.min_turn_radius < 12.0


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        VehicleParams(wheelbase=0.0)


def test_steady_state_yaw_rate_matches_kinematic_at_low_speed():
    p = VehicleParams()
    v, d = 1.0, 0.05
    assert p.steady_state_yaw_rate(v, d) == pytest.approx(v * d / p.wheelbase, rel=1e-3)
