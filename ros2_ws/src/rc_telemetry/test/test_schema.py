import json

import pytest

from rc_telemetry.schema import Telemetry


def test_roundtrip_and_nulls():
    t = Telemetry(ts=1.0, state="ARMED", throttle=0.25, steering=-0.5, us_fl=1.2).validate()
    d = json.loads(t.to_json())
    assert d["state"] == "ARMED"
    assert d["us_fl"] == 1.2
    assert d["us_rr"] is None          # unset sensor stays null
    assert d["speed_mps"] is None      # pending hardware stays null, never 0


@pytest.mark.parametrize("kwargs", [
    {"steering": 2.0},
    {"throttle": -0.1},
    {"state": "DRIVE"},
    {"battery_pct": 150.0},
])
def test_rejects_out_of_range(kwargs):
    with pytest.raises(AssertionError):
        Telemetry(ts=1.0, **kwargs).validate()
