"""Unit tests for core conversion helpers."""

from __future__ import annotations

import pandas as pd

from Source.config import WeppcliffConfig
from Source.data_model import WeppcliffData
from Source.units import convert_units


def test_metric_precip_exception_converts_inches_to_mm() -> None:
    """-u m PRECIP converts exception variable from English to metric."""
    cfg = WeppcliffConfig(
        u="m",
        args=["-u", "m", "PRECIP"],
        u_loc=0,
    )
    data = WeppcliffData()
    data[1] = pd.DataFrame({"DT_1": ["2000-01-01 00:00:00"], "PRECIP": [1.0], "DUR": [60.0]})
    data[2] = pd.DataFrame({"DT_2": [], "AIR_TEMP": [], "REL_HUM": []})
    data[3] = pd.DataFrame({
        "DT_3": ["2000-01-01"],
        "MIN_TEMP": [20.0],
        "MAX_TEMP": [30.0],
        "SO_RAD": [200.0],
        "W_VEL": [1.0],
        "W_DIR": [0.0],
        "DP_TEMP": [15.0],
        "DUR": [1440.0],
    })
    data[4] = {"STATION": "T", "LAT": "0", "LON": "0", "ELEV": "0"}
    out = convert_units(data, cfg)
    assert abs(out.slot_df(1)["PRECIP"].iloc[0] - 25.4) < 0.01