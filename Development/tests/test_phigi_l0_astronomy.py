"""Unit tests for PHIGI L0 clear-sky radiation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.phigi.l0_astronomy import (  # noqa: E402
    clear_sky_radiation_langleys,
    clear_sky_series_langleys,
    day_of_year_from_yyyymmdd,
    extraterrestrial_radiation_mj_m2_day,
)


def test_day_of_year_jan_1() -> None:
    assert day_of_year_from_yyyymmdd("20200615") == 167


def test_extraterrestrial_midlatitude_summer_positive() -> None:
    ra = extraterrestrial_radiation_mj_m2_day(40.0, 172)
    assert 30.0 < ra < 45.0


def test_clear_sky_summer_exceeds_winter() -> None:
    summer = clear_sky_radiation_langleys(40.0, 172, elev_m=200.0)
    winter = clear_sky_radiation_langleys(40.0, 15, elev_m=200.0)
    assert summer > winter > 0.0


def test_clear_sky_series_length() -> None:
    dates = ["20200101", "20200701"]
    series = clear_sky_series_langleys(dates, lat_deg=35.0, elev_m=0.0)
    assert len(series) == 2
    assert np.all(series >= 0.0)
    assert series[1] > series[0]