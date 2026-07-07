"""Unit tests for PHIGI L5 storm generator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.config import WeppcliffConfig  # noqa: E402
from Source.phigi.l5_storm_generator import (  # noqa: E402
    generate_parametric_day,
    impute_phigi_subdaily,
)


def _observed_wet_day(*, cluster: int = 21, ymd: str = "20190615") -> pd.DataFrame:
    base = pd.to_datetime(ymd, format="%Y%m%d") + pd.Timedelta(hours=14)
    return pd.DataFrame(
        {
            "DT_1": [base, base + pd.Timedelta(minutes=20)],
            "YYYYMMDD": [ymd, ymd],
            "CLUSTER": [cluster, cluster],
            "DUR": [12.0, 8.0],
            "PRECIP": [0.8, 0.4],
        }
    )


def test_generate_parametric_day_sums_to_daily() -> None:
    out = generate_parametric_day(
        yyyymmdd="20190701",
        cluster=21,
        bps=2,
        daily_precip=1.5,
        elen=2,
    )
    assert len(out) == 2
    assert out["PRECIP"].sum() == pytest.approx(1.5, rel=1e-6)


def test_impute_phigi_subdaily_uses_template_when_donors_exist() -> None:
    observed = _observed_wet_day()
    imp = observed.copy()
    imp.loc[0, ["DUR", "PRECIP", "HOUR", "MINUTE", "SECOND"]] = np.nan
    imp["HOUR"] = pd.to_datetime(imp["DT_1"]).dt.hour.astype(float)
    imp["MINUTE"] = pd.to_datetime(imp["DT_1"]).dt.minute.astype(float)
    imp["SECOND"] = pd.to_datetime(imp["DT_1"]).dt.second.astype(float)

    daily = pd.DataFrame(
        {
            "YYYYMMDD": ["20190615"],
            "CLUSTER": [21],
            "BPS": [2],
            "ELEN": [1],
            "PRECIP": [1.2],
        }
    )
    config = WeppcliffConfig(gf="phigi", pm="f")
    out, strategies, _, n_days = impute_phigi_subdaily(
        imp,
        observed_ts=observed,
        daily_events=daily,
        config=config,
    )
    assert n_days == 1
    assert out["PRECIP"].notna().all()
    assert any(s.startswith("storm_template_") or s.startswith("parametric_pulse_") for s in strategies)


def test_impute_phigi_subdaily_parametric_without_donors() -> None:
    imp = pd.DataFrame(
        {
            "DT_1": pd.to_datetime(["20190801 12:00:00", "20190801 12:30:00"]),
            "YYYYMMDD": ["20190801", "20190801"],
            "CLUSTER": [31, 31],
            "DUR": [np.nan, np.nan],
            "PRECIP": [np.nan, np.nan],
            "HOUR": [12.0, 12.0],
            "MINUTE": [0.0, 30.0],
            "SECOND": [0.0, 0.0],
        }
    )
    daily = pd.DataFrame(
        {
            "YYYYMMDD": ["20190801"],
            "CLUSTER": [31],
            "BPS": [2],
            "ELEN": [1],
            "PRECIP": [0.6],
        }
    )
    config = WeppcliffConfig(gf="phigi", pm="f")
    out, strategies, _, _ = impute_phigi_subdaily(
        imp,
        observed_ts=pd.DataFrame(columns=["DT_1", "YYYYMMDD", "CLUSTER", "DUR", "PRECIP"]),
        daily_events=daily,
        config=config,
    )
    assert out["PRECIP"].notna().all()
    assert out["PRECIP"].sum() == pytest.approx(0.6, rel=1e-4)
    assert any(s.startswith("parametric_pulse_") for s in strategies)


