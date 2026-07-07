"""Unit tests for PHIGI L2 local weather-type mixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.phigi.l0_astronomy import clear_sky_series_langleys  # noqa: E402
from Source.config import WeppcliffConfig  # noqa: E402
from Source.phigi.l2_weather_types import (  # noqa: E402
    assign_weather_types,
    fit_local_weather_types,
    weather_type_cluster_labels,
)
from Source.phigi.pipeline import (  # noqa: E402
    merge_phigi_clusters,
    phigi_wire_weather_type_clusters,
)


def _bimodal_summer_panel(n: int = 200, *, lat: float = 40.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-06-01", periods=n, freq="D")
    ymd = dates.strftime("%Y%m%d")
    r_cs = clear_sky_series_langleys(ymd.tolist(), lat_deg=lat, elev_m=100.0)
    comp = rng.integers(0, 2, n)
    tau = np.where(comp == 0, 0.35, 0.75) + rng.normal(0, 0.04, n)
    tau = np.clip(tau, 0.1, 0.95)
    tmax = np.where(comp == 0, 22.0, 30.0) + rng.normal(0, 1.5, n)
    tmin = tmax - np.where(comp == 0, 8.0, 12.0) + rng.normal(0, 0.8, n)
    return pd.DataFrame(
        {
            "YYYYMMDD": ymd,
            "SO_RAD": tau * r_cs,
            "MIN_TEMP": tmin,
            "MAX_TEMP": tmax,
            "W_VEL": np.where(comp == 0, 2.0, 4.0) + rng.normal(0, 0.3, n),
            "DP_TEMP": tmin - 2.0 + rng.normal(0, 1.0, n),
        }
    )


def test_fit_detects_multiple_components() -> None:
    panel = _bimodal_summer_panel(180)
    model = fit_local_weather_types(
        panel,
        panel["YYYYMMDD"],
        lat_deg=40.0,
        elev_m=100.0,
        min_n=40,
        max_components=4,
        rng=np.random.default_rng(1),
    )
    summer = model.seasons.get(3)
    assert summer is not None
    assert summer.n_components >= 2
    assert summer.n_fit >= 40


def test_assign_weather_types_covers_all_days() -> None:
    panel = _bimodal_summer_panel(120)
    model = fit_local_weather_types(
        panel,
        panel["YYYYMMDD"],
        lat_deg=40.0,
        elev_m=100.0,
        min_n=30,
        rng=np.random.default_rng(2),
    )
    panel.loc[10:15, "MIN_TEMP"] = np.nan
    comp = assign_weather_types(panel, panel["YYYYMMDD"], model, lat_deg=40.0, elev_m=100.0)
    assert len(comp) == len(panel)
    assert (comp >= 0).all()


def test_wire_weather_type_clusters_replaces_calendar_cluster() -> None:
    panel = _bimodal_summer_panel(120)
    config = WeppcliffConfig(gf="phigi", lat=40.0)
    model = fit_local_weather_types(
        panel,
        panel["YYYYMMDD"],
        lat_deg=40.0,
        elev_m=100.0,
        min_n=30,
        rng=np.random.default_rng(4),
    )
    components = assign_weather_types(panel, panel["YYYYMMDD"], model, lat_deg=40.0, elev_m=100.0)
    config.phigi_weather_type_labels = components

    dly = panel[["SO_RAD", "MIN_TEMP", "MAX_TEMP", "W_VEL", "DP_TEMP"]].copy()
    dly.insert(0, "CLUSTER", 3)
    wired = phigi_wire_weather_type_clusters(
        dly,
        panel["YYYYMMDD"],
        config=config,
    )
    assert wired["CLUSTER"].nunique() > 1
    assert config.impute_granularity == "weather_type"
    assert not config.phigi_cluster_lut.empty

    merged = merge_phigi_clusters(
        panel[["YYYYMMDD", "MIN_TEMP"]],
        config=config,
    )
    assert "CLUSTER" in merged.columns
    assert merged["CLUSTER"].notna().all()


def test_weather_type_cluster_labels_unique_by_season() -> None:
    dates = pd.date_range("2019-01-01", periods=365, freq="D").strftime("%Y%m%d")
    comp = np.zeros(365, dtype=int)
    labels = weather_type_cluster_labels(dates, comp, lat_deg=40.0)
    assert labels.min() >= 10
    assert len(np.unique(labels)) >= 4