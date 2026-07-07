"""Unit tests for PHIGI L3 directed daily imputation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.config import WeppcliffConfig  # noqa: E402
from Source.phigi.l0_astronomy import clear_sky_series_langleys  # noqa: E402
from Source.phigi.l2_weather_types import fit_local_weather_types  # noqa: E402
from Source.phigi.l3_daily import (  # noqa: E402
    _circular_mean_degrees,
    _enforce_temperature_order,
    impute_daily_posterior,
    impute_daily_temperature,
)


def _daily_panel(n: int = 120, *, lat: float = 40.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n, freq="D")
    ymd = dates.strftime("%Y%m%d")
    r_cs = clear_sky_series_langleys(ymd.tolist(), lat_deg=lat, elev_m=100.0)
    tau = np.clip(0.5 + 0.1 * np.sin(2 * np.pi * dates.dayofyear / 365.25), 0.15, 0.9)
    so_rad = tau * r_cs
    tmax = 12.0 + 8.0 * np.sin(2 * np.pi * (dates.dayofyear - 30) / 365.25) + rng.normal(0, 1.5, n)
    tmin = tmax - rng.uniform(4.0, 10.0, n)
    w_vel = np.clip(2.0 + rng.normal(0, 0.6, n), 0.2, 12.0)
    w_dir = rng.uniform(0, 360, n)
    dp = tmin - rng.uniform(2.0, 8.0, n)
    months = dates.month.to_numpy()
    cluster = np.where(months <= 3, 11, np.where(months <= 6, 21, np.where(months <= 9, 31, 41)))
    return pd.DataFrame(
        {
            "YYYYMMDD": ymd,
            "CLUSTER": cluster,
            "SO_RAD": so_rad,
            "MAX_TEMP": tmax,
            "MIN_TEMP": tmin,
            "DP_TEMP": dp,
            "W_VEL": w_vel,
            "W_DIR": w_dir,
        }
    )


def test_circular_mean_degrees_wraps() -> None:
    mean = _circular_mean_degrees(np.array([350.0, 10.0]))
    assert abs(mean - 0.0) < 2.0 or abs(mean - 360.0) < 2.0


def test_enforce_temperature_order() -> None:
    tmin = np.array([5.0, 12.0])
    tmax = np.array([3.0, 15.0])
    out_min, out_max, n_bad = _enforce_temperature_order(tmin, tmax)
    assert n_bad == 1
    assert out_min[0] == out_max[0]
    assert out_min[1] < out_max[1]


def test_impute_daily_temperature_fills_gaps() -> None:
    panel = _daily_panel(90)
    panel.loc[10:20, ["MIN_TEMP", "MAX_TEMP"]] = np.nan
    config = WeppcliffConfig(gf="phigi", lat=40.0, kg="Dfb")
    out = impute_daily_temperature(
        panel.drop(columns=["YYYYMMDD"]),
        panel["YYYYMMDD"],
        lat_deg=40.0,
        elev_m=100.0,
        config=config,
    )
    assert out["MIN_TEMP"].iloc[10:21].notna().all()
    assert out["MAX_TEMP"].iloc[10:21].notna().all()
    assert (out["MIN_TEMP"] <= out["MAX_TEMP"]).all()


def test_temperature_uses_cluster_donors() -> None:
    panel = _daily_panel(60, seed=1)
    panel["CLUSTER"] = 31
    panel.loc[panel.index[:30], "CLUSTER"] = 30
    panel.loc[panel.index[:30], "MAX_TEMP"] += 8.0
    panel.loc[panel.index[:30], "MIN_TEMP"] += 8.0
    panel.loc[15, "MAX_TEMP"] = np.nan
    panel.loc[15, "MIN_TEMP"] = np.nan
    config = WeppcliffConfig(gf="phigi", lat=40.0, kg="Dfb")
    out_warm = impute_daily_temperature(
        panel.drop(columns=["YYYYMMDD"]),
        panel["YYYYMMDD"],
        lat_deg=40.0,
        elev_m=100.0,
        config=config,
    )
    panel_cold = panel.copy()
    panel_cold.loc[15, "CLUSTER"] = 31
    out_cold = impute_daily_temperature(
        panel_cold.drop(columns=["YYYYMMDD"]),
        panel_cold["YYYYMMDD"],
        lat_deg=40.0,
        elev_m=100.0,
        config=config,
    )
    assert out_warm["MAX_TEMP"].iloc[15] > out_cold["MAX_TEMP"].iloc[15]


def test_impute_daily_posterior_fills_moisture_wind() -> None:
    panel = _daily_panel(80, seed=2)
    model = fit_local_weather_types(
        panel.drop(columns=["YYYYMMDD"]),
        panel["YYYYMMDD"],
        lat_deg=40.0,
        elev_m=100.0,
    )
    panel.loc[5:15, ["DP_TEMP", "W_VEL", "W_DIR"]] = np.nan
    config = WeppcliffConfig(gf="phigi", lat=40.0, kg="Cfa")
    out = impute_daily_posterior(
        panel.drop(columns=["YYYYMMDD"]),
        panel["YYYYMMDD"],
        lat_deg=40.0,
        elev_m=100.0,
        config=config,
        weather_model=model,
    )
    assert out["DP_TEMP"].iloc[5:16].notna().all()
    assert out["W_VEL"].iloc[5:16].notna().all()
    assert out["W_DIR"].iloc[5:16].notna().all()