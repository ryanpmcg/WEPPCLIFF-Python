"""Unit tests for PHIGI τ decomposition and solar imputation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.config import WeppcliffConfig  # noqa: E402
from Source.phigi.l0_astronomy import clear_sky_series_langleys  # noqa: E402
from Source.phigi.l3_tau import (  # noqa: E402
    compute_tau,
    impute_daily_solar_radiation,
    resolve_tau_qc_bounds,
    tau_qc_mask,
)
from Source.phigi.diagnose import mask_sorad_holdout, sorad_rmse  # noqa: E402


def _synthetic_panel(n: int = 120, *, lat: float = 40.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n, freq="D")
    ymd = dates.strftime("%Y%m%d")
    r_cs = clear_sky_series_langleys(ymd.tolist(), lat_deg=lat, elev_m=100.0)
    tau = np.clip(0.45 + 0.15 * np.sin(2 * np.pi * dates.dayofyear / 365.25), 0.1, 0.95)
    tau += rng.normal(0, 0.03, n)
    so_rad = tau * r_cs
    return pd.DataFrame(
        {
            "YYYYMMDD": ymd,
            "CLUSTER": dates.month,
            "SO_RAD": so_rad,
        }
    )


def test_compute_tau_roundtrip() -> None:
    panel = _synthetic_panel(30)
    r_cs = clear_sky_series_langleys(panel["YYYYMMDD"], lat_deg=40.0, elev_m=0.0)
    tau = compute_tau(panel["SO_RAD"].to_numpy(), r_cs)
    assert np.nanmean(np.abs(tau - 0.5)) < 0.25


def test_tau_qc_flags_extreme() -> None:
    tau = np.array([0.01, 0.5, 1.4, np.nan])
    lo, hi = resolve_tau_qc_bounds("auto")
    bad = tau_qc_mask(tau, lo=lo, hi=hi)
    assert bool(bad[0]) and not bool(bad[1]) and bool(bad[2])


def test_impute_daily_solar_fills_missing() -> None:
    panel = _synthetic_panel(90)
    panel.loc[10:20, "SO_RAD"] = np.nan
    dly = panel[["CLUSTER", "SO_RAD"]].copy()
    config = WeppcliffConfig(gf="phigi", tauqc="auto", lat=40.0)
    out = impute_daily_solar_radiation(
        dly,
        panel["YYYYMMDD"],
        lat_deg=40.0,
        elev_m=100.0,
        config=config,
    )
    assert out["SO_RAD"].iloc[10:21].notna().all()


def test_holdout_rmse_beats_naive_mean() -> None:
    panel = _synthetic_panel(200)
    masked, holdout = mask_sorad_holdout(panel["SO_RAD"], fraction=0.2, seed=1)
    dly = panel[["CLUSTER", "SO_RAD"]].copy()
    dly["SO_RAD"] = masked
    truth = panel["SO_RAD"]

    config = WeppcliffConfig(gf="phigi", tauqc="auto", lat=40.0)
    phigi_out = impute_daily_solar_radiation(
        dly.copy(),
        panel["YYYYMMDD"],
        lat_deg=40.0,
        elev_m=100.0,
        config=config,
    )

    naive = dly.copy()
    naive["SO_RAD"] = naive["SO_RAD"].fillna(naive["SO_RAD"].mean())

    rmse_phigi = sorad_rmse(truth, phigi_out["SO_RAD"], holdout)
    rmse_naive = sorad_rmse(truth, naive["SO_RAD"], holdout)
    assert rmse_phigi < rmse_naive


def test_tau_impute_uses_weather_type_cluster_donors() -> None:
    """τ fill should prefer donors in the same CLUSTER stratum."""
    panel = _synthetic_panel(60, seed=2)
    r_cs = clear_sky_series_langleys(panel["YYYYMMDD"], lat_deg=40.0, elev_m=100.0)
    panel["CLUSTER"] = 31
    panel.loc[panel.index[:30], "CLUSTER"] = 30
    panel.loc[panel.index[30:], "SO_RAD"] *= 0.55
    panel.loc[10, "SO_RAD"] = np.nan
    dly = panel[["CLUSTER", "SO_RAD"]].copy()
    config = WeppcliffConfig(gf="phigi", tauqc="auto", lat=40.0, kg="Dfb")
    out_hi = impute_daily_solar_radiation(
        dly.copy(),
        panel["YYYYMMDD"],
        lat_deg=40.0,
        elev_m=100.0,
        config=config,
    )
    dly_low = dly.copy()
    dly_low.loc[10, "CLUSTER"] = 31
    out_low = impute_daily_solar_radiation(
        dly_low,
        panel["YYYYMMDD"],
        lat_deg=40.0,
        elev_m=100.0,
        config=config,
    )
    assert out_hi["SO_RAD"].iloc[10] > out_low["SO_RAD"].iloc[10]


def test_corrupt_sorad_invalidation_and_recovery() -> None:
    panel = _synthetic_panel(60)
    panel.loc[30:40, "SO_RAD"] *= 0.1
    dly = panel[["CLUSTER", "SO_RAD"]].copy()
    config = WeppcliffConfig(gf="phigi", tauqc="auto", lat=40.0)
    out = impute_daily_solar_radiation(
        dly,
        panel["YYYYMMDD"],
        lat_deg=40.0,
        elev_m=100.0,
        config=config,
    )
    truth = _synthetic_panel(60)["SO_RAD"]
    err = np.abs(out["SO_RAD"].iloc[30:41] - truth.iloc[30:41]).mean()
    raw_err = np.abs(dly["SO_RAD"].iloc[30:41] - truth.iloc[30:41]).mean()
    assert err < raw_err