"""Unit tests for PHIGI L4 hurdle precipitation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.config import WeppcliffConfig  # noqa: E402
from Source.phigi.l4_precip_hurdle import (  # noqa: E402
    compute_precip_constraints,
    enforce_dry_days,
    impute_event_occurrence,
    impute_precip_hurdle,
    impute_wet_event_characteristics,
)


def _panel(n: int = 120, *, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n, freq="D")
    ymd = dates.strftime("%Y%m%d")
    cluster = np.where(dates.month <= 3, 11, np.where(dates.month <= 6, 21, 31))
    event = (rng.random(n) < 0.2).astype(float)
    so_rad = 300.0 + rng.normal(0, 20, n)
    so_rad[event == 0] *= 1.1
    var = pd.DataFrame(
        {
            "CLUSTER": cluster,
            "SO_RAD": so_rad,
            "MAX_TEMP": 15 + rng.normal(0, 3, n),
            "MIN_TEMP": 5 + rng.normal(0, 3, n),
            "DP_TEMP": 3 + rng.normal(0, 2, n),
            "W_VEL": 2 + rng.normal(0, 0.5, n),
            "W_DIR": rng.uniform(0, 360, n),
            "EVENT": event,
        }
    )
    pcp = pd.DataFrame(
        {
            "BPS": np.where(event == 1, rng.integers(1, 4, n), 0),
            "ELEN": np.where(event == 1, rng.integers(1, 3, n), 0),
            "PRECIP": np.where(event == 1, rng.uniform(0.1, 1.5, n), 0.0),
        }
    )
    return var, pcp, pd.Series(ymd)


def test_compute_precip_constraints_ignores_cluster_counts() -> None:
    """CLUSTER is a donor pool; gap detection must not use count limits 7 / 0.025."""
    var, _, ymd = _panel(90, seed=1)
    var.loc[var.index[:60], "CLUSTER"] = 99
    var.loc[var.index[:60], "EVENT"] = 0
    cons = compute_precip_constraints(
        ymd.to_numpy(),
        var["CLUSTER"].to_numpy(dtype=float),
        var["EVENT"].to_numpy(dtype=float),
    )
    assert cons.dry_clusters == set()


def test_impute_event_fills_missing() -> None:
    var, _, ymd = _panel(80, seed=2)
    var.loc[10:25, "EVENT"] = np.nan
    out = impute_event_occurrence(var, ymd, lat_deg=40.0, elev_m=100.0)
    assert out["EVENT"].iloc[10:26].notna().all()


def test_true_dry_months_force_zero_event() -> None:
    var, _, ymd = _panel(60, seed=3)
    var["CLUSTER"] = 5.0
    var["EVENT"] = np.nan
    cons = compute_precip_constraints(
        ymd.to_numpy(),
        var["CLUSTER"].to_numpy(dtype=float),
        np.zeros(60),
    )
    out = impute_event_occurrence(
        var,
        ymd,
        lat_deg=40.0,
        elev_m=100.0,
        constraints=cons,
        rng=np.random.default_rng(0),
    )
    assert (out["EVENT"] == 0).all()


def test_enforce_dry_days_zeros_precip() -> None:
    var, pcp, _ = _panel(40, seed=4)
    var.loc[var.index[5:15], "EVENT"] = 0
    pcp.loc[pcp.index[5:15], ["PRECIP", "BPS", "ELEN"]] = np.nan
    out = enforce_dry_days(var, pcp)
    assert (out["PRECIP"].iloc[5:15] == 0).all()
    assert (out["BPS"].iloc[5:15] == 0).all()


def test_impute_wet_event_characteristics_fills_bps() -> None:
    eve = pd.DataFrame(
        {
            "YYYYMMDD": ["20190101", "20190102", "20190103"],
            "CLUSTER": [21, 21, 31],
            "BPS": [2.0, np.nan, np.nan],
            "ELEN": [1.0, np.nan, np.nan],
            "PRECIP": [0.5, np.nan, np.nan],
        }
    )
    out = impute_wet_event_characteristics(eve, lat_deg=40.0)
    assert out["BPS"].notna().all()
    assert out["PRECIP"].notna().all()
    assert out["BPS"].iloc[1] == 2.0


def test_impute_precip_hurdle_orchestrator() -> None:
    var, pcp, ymd = _panel(50, seed=5)
    var.loc[8:18, "EVENT"] = np.nan
    config = WeppcliffConfig(gf="phigi", lat=40.0)
    var_out, pcp_out = impute_precip_hurdle(
        var,
        pcp,
        ymd,
        lat_deg=40.0,
        elev_m=100.0,
        config=config,
    )
    assert var_out["EVENT"].iloc[8:19].notna().all()
    dry = var_out["EVENT"].to_numpy() == 0
    orig_wet = pcp["PRECIP"].to_numpy() > 0
    enforced = dry & ~orig_wet
    assert (pcp_out.loc[enforced, "PRECIP"].fillna(0) == 0).all()