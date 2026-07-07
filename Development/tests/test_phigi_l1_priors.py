"""Unit tests for PHIGI L1 Köppen prior store and multivariate sampling."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.config import WeppcliffConfig  # noqa: E402
from Source.phigi.l1_priors import KoppenPriorStore, load_default_priors  # noqa: E402
from Source.phigi.l3_tau import impute_daily_solar_radiation  # noqa: E402


def test_v2_prior_store_loads_bundled() -> None:
    store = load_default_priors()
    assert store.version == 2
    assert "Dfb" in store.zones
    assert "TAU" in store.variables


def test_dfb_tau_prior_seasonal_variation() -> None:
    store = load_default_priors()
    winter = store.tau_prior("Dfb", 1)
    summer = store.tau_prior("Dfb", 3)
    assert winter is not None and summer is not None
    assert summer.mean > winter.mean


def test_resolve_zone_kmqe() -> None:
    store = load_default_priors()
    zone = store.resolve_zone("AUTO", 42.2167, -71.1167)
    assert zone == "Dfb"


def test_multivariate_draw_tau_bounds() -> None:
    store = load_default_priors()
    rng = np.random.default_rng(0)
    for _ in range(200):
        draw = store.sample_multivariate("Dfb", 2, rng=rng)
        assert 0.02 <= draw["TAU"] <= 1.15


def test_multivariate_draw_temperature_order() -> None:
    store = load_default_priors()
    rng = np.random.default_rng(1)
    ok = 0
    n = 0
    for zone in store.zones:
        for season in (1, 2, 3, 4):
            for _ in range(250):
                draw = store.sample_multivariate(zone, season, rng=rng)
                n += 1
                if draw["MIN_TEMP"] <= draw["MAX_TEMP"]:
                    ok += 1
    assert ok == n


def test_koppen_fallback_when_no_local_tau() -> None:
    """All-missing SO_RAD should impute via Köppen τ prior (no cluster donors)."""
    dates = pd.date_range("2019-01-15", periods=5, freq="D")
    panel = pd.DataFrame(
        {
            "YYYYMMDD": dates.strftime("%Y%m%d"),
            "CLUSTER": [99] * 5,
            "SO_RAD": [np.nan] * 5,
        }
    )
    config = WeppcliffConfig(gf="phigi", tauqc="auto", lat=42.2167, lon=-71.1167, kg="AUTO")
    out = impute_daily_solar_radiation(
        panel[["CLUSTER", "SO_RAD"]].copy(),
        panel["YYYYMMDD"],
        lat_deg=42.2167,
        elev_m=199.0,
        config=config,
    )
    assert out["SO_RAD"].notna().all()
    assert (out["SO_RAD"] > 0).all()