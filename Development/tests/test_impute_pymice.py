"""PyMICE-backed gap-fill smoke tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Source.config import WeppcliffConfig, set_config
from Source.impute import create_cluster_col, impute_by_cluster


@pytest.fixture
def pymice_config():
    cfg = WeppcliffConfig(pm="t", io=25, im="DEFAULT", qi="f", iv="f")
    set_config(cfg)
    return cfg


def test_create_cluster_col_uses_seasonal_or_finer(pymice_config):
    del pymice_config
    rng = np.random.default_rng(3)
    n = 400
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "YYYYMMDD": dates.strftime("%Y%m%d"),
        "MIN_TEMP": rng.normal(0, 10, n),
        "MAX_TEMP": rng.normal(15, 10, n),
    })
    out = create_cluster_col(df, "YYYYMMDD", "%Y%m%d")
    assert "CLUSTER" in out.columns
    assert out["CLUSTER"].nunique() >= 4


def test_create_cluster_col_preserves_subdaily_rows(pymice_config):
    del pymice_config
    rng = np.random.default_rng(5)
    dates = pd.date_range("2018-01-01", periods=400 * 4, freq="6h")
    ymd = dates.strftime("%Y%m%d")
    df = pd.DataFrame({
        "DT_1": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "DUR": rng.uniform(1, 10, len(dates)),
        "PRECIP": rng.uniform(0, 2, len(dates)),
    })
    out = create_cluster_col(df, "DT_1", "%Y-%m-%d %H:%M:%S")
    assert len(out) == len(df)
    assert out["CLUSTER"].notna().all()


def test_impute_by_cluster_restores_missing_temps(pymice_config):
    del pymice_config
    rng = np.random.default_rng(4)
    n = 400
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "YYYYMMDD": dates.strftime("%Y%m%d"),
        "MIN_TEMP": rng.normal(0, 10, n),
        "MAX_TEMP": rng.normal(15, 10, n),
    })
    df.loc[50:80, "MIN_TEMP"] = np.nan
    clustered = create_cluster_col(df, "YYYYMMDD", "%Y%m%d")
    fil = clustered[["CLUSTER", "MIN_TEMP", "MAX_TEMP"]]
    out = impute_by_cluster(fil, "midastouch")
    assert out["MIN_TEMP"].notna().all()
    assert out["MAX_TEMP"].notna().all()