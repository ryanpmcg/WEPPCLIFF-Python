"""Storm-template sub-daily imputation smoke test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from Source.config import WeppcliffConfig, set_config


@pytest.fixture
def storm_config():
    cfg = WeppcliffConfig(pm="t", st="t", io=20, verb="f")
    set_config(cfg)
    return cfg


def test_storm_template_module_importable():
    from Source.weppcliff_impute import impute_subdaily_timeseries
    from Source.weppcliff_impute.overlap import resolve_overlaps

    assert callable(impute_subdaily_timeseries)
    assert callable(resolve_overlaps)


def test_resolve_overlaps_on_synthetic(storm_config):
    del storm_config
    from Source.weppcliff_impute.overlap import resolve_overlaps

    base = pd.to_datetime("20200601")
    df = pd.DataFrame({
        "DT_1": [base, base + pd.Timedelta(minutes=1)],
        "YYYYMMDD": ["20200601", "20200601"],
        "DUR": [8.0, 4.0],
        "PRECIP": [1.5, 2.0],
    })
    out = resolve_overlaps(df)
    assert len(out) == 2
    assert out["PRECIP"].sum() == pytest.approx(3.5)