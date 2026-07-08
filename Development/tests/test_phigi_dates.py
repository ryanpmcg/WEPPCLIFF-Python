"""Tests for YYYYMMDD normalization on gap-fill paths."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.phigi.dates import as_yyyymmdd_strings, coerce_yyyymmdd_series  # noqa: E402
from Source.phigi.l4_precip_hurdle import compute_precip_constraints  # noqa: E402


def test_float_yyyymmdd_normalizes() -> None:
    values = [20190101.0, 20190102.0, np.nan]
    out = as_yyyymmdd_strings(values)
    assert out[0] == "20190101"
    assert out[1] == "20190102"
    assert out[2] == ""


def test_coerce_series_for_merge() -> None:
    series = pd.Series([20190101.0, 20190102.0])
    out = coerce_yyyymmdd_series(series)
    assert out.iloc[0] == "20190101"
    assert str(out.dtype) in {"object", "string", "str"}


def test_precip_constraints_accepts_float_dates() -> None:
    dates = as_yyyymmdd_strings([20190101.0, 20190102.0, 20190103.0])
    cluster = np.array([11.0, 11.0, 11.0])
    event = np.array([0.0, 1.0, 0.0])
    cons = compute_precip_constraints(dates, cluster, event)
    assert cons.force_dry.shape == (3,)