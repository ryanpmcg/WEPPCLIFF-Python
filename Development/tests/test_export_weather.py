"""Tests for weather timeseries export with KE/EI aggregation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.config import WeppcliffConfig, set_config  # noqa: E402
from Source.data_model import WeppcliffData  # noqa: E402
from Source.export_weather import (  # noqa: E402
    OMITTED_STORMS_COL,
    aggregate_erosion_indices,
    annotate_storm_export_frame,
    build_weather_export_frame,
)
from Source.io import export_data  # noqa: E402


def _storm_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "DT": [
            "2000-01-15 12:00:00",
            "2000-01-20 08:00:00",
            "2000-02-10 09:00:00",
            "2001-01-05 10:00:00",
        ],
        "DEPTH": [12.7, 6.0, 20.0, 30.0],
        "R2_KE": [1.0, 2.0, 4.0, 8.0],
        "R2_EI": [10.0, 20.0, 40.0, 80.0],
    })


def test_monthly_erosion_sums_honor_pmd() -> None:
    storms = _storm_frame()
    storms.loc[1, "R2_KE"] = np.nan

    cfg_ignore = WeppcliffConfig(pmd="f")
    monthly_ignore = aggregate_erosion_indices(storms, cfg_ignore, "monthly_ts")
    jan_ignore = monthly_ignore.loc[monthly_ignore["YYYYMM"] == "200001", "R2_KE"].iloc[0]
    assert jan_ignore == pytest.approx(1.0)
    jan_ei = monthly_ignore.loc[monthly_ignore["YYYYMM"] == "200001", "R2_EI"].iloc[0]
    assert jan_ei == pytest.approx(30.0)

    cfg_preserve = WeppcliffConfig(pmd="t")
    monthly_preserve = aggregate_erosion_indices(storms, cfg_preserve, "monthly_ts")
    jan_preserve = monthly_preserve.loc[monthly_preserve["YYYYMM"] == "200001", "R2_KE"].iloc[0]
    assert np.isnan(jan_preserve)


def test_monthly_mean_uses_mean_of_monthly_totals() -> None:
    storms = _storm_frame()
    config = WeppcliffConfig(pmd="f")
    clim = aggregate_erosion_indices(storms, config, "monthly_mean")
    jan = clim.loc[clim["MM"] == 1, "R2_KE"].iloc[0]
    assert jan == pytest.approx(5.5)  # mean of 3.0 (2000) and 8.0 (2001)


def test_annual_mean_is_mean_of_annual_totals() -> None:
    storms = _storm_frame()
    config = WeppcliffConfig(pmd="f")
    annual = aggregate_erosion_indices(storms, config, "annual_mean")
    assert annual.loc[0, "R2_KE"] == pytest.approx(7.5)
    assert annual.loc[0, "R2_EI"] == pytest.approx(75.0)


def test_daily_export_includes_ke_ei_columns() -> None:
    config = WeppcliffConfig(pmd="f", ei="t")
    daily = pd.DataFrame({
        "YYYYMMDD": ["20000115", "20000120", "20000210", "20010105"],
        "MAX_TEMP": [1.0, 2.0, 4.0, 8.0],
    })
    data = WeppcliffData({10: daily, 9: _storm_frame()})
    out = build_weather_export_frame(data, config, "daily_ts")
    assert "R2_KE" in out.columns
    assert "R2_EI" in out.columns
    assert out.loc[out["YYYYMMDD"] == "20000115", "R2_KE"].iloc[0] == pytest.approx(1.0)
    assert out.loc[out["YYYYMMDD"] == "20000120", "R2_KE"].iloc[0] == pytest.approx(2.0)


def test_omit_storms_excludes_shallow_events_from_ke_ei() -> None:
    storms = _storm_frame()
    daily = pd.DataFrame({
        "YYYYMMDD": ["20000115", "20000120", "20000210", "20010105"],
        "MAX_TEMP": [1.0, 2.0, 4.0, 8.0],
    })
    include_all = build_weather_export_frame(
        WeppcliffData({10: daily, 9: storms}),
        WeppcliffConfig(pmd="f", ei="t", os="f"),
        "daily_ts",
    )
    jan20_include = include_all.loc[include_all["YYYYMMDD"] == "20000120", "R2_KE"].iloc[0]
    assert jan20_include == pytest.approx(2.0)
    assert include_all[OMITTED_STORMS_COL].sum() == 0

    omit_shallow = build_weather_export_frame(
        WeppcliffData({10: daily, 9: storms}),
        WeppcliffConfig(pmd="f", ei="t", os="t", sod=0.5, u="m"),
        "daily_ts",
    )
    jan20_row = omit_shallow.loc[omit_shallow["YYYYMMDD"] == "20000120"].iloc[0]
    assert jan20_row[OMITTED_STORMS_COL] == 1
    assert jan20_row["R2_KE"] == pytest.approx(0.0)
    jan15 = omit_shallow.loc[omit_shallow["YYYYMMDD"] == "20000115", "R2_KE"].iloc[0]
    assert jan15 == pytest.approx(1.0)


def test_storm_export_marks_omitted_events() -> None:
    storms = _storm_frame()
    out = annotate_storm_export_frame(storms, WeppcliffConfig(os="t", sod=0.5, u="m"))
    assert out.loc[out["DT"] == "2000-01-20 08:00:00", OMITTED_STORMS_COL].iloc[0] == 1
    assert out.loc[out["DT"] == "2000-01-15 12:00:00", OMITTED_STORMS_COL].iloc[0] == 0


def test_build_export_frame_merges_weather_and_erosion() -> None:
    config = WeppcliffConfig(pmd="f", ei="t")
    data = WeppcliffData({
        11: pd.DataFrame({
            "YYYYMM": ["200001", "200002"],
            "MAX_TEMP": [5.0, 6.0],
            "PRECIP": [10.0, 20.0],
        }),
        9: _storm_frame(),
    })
    out = build_weather_export_frame(data, config, "monthly_ts")
    assert "R2_KE" in out.columns
    assert "MAX_TEMP" in out.columns
    assert out.loc[out["YYYYMM"] == "200001", "R2_KE"].iloc[0] == pytest.approx(3.0)


def test_biweekly_mean_uses_mean_of_biweekly_totals() -> None:
    storms = _storm_frame()
    config = WeppcliffConfig(pmd="f")
    clim = aggregate_erosion_indices(storms, config, "biweekly_mean")
    jan_first = clim.loc[(clim["MM"] == 1) & (clim["HALF"] == 1), "R2_KE"].iloc[0]
    jan_second = clim.loc[(clim["MM"] == 1) & (clim["HALF"] == 2), "R2_KE"].iloc[0]
    assert jan_first == pytest.approx(4.5)  # mean of 1.0 (2000-01-15) and 8.0 (2001-01-05)
    assert jan_second == pytest.approx(2.0)  # only 2000-01-20 in second half


def test_build_biweekly_export_frame_splits_month_halves() -> None:
    config = WeppcliffConfig(pmd="f", ei="t")
    daily = pd.DataFrame({
        "YYYYMMDD": [
            "20000105",
            "20000120",
            "20000210",
            "20010105",
        ],
        "MAX_TEMP": [1.0, 2.0, 4.0, 8.0],
        "MIN_TEMP": [0.0, 1.0, 3.0, 7.0],
    })
    data = WeppcliffData({10: daily, 9: _storm_frame()})
    out = build_weather_export_frame(data, config, "biweekly_mean")
    assert list(out.columns[:2]) == ["MM", "HALF"]
    jan_first = out.loc[(out["MM"] == "1") & (out["HALF"] == "1"), "MAX_TEMP"].iloc[0]
    jan_second = out.loc[(out["MM"] == "1") & (out["HALF"] == "2"), "MAX_TEMP"].iloc[0]
    assert jan_first == pytest.approx(4.5)
    assert jan_second == pytest.approx(2.0)
    assert "R2_KE" in out.columns


def test_export_weather_files_created(tmp_path: Path) -> None:
    from Source.args import assign_empty_args, assign_final_args, parse_args
    from Source.workflow import series_execution

    argv = [
        "-f", "ASOS_BPT_KMQE.csv",
        "-la", "y",
        "-ei", "t",
        "-ed", "7",
        "-d", str(ROOT / "Input"),
        "-o", str(tmp_path / "Output"),
        "-e", str(tmp_path / "Export"),
        "-p", str(tmp_path / "Plots"),
        "-verb", "f",
    ]
    config = parse_args(argv)
    config.args = argv
    assign_empty_args(config)
    set_config(config)
    series_execution([str(ROOT / "Input" / "ASOS_BPT_KMQE.csv")], 1, config)
    assign_final_args(config)

    export_dir = tmp_path / "Export"
    monthly = list(export_dir.glob("*-Monthly-Time-Series.csv"))
    assert monthly, "expected monthly timeseries export"
    frame = pd.read_csv(monthly[0])
    assert "YYYYMM" in frame.columns
    ke_cols = [c for c in frame.columns if c.endswith("_KE")]
    ei_cols = [c for c in frame.columns if c.endswith("_EI")]
    assert ke_cols, "expected kinetic energy columns in monthly export"
    assert ei_cols, "expected erosion index columns in monthly export"