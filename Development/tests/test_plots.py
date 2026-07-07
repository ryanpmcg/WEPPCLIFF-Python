"""Tests for WEPPCLIFF matplotlib plotting."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _run_with_plots(tmp_path: Path) -> list[Path]:
    from Source.args import assign_empty_args, assign_final_args, parse_args
    from Source.config import set_config
    from Source.workflow import series_execution

    plot_dir = tmp_path / "Plots"
    argv = [
        "-f", "ASOS_BPT_KMQE.csv",
        "-pd", "t",
        "-la", "y",
        "-d", str(ROOT / "Input"),
        "-o", str(tmp_path / "Output"),
        "-e", str(tmp_path / "Export"),
        "-p", str(plot_dir),
    ]
    config = parse_args(argv)
    config.args = argv
    assign_empty_args(config)
    set_config(config)
    series_execution([str(ROOT / "Input" / "ASOS_BPT_KMQE.csv")], 1, config)
    assign_final_args(config)
    return sorted(plot_dir.rglob("*.png"))


def test_graph_data_creates_png_files(tmp_path: Path) -> None:
    pngs = _run_with_plots(tmp_path)
    assert pngs, "expected plot PNG files under Plots/"
    names = " ".join(p.name for p in pngs)
    assert "Precipitation" in names
    assert any("Input-Time-Series.png" in p.name for p in pngs)
    assert all(" " not in p.name for p in pngs)


def test_font_sizes_scale_with_figure_width() -> None:
    from Source.plots import _TS_FIGSIZE, _font_sizes

    fonts = _font_sizes(_TS_FIGSIZE)
    assert fonts["title"] > fonts["label"] > fonts["legend"]
    assert fonts["title"] >= 25


def test_histogram_figsize_is_two_to_one() -> None:
    from Source.plots import _HIST_FIGSIZE

    width, height = _HIST_FIGSIZE
    assert width == pytest.approx(2 * height)


def test_observation_interval_stats() -> None:
    from Source.plots import _format_interval_minutes, _observation_interval_stats_minutes

    dt = pd.to_datetime(
        ["2000-01-01 00:00", "2000-01-01 00:30", "2000-01-01 02:00"],
    )
    stats = _observation_interval_stats_minutes(dt)
    assert stats is not None
    assert stats["min"] == pytest.approx(30.0)
    assert stats["max"] == pytest.approx(90.0)
    assert stats["median"] == pytest.approx(60.0)
    assert "min" in _format_interval_minutes(stats["min"])


def test_line_timeseries_breaks_at_nan() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from Source.plots import _plot_timeseries

    dt = pd.date_range("2000-01-01", periods=5, freq="D")
    values = [1.0, 2.0, np.nan, 4.0, 5.0]
    fig, ax = plt.subplots()
    _plot_timeseries(ax, dt, values, "l", color="black")
    line = ax.lines[0]
    ydata = line.get_ydata()
    assert np.isnan(ydata[2])
    assert len(ydata) == 5
    plt.close(fig)


def test_plot_labels_use_title_case(tmp_path: Path) -> None:
    from Source.plots import _title_label

    assert _title_label("PRECIP") == "Precipitation"
    assert _title_label("MIN_TEMP") == "Minimum Temperature"
    assert _title_label("PDUR") == "Peak Duration"
    assert _title_label("R2_KE") == "R2 Kinetic Energy"


def test_plot_skip_reason_missing_input_column() -> None:
    from Source.data_model import WeppcliffData
    from Source.plots import _plot_skip_reason

    data = WeppcliffData({1: pd.DataFrame({"DT_1": ["2000-01-01"], "PRECIP": [1.0]})})
    reason = _plot_skip_reason(data, 1, "AIR_TEMP")
    assert reason is not None
    assert "Air Temperature (AIR_TEMP)" in reason
    assert "not provided in the input file" in reason


def test_plot_skip_reason_no_plottable_values() -> None:
    from Source.data_model import WeppcliffData
    from Source.plots import _plot_skip_reason

    data = WeppcliffData({
        12: pd.DataFrame({"YYYY": ["2000"], "AIR_TEMP": [float("nan")]}),
    })
    reason = _plot_skip_reason(data, 12, "AIR_TEMP")
    assert reason is not None
    assert "no plottable values" in reason