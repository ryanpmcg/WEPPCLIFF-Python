"""Tests for CLIGEN .cli input parsing and conversion."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.cli_input import (  # noqa: E402
    cligen_to_input_frame,
    group_is_trusted,
    load_cligen_input,
    parse_cli_trust_groups,
    parse_cligen_file,
)
from Source.config import WeppcliffConfig, set_config


BF0 = ROOT / "Input" / "AS1-BF0.cli"
BF1 = ROOT / "Input" / "AS1-BF1.cli"
WEST = ROOT / "Input" / "West-Illinois.cli"


def test_parse_breakpoint_cli_header() -> None:
    cligen = parse_cligen_file(BF1)
    assert cligen.is_breakpoint is True
    assert cligen.header.station == "AS1"
    assert cligen.header.bf == 1
    assert len(cligen.header.monthly_precip) == 12
    assert len(cligen.days) > 4000


def test_parse_west_illinois_compact_cli() -> None:
    cligen = parse_cligen_file(WEST)
    assert cligen.is_breakpoint is False
    assert cligen.header.bf == 0
    assert len(cligen.days) > 7000
    assert cligen.days[0].year == 1901
    assert cligen.days[1].prcp == pytest.approx(1.7)


def test_parse_daily_cli_header() -> None:
    cligen = parse_cligen_file(BF0)
    assert cligen.is_breakpoint is False
    assert cligen.header.bf == 0
    assert cligen.days[0].prcp == pytest.approx(1.52)
    assert cligen.days[0].dur_h == pytest.approx(1.50)


def test_breakpoint_cli_converts_to_precip_timeseries() -> None:
    config = WeppcliffConfig()
    config.ctr = "all"
    config.cow = "none"
    frame = cligen_to_input_frame(parse_cligen_file(BF1), config)
    assert {"DT_1", "PRECIP", "DT_3", "MAX_TEMP", "STATION"}.issubset(frame.columns)
    assert frame["PRECIP"].notna().any()
    assert frame["DT_1"].notna().all()
    rainy = frame.loc[frame["PRECIP"] > 0, "PRECIP"].sum()
    assert rainy > 0


def test_daily_cli_converts_to_input_frame() -> None:
    config = WeppcliffConfig()
    frame = cligen_to_input_frame(parse_cligen_file(BF0), config)
    n_days = len(parse_cligen_file(BF0).days)
    assert len(frame) >= n_days
    assert frame["MAX_TEMP"].notna().any()


def test_overwrite_daily_clears_weather_columns() -> None:
    config = WeppcliffConfig()
    config.ctr = "all"
    config.cow = "daily"
    frame = cligen_to_input_frame(parse_cligen_file(BF0), config)
    assert frame["MAX_TEMP"].isna().all()
    assert frame["PRECIP"].notna().any()


def test_trust_groups_parser() -> None:
    assert parse_cli_trust_groups("all") == parse_cli_trust_groups(None, default_all=True)
    assert parse_cli_trust_groups("none") == set()
    assert parse_cli_trust_groups("daily,precip") == {"daily", "precip"}
    assert group_is_trusted("daily", {"daily"}, set()) is True
    assert group_is_trusted("daily", {"daily"}, {"daily"}) is False


def test_load_cligen_input_sets_config_from_header() -> None:
    config = WeppcliffConfig()
    set_config(config)
    frame = load_cligen_input(BF1, config)
    assert config.bf == 1
    assert str(config.pi).upper() == "F"
    assert config.cv == "0.0"
    assert frame["STATION"].iloc[0] == "AS1"


def test_cli_input_runs_through_workflow(tmp_path: Path) -> None:
    from Source.args import assign_empty_args, assign_final_args, parse_args
    from Source.workflow import series_execution

    argv = [
        "-f", "AS1-BF1.cli",
        "-la", "y",
        "-id", "t",
        "-qi", "t",
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
    series_execution([str(BF1)], 1, config)
    assign_final_args(config)
    out_cli = tmp_path / "Output" / "AS1.cli"
    assert out_cli.exists(), "expected regenerated .cli output"