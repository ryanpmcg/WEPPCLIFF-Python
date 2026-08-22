"""Tests for CLI breakpoint validation."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.cli_validate import (  # noqa: E402
    DAILY_LINE_RE,
    parse_cli_days,
    validate_breakpoint_sequence,
    validate_cli_file,
    Breakpoint,
)

AUTOTEST_CLI = ROOT / "Autotest" / "Output" / "AutoTest.1.cli"


def _ensure_autotest1_cli() -> Path:
    """Regenerate AutoTest.1.cli so file-based tests use current emission logic."""
    from Development.tests.test_autotest import _run_weppcliff

    name, args = "AutoTest.1", ["-f", "ASOS_BPT_KMQE.csv", "-la", "y", "-ed", "2"]
    return _run_weppcliff(name, args)


def test_daily_regex_accepts_na_weather_fields() -> None:
    line = "   1    3  2007      0     4.1   -5.0  360.2     NA     NA  -13.0 "
    assert DAILY_LINE_RE.match(line) is not None


def test_parse_cli_recognizes_2007_03_01_na_boundary() -> None:
    if not AUTOTEST_CLI.exists():
        pytest.skip("AutoTest.1.cli not generated yet")
    days = parse_cli_days(AUTOTEST_CLI)
    by_date = {d.date_key: d for d in days}
    assert "2007-03-01" in by_date
    mar1 = by_date["2007-03-01"]
    assert mar1.breaks_declared == 0
    assert len(mar1.breakpoints) == 0


def test_autotest1_cli_has_no_validation_errors() -> None:
    cli = _ensure_autotest1_cli()
    violations = validate_cli_file(cli, errors_only=True)
    assert violations == []


def test_midnight_wrap_allowed() -> None:
    bps = [
        Breakpoint(24.0, 0.0),
        Breakpoint(0.5, 1.5),
    ]
    violations = validate_breakpoint_sequence(
        bps, date_key="2000-01-01", stage="test", declared=2
    )
    assert violations == []


def test_declared_actual_mismatch_flagged() -> None:
    bps = [Breakpoint(1.0, 0.0)]
    violations = validate_breakpoint_sequence(
        bps, date_key="2000-01-01", stage="test", declared=2
    )
    assert any(v.rule == 1 for v in violations)


def test_negative_time_is_warn_for_r_parity() -> None:
    """R can emit slightly negative start times; rule 6 is warn, not hard error."""
    bps = [Breakpoint(-0.033, 0.0), Breakpoint(1.0, 0.5)]
    violations = validate_breakpoint_sequence(
        bps, date_key="2000-01-05", stage="test", declared=2
    )
    assert any(v.rule == 6 and v.severity == "warn" for v in violations)


def test_rle_matches_counter_gap_aggregation() -> None:
    """PR4: Python Counter gap increments must match R rle(increase_days)."""
    increase_rows = [3, 3, 3, 7, 8, 8]
    rle_vals = []
    rle_lens = []
    prev = None
    count = 0
    for row in increase_rows:
        if row != prev:
            if prev is not None:
                rle_vals.append(prev)
                rle_lens.append(count)
            prev = row
            count = 1
        else:
            count += 1
    if prev is not None:
        rle_vals.append(prev)
        rle_lens.append(count)

    updated = np.zeros(10, dtype=float)
    for val, length in zip(rle_vals, rle_lens):
        updated[val] += length

    counter_updated = np.zeros(10, dtype=float)
    for row_idx, count in Counter(increase_rows).items():
        counter_updated[row_idx] += count

    np.testing.assert_array_equal(updated, counter_updated)