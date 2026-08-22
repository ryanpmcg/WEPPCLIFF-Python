"""Tests for R-parity breakpoint time logic."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.cli_breakpoint_time import (  # noqa: E402
    first_breakpoint_time,
    gap_breakpoint_time,
    interval_crosses_midnight,
    should_cap_last_breakpoint_at_midnight,
    storm_start_time_cligen,
)


def test_interval_crosses_midnight_early_morning() -> None:
    assert interval_crosses_midnight("00", "02", 4.0) is True
    assert interval_crosses_midnight("00", "00", 4.0) is True
    assert interval_crosses_midnight("14", "30", 10.0) is False


def test_first_breakpoint_matches_r_raw_formula() -> None:
    # R: HOUR + MIN/60 - DUR/60 (HOUR may be "24" at exact midnight).
    t = first_breakpoint_time("00", "00", "24", "00", 4.0)
    assert t == pytest.approx(24.0 - 4.0 / 60.0)


def test_first_breakpoint_same_day_afternoon() -> None:
    t = first_breakpoint_time("14", "30", "14", "30", 10.0)
    assert abs(t - (14.5 - 10.0 / 60.0)) < 1e-9


def test_gap_breakpoint_matches_r_raw_formula() -> None:
    assert gap_breakpoint_time("00", "01", 5.0) == pytest.approx(
        storm_start_time_cligen("00", "01", 5.0)
    )


def test_should_cap_disabled_for_r_parity() -> None:
    assert not should_cap_last_breakpoint_at_midnight(
        23.933, "23", "56", 11.0, next_day_opens_with_midnight_continuation=True
    )
    assert not should_cap_last_breakpoint_at_midnight(
        24.0, "00", "00", 4.0, next_day_opens_with_midnight_continuation=True
    )
