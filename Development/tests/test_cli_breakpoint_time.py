"""Tests for midnight breakpoint time logic."""

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
)


def test_interval_crosses_midnight_early_morning() -> None:
    assert interval_crosses_midnight("00", "02", 4.0) is True
    assert interval_crosses_midnight("00", "00", 4.0) is True
    assert interval_crosses_midnight("14", "30", 10.0) is False


def test_first_breakpoint_post_midnight_continuation() -> None:
    # 00:02 with DUR=4 min — post-midnight portion of split storm
    t = first_breakpoint_time("00", "02", "00", "02", 4.0)
    assert t == 0.0


def test_first_breakpoint_same_day_afternoon() -> None:
    t = first_breakpoint_time("14", "30", "14", "30", 10.0)
    assert abs(t - (14.5 - 10.0 / 60.0)) < 1e-9


def test_gap_breakpoint_never_negative() -> None:
    assert gap_breakpoint_time("00", "01", 5.0) == 0.0


def test_should_cap_when_next_day_continues() -> None:
    assert should_cap_last_breakpoint_at_midnight(
        23.933, "23", "56", 11.0, next_day_opens_with_midnight_continuation=True
    )


def test_no_cap_when_already_at_midnight() -> None:
    assert not should_cap_last_breakpoint_at_midnight(
        24.0, "00", "00", 4.0, next_day_opens_with_midnight_continuation=True
    )


def test_autotest1_no_negative_times() -> None:
    from Development.tests.test_cli_validate import _ensure_autotest1_cli

    cli_path = _ensure_autotest1_cli()
    import re

    bp_re = re.compile(r"^\s*(-?\d+\.\d+)\s+(\d+\.\d+)\s*$")
    for line in cli_path.read_text().splitlines():
        m = bp_re.match(line)
        if m:
            assert float(m.group(1)) >= 0.0, line