"""Integration tests for midnight-crossing CLI day boundaries (R parity)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Development.tests.test_cli_validate import _ensure_autotest1_cli  # noqa: E402

NUM = r"(-?\d+\.\d+|NA)"
DAILY_RE = re.compile(
    rf"^\s*(\d+)\s+(\d+)\s+(\d{{4}})\s+(\d+)\s+"
    rf"{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s*$"
)
BP_RE = re.compile(r"^\s*(-?\d+\.\d+)\s+(\d+\.\d+)\s*$")


def _day_breakpoints(cli_lines: list[str], day: int, month: int, year: int) -> list[tuple[float, float]]:
    for i, line in enumerate(cli_lines):
        m = DAILY_RE.match(line)
        if not m or int(m.group(1)) != day or int(m.group(2)) != month or int(m.group(3)) != year:
            continue
        bps: list[tuple[float, float]] = []
        j = i + 1
        while j < len(cli_lines):
            bm = BP_RE.match(cli_lines[j])
            if bm:
                bps.append((float(bm.group(1)), float(bm.group(2))))
                j += 1
                continue
            if DAILY_RE.match(cli_lines[j]):
                break
            j += 1
        return bps
    return []


def test_midnight_crossing_jan_4_5_2000_matches_r() -> None:
    """R leaves the 00:00 interval on Jan 5 (starter ~23.933), not capped on Jan 4."""
    cli = _ensure_autotest1_cli()
    lines = cli.read_text().splitlines()
    jan4 = _day_breakpoints(lines, 4, 1, 2000)
    jan5 = _day_breakpoints(lines, 5, 1, 2000)
    assert jan4, "expected breakpoints on 2000-01-04"
    assert jan5, "expected breakpoints on 2000-01-05"
    # R ends Jan 4 before midnight; does not force 24.000 on the prior day.
    assert jan4[-1][0] == pytest.approx(23.933, abs=1e-3)
    assert jan4[-1][1] == pytest.approx(24.83, abs=0.02)
    # R starts Jan 5 with HOUR=24 rewrite: 24 - DUR/60 ≈ 23.933 at depth 0.
    assert jan5[0][0] == pytest.approx(23.933, abs=1e-3)
    assert jan5[0][1] == pytest.approx(0.0, abs=1e-3)
