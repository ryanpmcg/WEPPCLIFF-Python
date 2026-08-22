"""CLI breakpoint validation — enforces structural rules on export and .cli output."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# NA-aware daily row (bf=1): day month year breaks + six weather fields.
_NUM = r"(-?\d+\.\d+|NA)"
DAILY_LINE_RE = re.compile(
    rf"^\s*(\d+)\s+(\d+)\s+(\d{{4}})\s+(\d+)\s+"
    rf"{_NUM}\s+{_NUM}\s+{_NUM}\s+{_NUM}\s+{_NUM}\s+{_NUM}\s*$"
)
BP_LINE_RE = re.compile(r"^\s*(-?\d+\.\d+)\s+(\d+\.\d+)\s*$")

MIDNIGHT_WRAP_TOLERANCE = 1e-6


class CLIValidationError(Exception):
    """Raised when breakpoint validation fails before or after .cli write."""

    def __init__(self, violations: Sequence["Violation"]) -> None:
        self.violations = list(violations)
        preview = "; ".join(f"{v.date}: {v.message}" for v in self.violations[:5])
        extra = f" (+{len(self.violations) - 5} more)" if len(self.violations) > 5 else ""
        super().__init__(f"CLI validation failed ({len(self.violations)} issues): {preview}{extra}")


@dataclass
class Violation:
    date: str
    stage: str
    rule: int
    message: str
    severity: str = "error"  # error | warn

    def __str__(self) -> str:
        return f"[{self.severity}] {self.date} ({self.stage}) rule {self.rule}: {self.message}"


@dataclass
class Breakpoint:
    time_h: float
    depth_mm: float


@dataclass
class DayRecord:
    date_key: str
    day: int
    month: int
    year: int
    breaks_declared: int
    breakpoints: List[Breakpoint] = field(default_factory=list)


@dataclass
class SimResult:
    date_key: str
    day_index: int
    current_pcp: int
    last_pcp: int
    loop_bps: int
    header_bps: int
    expected_lines: int
    overflow: bool = False


def ymd_to_date_key(ymd: str) -> str:
    s = str(ymd)
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


def date_key_from_parts(day: int, month: int, year: int) -> str:
    return f"{year:04d}-{month:02d}-{day:02d}"


def is_breakpoint_cli(header: str) -> bool:
    return "Breaks" in header


def validate_raw_bps(daily_df: pd.DataFrame, pcp_df: pd.DataFrame) -> List[Violation]:
    """Checkpoint A: slot-10 BPS must match slot-5 row counts."""
    violations: List[Violation] = []
    dt = pd.to_datetime(pcp_df["DT_1"], format="mixed", errors="coerce")
    counts = dt.dt.strftime("%Y%m%d").value_counts()

    ymd = daily_df["YYYYMMDD"].astype(str)
    expected = ymd.map(lambda d: int(counts.get(d, 0)))
    raw = pd.to_numeric(daily_df["BPS"], errors="coerce")

    for idx in daily_df.index:
        dk = ymd_to_date_key(ymd.loc[idx])
        exp = int(expected.loc[idx])
        val = raw.loc[idx]
        if pd.isna(val):
            if exp != 0:
                violations.append(
                    Violation(dk, "raw_bps", 0, f"BPS is NA but slot 5 has {exp} rows", "error")
                )
            continue
        raw_i = int(val)
        if raw_i < 0:
            violations.append(Violation(dk, "raw_bps", 0, f"negative BPS={raw_i}", "error"))
        if raw_i != exp:
            violations.append(
                Violation(
                    dk,
                    "raw_bps",
                    0,
                    f"BPS={raw_i} does not match slot-5 row count {exp}",
                    "error",
                )
            )
    return violations


def _parse_int_series(val) -> int:
    if isinstance(val, str):
        return int(val.strip())
    return int(val)


def validate_export_bps(export) -> List[Violation]:
    """Checkpoint B: loop_bps / header_bps invariants on numeric export fields."""
    violations: List[Violation] = []
    daily_df = export[1]
    loop_vals = pd.to_numeric(daily_df["BPS"], errors="coerce").fillna(0).astype(int)
    header_vals = pd.to_numeric(daily_df["HEADER_BPS"], errors="coerce").fillna(0).astype(int)

    for i in range(len(daily_df)):
        ymd = str(daily_df.iloc[i]["YYYYMMDD"])
        dk = ymd_to_date_key(ymd)
        loop_bps = int(loop_vals.iloc[i])
        header_bps = int(header_vals.iloc[i])

        if loop_bps < 0 or header_bps < 0:
            violations.append(
                Violation(dk, "export_bps", 0, f"negative loop={loop_bps} header={header_bps}", "error")
            )
        if header_bps < loop_bps:
            violations.append(
                Violation(
                    dk,
                    "export_bps",
                    0,
                    f"header_bps={header_bps} < loop_bps={loop_bps}",
                    "error",
                )
            )
        if header_bps == 0 and loop_bps != 0:
            violations.append(
                Violation(
                    dk,
                    "export_bps",
                    0,
                    f"header_bps=0 but loop_bps={loop_bps}",
                    "error",
                )
            )
        if loop_bps == 0 and header_bps != 0:
            violations.append(
                Violation(
                    dk,
                    "export_bps",
                    0,
                    f"loop_bps=0 but header_bps={header_bps}",
                    "error",
                )
            )

    return violations


def simulate_day_lines(
    day_index: int,
    daily_df: pd.DataFrame,
    current_pcp: int,
    daily_rows: Sequence[int],
    emit_ctx,
    header_bps: int,
) -> Tuple[SimResult, int]:
    """
    Checkpoint C: simulate create_cli_file loop bounds for one day.

    Returns (SimResult, next_current_pcp).
    """
    from Source.cli_emission import _count_breakpoint_lines

    i = daily_rows[day_index] if day_index < len(daily_rows) else day_index
    ymd = str(daily_df.iloc[i]["YYYYMMDD"])
    dk = ymd_to_date_key(ymd)
    loop_bps = int(daily_df.iloc[i]["BPS"]) if daily_df.iloc[i]["BPS"] == daily_df.iloc[i]["BPS"] else 0

    last_pcp, expected_lines, next_current, overflow = _count_breakpoint_lines(
        loop_bps, current_pcp, emit_ctx
    )
    return (
        SimResult(
            date_key=dk,
            day_index=i,
            current_pcp=current_pcp,
            last_pcp=last_pcp,
            loop_bps=loop_bps,
            header_bps=header_bps,
            expected_lines=expected_lines,
            overflow=overflow,
        ),
        next_current,
    )


def validate_loop_bounds(sim: SimResult, pcp_len: int) -> List[Violation]:
    """Validate pre-write loop indices for a single day."""
    violations: List[Violation] = []
    if sim.loop_bps <= 0:
        return violations

    if sim.current_pcp < 1:
        violations.append(
            Violation(sim.date_key, "loop_bounds", 0, f"current_pcp={sim.current_pcp} < 1", "error")
        )
    if sim.last_pcp < sim.current_pcp:
        violations.append(
            Violation(
                sim.date_key,
                "loop_bounds",
                0,
                f"last_pcp={sim.last_pcp} < current_pcp={sim.current_pcp}",
                "error",
            )
        )
    if sim.last_pcp > pcp_len:
        violations.append(
            Violation(
                sim.date_key,
                "loop_bounds",
                0,
                f"last_pcp={sim.last_pcp} exceeds pcp_df length {pcp_len}",
                "error",
            )
        )
    if sim.overflow:
        violations.append(
            Violation(sim.date_key, "loop_bounds", 0, "precip row index overflow", "error")
        )
    return violations


def _is_midnight_wrap(prev_t: float, t: float) -> bool:
    # R rewrites exact midnight to HOUR=24, so a later storm the same calendar
    # day can legitimately go 24.000 -> afternoon (not only morning).
    return prev_t >= 24.0 - MIDNIGHT_WRAP_TOLERANCE and t < prev_t


def validate_breakpoint_sequence(
    breakpoints: Sequence[Breakpoint],
    *,
    date_key: str,
    stage: str,
    declared: int,
    allow_midnight_wrap: bool = True,
    check_negative_start: bool = True,
) -> List[Violation]:
    """Rules 1–5 (+ optional rule 6 warn) on a day's breakpoint lines."""
    violations: List[Violation] = []
    n = len(breakpoints)

    if n != declared:
        # R WEPPCLIFF can declare more breaks than it writes (rain-gap adjbrkpt).
        # Warn so AutoTest.1 can match R; structural gates may still surface the delta.
        violations.append(
            Violation(
                date_key,
                stage,
                1,
                f"declared_breaks={declared} actual_lines={n}",
                "warn",
            )
        )

    if declared == 0 or n == 0:
        return violations

    if breakpoints[0].depth_mm != 0.0:
        violations.append(
            Violation(
                date_key,
                stage,
                2,
                f"first depth must be 0.00, got {breakpoints[0].depth_mm:.2f}",
                "error",
            )
        )

    if check_negative_start and breakpoints[0].time_h < 0:
        # R can emit slightly negative starts; keep as warning for parity runs.
        violations.append(
            Violation(
                date_key,
                stage,
                6,
                f"first breakpoint time {breakpoints[0].time_h:.3f} < 0",
                "warn",
            )
        )

    for j in range(n):
        if breakpoints[j].time_h < 0:
            violations.append(
                Violation(
                    date_key,
                    stage,
                    6,
                    f"time {breakpoints[j].time_h:.3f} < 0 (bp {j + 1})",
                    "warn",
                )
            )

    for j in range(1, n):
        prev = breakpoints[j - 1]
        cur = breakpoints[j]
        if cur.time_h < prev.time_h and not (
            allow_midnight_wrap and _is_midnight_wrap(prev.time_h, cur.time_h)
        ):
            violations.append(
                Violation(
                    date_key,
                    stage,
                    3,
                    f"time decreased {prev.time_h:.3f} -> {cur.time_h:.3f} (bp {j + 1})",
                    "error",
                )
            )
        if cur.depth_mm < prev.depth_mm:
            violations.append(
                Violation(
                    date_key,
                    stage,
                    4,
                    f"depth decreased {prev.depth_mm:.2f} -> {cur.depth_mm:.2f} (bp {j + 1})",
                    "error",
                )
            )
        if cur.time_h > 24.0:
            violations.append(
                Violation(
                    date_key,
                    stage,
                    5,
                    f"time {cur.time_h:.3f} exceeds 24.00 (bp {j + 1})",
                    "error",
                )
            )

    if breakpoints[-1].time_h > 24.0:
        violations.append(
            Violation(
                date_key,
                stage,
                5,
                f"last time {breakpoints[-1].time_h:.3f} exceeds 24.00",
                "error",
            )
        )

    return violations


def validate_day_emission(
    breakpoints: Sequence[Breakpoint],
    *,
    date_key: str,
    header_bps: int,
) -> List[Violation]:
    """Validate emitted breakpoint lines for one day (rules 1–5 on content)."""
    return validate_breakpoint_sequence(
        breakpoints,
        date_key=date_key,
        stage="day_emission",
        declared=header_bps,
    )


def parse_cli_days(path: Path | None = None, *, text: str | None = None) -> List[DayRecord]:
    """Parse a breakpoint-format .cli file into per-day records."""
    if text is None:
        if path is None:
            raise ValueError("parse_cli_days requires path or text")
        text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    days: List[DayRecord] = []
    i = 0
    while i < len(lines):
        m = DAILY_LINE_RE.match(lines[i])
        if not m:
            i += 1
            continue
        day = DayRecord(
            date_key=date_key_from_parts(int(m.group(1)), int(m.group(2)), int(m.group(3))),
            day=int(m.group(1)),
            month=int(m.group(2)),
            year=int(m.group(3)),
            breaks_declared=int(m.group(4)),
        )
        i += 1
        while i < len(lines):
            bm = BP_LINE_RE.match(lines[i])
            if bm:
                day.breakpoints.append(
                    Breakpoint(time_h=float(bm.group(1)), depth_mm=float(bm.group(2)))
                )
                i += 1
                continue
            if DAILY_LINE_RE.match(lines[i]):
                break
            i += 1
        days.append(day)
    return days


def validate_cli_file(path: Path, *, errors_only: bool = False) -> List[Violation]:
    """Checkpoint D: post-write validation of an on-disk .cli file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    if not is_breakpoint_cli(text):
        return []

    violations: List[Violation] = []
    for day in parse_cli_days(text=text):
        day_v = validate_breakpoint_sequence(
            day.breakpoints,
            date_key=day.date_key,
            stage="cli_file",
            declared=day.breaks_declared,
        )
        violations.extend(day_v)

    if errors_only:
        return [v for v in violations if v.severity == "error"]
    return violations


def raise_on_errors(violations: Sequence[Violation]) -> None:
    """Raise CLIValidationError if any error-severity violations exist."""
    errors = [v for v in violations if v.severity == "error"]
    if errors:
        raise CLIValidationError(errors)