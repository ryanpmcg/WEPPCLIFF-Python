#!/usr/bin/env python3
"""Analyze AutoTest.1 breakpoint differences between R and Python .cli outputs."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.cli_validate import (  # noqa: E402
    BP_LINE_RE as BP_RE,
    Breakpoint,
    DAILY_LINE_RE as DAILY_RE,
    validate_breakpoint_sequence,
)

R_CLI = Path("/Users/home/Software/Antigravity/WEPPCLIFF/AUTOTEST/OUTPUT/AutoTest.1.cli")
PY_CLI = ROOT / "Autotest" / "Output" / "AutoTest.1.cli"
REPORT = ROOT / "Development" / "reports" / "AutoTest.1_Breakpoint_Diff_Report.txt"


@dataclass
class ParsedBreakpoint:
    time_h: float
    depth_mm: float
    line_no: int


@dataclass
class DayRecord:
    line_no: int
    day: int
    month: int
    year: int
    breaks_declared: int
    tmax: float
    tmin: float
    breakpoints: List[ParsedBreakpoint] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)

    @property
    def date_key(self) -> str:
        return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"


def parse_cli(path: Path) -> Tuple[List[str], List[DayRecord]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    days: List[DayRecord] = []
    i = 0
    while i < len(lines):
        m = DAILY_RE.match(lines[i])
        if not m:
            i += 1
            continue
        def _opt_float(val: str) -> float:
            return float(val) if val != "NA" else float("nan")

        day = DayRecord(
            line_no=i + 1,
            day=int(m.group(1)),
            month=int(m.group(2)),
            year=int(m.group(3)),
            breaks_declared=int(m.group(4)),
            tmax=_opt_float(m.group(5)),
            tmin=_opt_float(m.group(6)),
        )
        i += 1
        while i < len(lines):
            bm = BP_RE.match(lines[i])
            if bm:
                day.breakpoints.append(
                    ParsedBreakpoint(
                        time_h=float(bm.group(1)),
                        depth_mm=float(bm.group(2)),
                        line_no=i + 1,
                    )
                )
                i += 1
                continue
            if DAILY_RE.match(lines[i]):
                break
            i += 1
        days.append(day)
    return lines, days


def validate_day(day: DayRecord, source: str) -> None:
    bps = [
        Breakpoint(time_h=b.time_h, depth_mm=b.depth_mm)
        for b in day.breakpoints
    ]
    for v in validate_breakpoint_sequence(
        bps,
        date_key=day.date_key,
        stage=source,
        declared=day.breaks_declared,
        allow_midnight_wrap=True,
    ):
        if v.severity == "error":
            day.violations.append(f"{source}: {v.message}")


def compare_days(r_day: DayRecord, p_day: DayRecord) -> Optional[dict]:
    if (
        r_day.breaks_declared == p_day.breaks_declared
        and len(r_day.breakpoints) == len(p_day.breakpoints)
        and all(
            abs(a.time_h - b.time_h) < 1e-6 and abs(a.depth_mm - b.depth_mm) < 1e-6
            for a, b in zip(r_day.breakpoints, p_day.breakpoints)
        )
    ):
        return None
    return {
        "date": r_day.date_key,
        "r_line": r_day.line_no,
        "py_line": p_day.line_no,
        "r_breaks": r_day.breaks_declared,
        "py_breaks": p_day.breaks_declared,
        "r_bp_count": len(r_day.breakpoints),
        "py_bp_count": len(p_day.breakpoints),
        "r_bps": [(b.time_h, b.depth_mm) for b in r_day.breakpoints],
        "py_bps": [(b.time_h, b.depth_mm) for b in p_day.breakpoints],
        "r_violations": r_day.violations,
        "py_violations": p_day.violations,
    }


def format_bp_list(bps: List[Tuple[float, float]], max_show: int = 20) -> List[str]:
    out = []
    for idx, (t, d) in enumerate(bps):
        if idx >= max_show:
            out.append(f"    ... ({len(bps) - max_show} more)")
            break
        out.append(f"    [{idx+1:3d}] t={t:7.3f}  depth={d:6.2f}")
    return out


def main() -> None:
    _, r_days = parse_cli(R_CLI)
    _, p_days = parse_cli(PY_CLI)

    for d in r_days:
        validate_day(d, "R")
    for d in p_days:
        validate_day(d, "Python")

    r_by_date = {d.date_key: d for d in r_days}
    p_by_date = {d.date_key: d for d in p_days}
    all_dates = sorted(set(r_by_date) | set(p_by_date))

    diffs = []
    for dk in all_dates:
        rd = r_by_date.get(dk)
        pd = p_by_date.get(dk)
        if rd is None or pd is None:
            diffs.append({"date": dk, "missing": "R" if rd is None else "Python"})
            continue
        cmp = compare_days(rd, pd)
        if cmp:
            diffs.append(cmp)

    r_violation_days = [d for d in r_days if d.violations]
    py_violation_days = [d for d in p_days if d.violations]

    lines: List[str] = []
    w = lines.append

    w("=" * 88)
    w("AutoTest.1 Breakpoint Difference Report")
    w("=" * 88)
    w("")
    w("Sources:")
    w(f"  R:      {R_CLI}")
    w(f"  Python: {PY_CLI}")
    w("")
    w("Validation rules applied:")
    w("  1. Daily 'Breaks' column must equal number of breakpoint lines that follow.")
    w("  2. First breakpoint line must be depth 0.00 (start of recorded precipitation).")
    w("  3. Within a day: times non-decreasing; cumulative depths non-decreasing.")
    w("  4. No breakpoint time may exceed 24.00 hours.")
    w("  5. (Flat depth plateaus imply zero increment — not flagged as violation.)")
    w("")
    w("-" * 88)
    w("SUMMARY")
    w("-" * 88)
    w(f"  R daily rows parsed:      {len(r_days)}")
    w(f"  Python daily rows parsed: {len(p_days)}")
    w(f"  Calendar days with any breakpoint diff: {len(diffs)}")
    w(f"  R days failing validation rules:      {len(r_violation_days)}")
    w(f"  Python days failing validation rules: {len(py_violation_days)}")
    w("")

    w("-" * 88)
    w("VALIDATION VIOLATIONS BY IMPLEMENTATION")
    w("-" * 88)
    w("")
    w(f"R violations ({len(r_violation_days)} days):")
    if not r_violation_days:
        w("  (none)")
    else:
        for d in r_violation_days[:50]:
            w(f"  {d.date_key} (daily line {d.line_no}, declared={d.breaks_declared}, actual={len(d.breakpoints)}):")
            for v in d.violations:
                w(f"    - {v}")
        if len(r_violation_days) > 50:
            w(f"  ... and {len(r_violation_days) - 50} more days")
    w("")
    w(f"Python violations ({len(py_violation_days)} days):")
    if not py_violation_days:
        w("  (none)")
    else:
        for d in py_violation_days[:50]:
            w(f"  {d.date_key} (daily line {d.line_no}, declared={d.breaks_declared}, actual={len(d.breakpoints)}):")
            for v in d.violations:
                w(f"    - {v}")
        if len(py_violation_days) > 50:
            w(f"  ... and {len(py_violation_days) - 50} more days")
    w("")

    w("-" * 88)
    w("BREAKPOINT COUNT MISMATCHES (declared vs actual lines)")
    w("-" * 88)
    r_count_mis = [d for d in r_days if d.breaks_declared != len(d.breakpoints)]
    py_count_mis = [d for d in p_days if d.breaks_declared != len(d.breakpoints)]
    w(f"  R:      {len(r_count_mis)} days")
    w(f"  Python: {len(py_count_mis)} days")
    w("")

    w("-" * 88)
    w("DECLARED BREAK COUNT DIFFERENCES (R vs Python, same calendar day)")
    w("-" * 88)
    break_count_diffs = [
        dk for dk in all_dates
        if dk in r_by_date and dk in p_by_date
        and r_by_date[dk].breaks_declared != p_by_date[dk].breaks_declared
    ]
    w(f"  Days where declared break count differs: {len(break_count_diffs)}")
    for dk in break_count_diffs[:100]:
        rd, pd = r_by_date[dk], p_by_date[dk]
        w(
            f"  {dk}: R declared={rd.breaks_declared} ({len(rd.breakpoints)} lines) "
            f"| Py declared={pd.breaks_declared} ({len(pd.breakpoints)} lines) "
            f"| delta={pd.breaks_declared - rd.breaks_declared:+d}"
        )
    if len(break_count_diffs) > 100:
        w(f"  ... and {len(break_count_diffs) - 100} more")
    w("")

    w("-" * 88)
    w("DETAILED BREAKPOINT DIFFERENCES (first 80 divergent days)")
    w("-" * 88)
    w("")
    shown = 0
    for item in diffs:
        if "missing" in item:
            w(f"DATE {item['date']}: missing in {item['missing']}")
            w("")
            shown += 1
            if shown >= 80:
                break
            continue

        w(f"DATE {item['date']}")
        w(f"  R daily line {item['r_line']}: declared breaks = {item['r_breaks']}, lines = {item['r_bp_count']}")
        w(f"  Python daily line {item['py_line']}: declared breaks = {item['py_breaks']}, lines = {item['py_bp_count']}")
        if item["r_violations"]:
            w("  R rule violations:")
            for v in item["r_violations"]:
                w(f"    - {v}")
        if item["py_violations"]:
            w("  Python rule violations:")
            for v in item["py_violations"]:
                w(f"    - {v}")

        r_bps = item["r_bps"]
        py_bps = item["py_bps"]
        max_len = max(len(r_bps), len(py_bps))
        first_mismatch = None
        for j in range(max_len):
            rt, rd = r_bps[j] if j < len(r_bps) else (None, None)
            pt, pd = py_bps[j] if j < len(py_bps) else (None, None)
            if rt is None or pt is None or abs(rt - pt) > 1e-6 or abs(rd - pd) > 1e-6:
                first_mismatch = j
                break

        if first_mismatch is not None:
            w(f"  First mismatch at breakpoint index {first_mismatch + 1}:")
            start = max(0, first_mismatch - 2)
            end = min(max_len, first_mismatch + 5)
            w("  Side-by-side (index | R time/depth | Python time/depth):")
            for j in range(start, end):
                rt, rd = r_bps[j] if j < len(r_bps) else (None, None)
                pt, pd = py_bps[j] if j < len(py_bps) else (None, None)
                rs = f"{rt:7.3f} / {rd:6.2f}" if rt is not None else "   ---  /   ---"
                ps = f"{pt:7.3f} / {pd:6.2f}" if pt is not None else "   ---  /   ---"
                mark = " <--" if j == first_mismatch else ""
                w(f"    [{j+1:3d}]  R: {rs}   |   Py: {ps}{mark}")

        if len(r_bps) <= 12 and len(py_bps) <= 12:
            w("  R breakpoints:")
            w("\n".join(format_bp_list(r_bps, 12)))
            w("  Python breakpoints:")
            w("\n".join(format_bp_list(py_bps, 12)))
        w("")
        shown += 1
        if shown >= 80:
            w(f"... truncated; {len(diffs) - 80} additional divergent days not shown.")
            break

    w("-" * 88)
    w("INTERPRETATION NOTES")
    w("-" * 88)
    w("  - If only one implementation has validation violations, that side is likely wrong.")
    w("  - If both pass validation but differ, compare which matches input precip timestamps.")
    w("  - Declared breaks should equal breakpoint line count; mismatch is a hard error.")
    w("  - A day with breaks=0 must have no breakpoint lines following the daily row.")
    w("")
    w("=" * 88)

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()