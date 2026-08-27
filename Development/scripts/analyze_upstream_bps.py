#!/usr/bin/env python3
"""Compare upstream BPS / breakpoint accounting between R .rds and Python .pkl.

Analysis-only helper for CLI breakpoint enforcement planning.
"""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.data_model import WeppcliffData  # noqa: E402
from Source.export_store import load_exported_store  # noqa: E402

R_RDS = Path("/Users/home/Software/Antigravity/WEPPCLIFF/AUTOTEST/OUTPUT/AutoTest.1.rds")
PY_PKL = ROOT / "Autotest" / "Output" / "AutoTest.1.npz"
REPORT = ROOT / "Development" / "reports" / "AutoTest.1_Upstream_BPS_Report.txt"

# Python validation-violation days from breakpoint diff report
PY_VIOLATION_DATES = {
    "2002-08-23",
    "2003-02-21",
    "2007-02-28",
    "2007-06-06",
    "2007-12-20",
    "2015-01-27",
    "2015-02-15",
    "2018-02-22",
}


def load_r_slots() -> dict[int, pd.DataFrame]:
    """Load slot 5 and 10 from R RDS via Rscript CSV export."""
    tmp_dir = ROOT / "Development" / "reports" / "_r_export"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    f5 = tmp_dir / "slot5.csv"
    f10 = tmp_dir / "slot10.csv"
    r_code = f"""
d <- readRDS("{R_RDS}")
write.csv(d[[5]], "{f5}", row.names = FALSE)
write.csv(d[[10]], "{f10}", row.names = FALSE)
"""
    subprocess.check_call(["Rscript", "-e", r_code])
    return {
        5: pd.read_csv(f5),
        10: pd.read_csv(f10),
    }


def load_py_slots() -> dict[int, pd.DataFrame]:
    path = PY_PKL
    if not path.exists():
        legacy = path.with_suffix(".pkl")
        if legacy.exists():
            path = legacy
    data = WeppcliffData(load_exported_store(path))
    return {5: data.slot_df(5), 10: data.slot_df(10)}


def _parse_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="mixed", errors="coerce")


def raw_bps_from_pcp(pcp_df: pd.DataFrame) -> pd.Series:
    pcp = pcp_df.copy()
    pcp["DT_1"] = _parse_dt(pcp["DT_1"])
    dates = pcp["DT_1"].dt.strftime("%Y%m%d")
    return pcp.groupby(dates).size().astype(int)


def compute_export_bps(daily_df: pd.DataFrame, pcp_df: pd.DataFrame) -> pd.DataFrame:
    """Mirror generate_export_data BPS adjustments (both loop and header values)."""
    daily = daily_df.copy()
    daily["BPS"] = daily["BPS"].fillna(0)
    daily_ymd = daily["YYYYMMDD"].astype(str).to_numpy()
    daily_bps_adj = np.zeros(len(daily), dtype=float)

    pcp = pcp_df.copy()
    pcp["DT_1"] = _parse_dt(pcp["DT_1"])
    pcp_days = pcp["DT_1"].dt.strftime("%Y%m%d")
    pcp_times = (
        pcp["DT_1"].dt.hour * 3600
        + pcp["DT_1"].dt.minute * 60
        + pcp["DT_1"].dt.second
    )
    pcp_times = pcp_times.replace(0, 24 * 3600)
    clock_resets = pcp_times.diff()

    decrease_days = pcp_days[pcp["DT_1"].dt.strftime("%H:%M:%S") == "00:00:00"].unique()
    for dd in decrease_days:
        daily_bps_adj[daily_ymd == dd] -= 1
        prev = (pd.to_datetime(dd, format="%Y%m%d") - pd.Timedelta(days=1)).strftime("%Y%m%d")
        daily_bps_adj[daily_ymd == prev] += 1

    updated_bps = daily["BPS"].to_numpy(dtype=float) + daily_bps_adj
    pos = updated_bps > 0
    updated_bps[pos] = updated_bps[pos] + 1
    loop_bps = updated_bps.copy()

    pcp_new_days = pcp.index[(clock_resets < 0) | (pcp["ANT"] >= 1440) | (pcp["DUR"] >= 1440)]
    pcp_rain_gap = pcp.index[pcp["BTW"] > 0]
    pcp_gap_increase = pcp_rain_gap.difference(pcp_new_days)
    gap_days: list[str] = []
    for idx in pcp_gap_increase:
        dt = pcp.loc[idx, "DT_1"]
        if dt.strftime("%H%M%S") == "000000":
            dt = dt - pd.Timedelta(hours=24)
        gap_days.append(dt.strftime("%Y%m%d"))

    increase_rows: list[int] = []
    for day in gap_days:
        increase_rows.extend(np.where(daily_ymd == day)[0].tolist())
    header_bps = loop_bps.copy()
    if increase_rows:
        for row_idx, count in Counter(increase_rows).items():
            header_bps[row_idx] += count

    return pd.DataFrame({
        "YYYYMMDD": daily["YYYYMMDD"].astype(str),
        "raw_bps": daily_df["BPS"].fillna(0).astype(float),
        "loop_bps": loop_bps.astype(int),
        "header_bps": header_bps.astype(int),
    })


def simulate_cli_loop_bounds(
    daily_export: pd.DataFrame,
    pcp_len: int,
) -> pd.DataFrame:
    """Simulate create_cli_file current_pcp / last_pcp without writing lines."""
    current_pcp = 1
    rows = []
    for i in range(len(daily_export)):
        ymd = daily_export.iloc[i]["YYYYMMDD"]
        loop_bps = int(daily_export.iloc[i]["loop_bps"])
        header_bps = int(daily_export.iloc[i]["header_bps"])
        lines_written = 0
        last_pcp = current_pcp - 1
        if loop_bps > 0:
            last_pcp = current_pcp + loop_bps - 2
            # first line + precip lines + optional gap lines (upper bound: 2 per pcp row)
            lines_written = 1 + max(0, last_pcp - current_pcp + 1)
            if last_pcp >= pcp_len:
                lines_written = -1  # overflow marker
        rows.append({
            "YYYYMMDD": ymd,
            "current_pcp_start": current_pcp,
            "last_pcp": last_pcp,
            "loop_bps": loop_bps,
            "header_bps": header_bps,
            "min_lines": lines_written,
            "header_mismatch": header_bps != loop_bps,
        })
        current_pcp = last_pcp + 1
    return pd.DataFrame(rows)


def ymd_to_date_key(ymd: str) -> str:
    s = str(ymd)
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


def main() -> None:
    r_slots = load_r_slots()
    py_slots = load_py_slots()

    r_pcp, r_daily = r_slots[5], r_slots[10]
    py_pcp, py_daily = py_slots[5], py_slots[10]

    r_pcp_count = raw_bps_from_pcp(r_pcp)
    py_pcp_count = raw_bps_from_pcp(py_pcp)

    r_export = compute_export_bps(r_daily, r_pcp)
    py_export = compute_export_bps(py_daily, py_pcp)

    r_sim = simulate_cli_loop_bounds(r_export, len(r_pcp))
    py_sim = simulate_cli_loop_bounds(py_export, len(py_pcp))

    # Merge comparison on calendar day
    cmp = r_export.merge(py_export, on="YYYYMMDD", suffixes=("_r", "_py"))
    cmp["date_key"] = cmp["YYYYMMDD"].map(ymd_to_date_key)

    r_sim = r_sim.rename(columns={
        "current_pcp_start": "current_pcp_r",
        "last_pcp": "last_pcp_r",
        "header_mismatch": "header_loop_mismatch_r",
    })
    py_sim = py_sim.rename(columns={
        "current_pcp_start": "current_pcp_py",
        "last_pcp": "last_pcp_py",
        "header_mismatch": "header_loop_mismatch_py",
        "min_lines": "min_lines_py",
    })
    cmp = cmp.merge(
        r_sim[["YYYYMMDD", "current_pcp_r", "last_pcp_r", "header_loop_mismatch_r"]],
        on="YYYYMMDD",
    )
    cmp = cmp.merge(
        py_sim[["YYYYMMDD", "current_pcp_py", "last_pcp_py", "header_loop_mismatch_py", "min_lines_py"]],
        on="YYYYMMDD",
    )

    cmp["raw_bps_diff"] = cmp["raw_bps_py"] - cmp["raw_bps_r"]
    cmp["loop_bps_diff"] = cmp["loop_bps_py"] - cmp["loop_bps_r"]
    cmp["header_bps_diff"] = cmp["header_bps_py"] - cmp["header_bps_r"]
    cmp["pcp_count_r"] = cmp["YYYYMMDD"].map(r_pcp_count).fillna(0).astype(int)
    cmp["pcp_count_py"] = cmp["YYYYMMDD"].map(py_pcp_count).fillna(0).astype(int)
    cmp["pcp_count_diff"] = cmp["pcp_count_py"] - cmp["pcp_count_r"]

    lines: list[str] = []
    w = lines.append

    w("=" * 88)
    w("AutoTest.1 Upstream BPS Comparison (R .rds vs Python .pkl)")
    w("=" * 88)
    w("")
    w(f"R RDS:   {R_RDS}")
    w(f"Py PKL:  {PY_PKL}")
    w("")
    w("Pipeline stages compared:")
    w("  A. slot 10 raw BPS (process_precip_data output)")
    w("  B. slot 5 precip row count per calendar day")
    w("  C. loop_bps  — value used by create_cli_file iteration (daily_df.BPS after midnight/+1)")
    w("  D. header_bps — adjbrkpt written on daily line (after rain-gap adjustments)")
    w("")

    w("-" * 88)
    w("DATA SHAPE")
    w("-" * 88)
    w(f"  R  slot5 rows: {len(r_pcp):>6}  slot10 rows: {len(r_daily):>6}")
    w(f"  Py slot5 rows: {len(py_pcp):>6}  slot10 rows: {len(py_daily):>6}")
    w(f"  Daily rows in comparison: {len(cmp)}")
    w("")

    raw_diff_days = cmp[cmp["raw_bps_diff"] != 0]
    pcp_diff_days = cmp[cmp["pcp_count_diff"] != 0]
    loop_diff_days = cmp[cmp["loop_bps_diff"] != 0]
    header_diff_days = cmp[cmp["header_bps_diff"] != 0]

    w("-" * 88)
    w("STAGE DIVERGENCE COUNTS (R vs Python)")
    w("-" * 88)
    w(f"  A. Raw slot10 BPS differs:     {len(raw_diff_days):>5} days")
    w(f"  B. Slot5 row count differs:      {len(pcp_diff_days):>5} days")
    w(f"  C. loop_bps differs:           {len(loop_diff_days):>5} days")
    w(f"  D. header_bps (adjbrkpt) diff: {len(header_diff_days):>5} days")
    w("")

    w("-" * 88)
    w("HEADER vs LOOP MISMATCH (within each implementation)")
    w("-" * 88)
    w(f"  R  days where header_bps != loop_bps: {int(cmp['header_loop_mismatch_r'].sum())}")
    w(f"  Py days where header_bps != loop_bps: {int(cmp['header_loop_mismatch_py'].sum())}")
    w("  (Expected: rain-gap days where adjbrkpt exceeds loop bound.)")
    w("")

    w("-" * 88)
    w("PYTHON CLI VIOLATION DAYS — UPSTREAM STATE")
    w("-" * 88)
    for dk in sorted(PY_VIOLATION_DATES):
        row = cmp[cmp["date_key"] == dk]
        if row.empty:
            w(f"  {dk}: NOT FOUND in daily table")
            continue
        r = row.iloc[0]
        w(f"  {dk}:")
        w(
            f"    raw BPS     R={int(r['raw_bps_r'])} Py={int(r['raw_bps_py'])} "
            f"(pcp rows R={r['pcp_count_r']} Py={r['pcp_count_py']})"
        )
        w(
            f"    loop_bps    R={int(r['loop_bps_r'])} Py={int(r['loop_bps_py'])} "
            f"| header_bps R={int(r['header_bps_r'])} Py={int(r['header_bps_py'])}"
        )
        w(
            f"    Py loop sim: current_pcp={int(r['current_pcp_py'])} "
            f"last_pcp={int(r['last_pcp_py'])} min_lines={int(r['min_lines_py'])}"
        )
        if int(r["min_lines_py"]) == -1:
            w("    *** last_pcp exceeds precip dataframe length — CLI loop overflow ***")
        if int(r["loop_bps_py"]) == 0 and int(r["min_lines_py"]) > 0:
            w("    *** loop_bps=0 but prior day left current_pcp inside next day's rows ***")
    w("")

    w("-" * 88)
    w("FIRST 30 DAYS WITH ANY UPSTREAM DIFF (chronological)")
    w("-" * 88)
    any_diff = cmp[
        (cmp["raw_bps_diff"] != 0)
        | (cmp["pcp_count_diff"] != 0)
        | (cmp["loop_bps_diff"] != 0)
        | (cmp["header_bps_diff"] != 0)
    ].sort_values("YYYYMMDD")
    for _, r in any_diff.head(30).iterrows():
        w(
            f"  {r['date_key']}: raw {int(r['raw_bps_r'])}->{int(r['raw_bps_py'])} "
            f"pcp {int(r['pcp_count_r'])}->{int(r['pcp_count_py'])} "
            f"loop {int(r['loop_bps_r'])}->{int(r['loop_bps_py'])} "
            f"header {int(r['header_bps_r'])}->{int(r['header_bps_py'])}"
        )
    if len(any_diff) > 30:
        w(f"  ... {len(any_diff) - 30} more days with upstream diffs")
    w("")

    w("-" * 88)
    w("DIAGNOSIS")
    w("-" * 88)
    if len(pcp_diff_days) > 0:
        w("  • Slot 5 precip timeseries differs between R and Python — bugs are NOT limited to CLI.")
    else:
        w("  • Slot 5 precip row counts match R — precip QC pipeline is aligned at export time.")
    if len(raw_diff_days) > 0 and len(pcp_diff_days) == 0:
        w("  • Raw BPS differs without pcp count diff — investigate process_precip_data aggregation.")
    if len(loop_diff_days) > len(raw_diff_days):
        w("  • loop_bps diverges more than raw BPS — midnight/gap adjustment logic may differ.")
    elif len(loop_diff_days) == len(raw_diff_days) and len(raw_diff_days) > 0:
        w("  • loop_bps tracks raw BPS diffs — export adjustment is not amplifying divergence.")
    w("  • Python's 8 catastrophic CLI days are consistent with create_cli_file index drift")
    w("    when loop_bps=0 (skip breakpoint block) but current_pcp is not advanced past prior day.")
    w("  • R's 118 declared≠actual days are largely header_bps vs lines-written accounting,")
    w("    not upstream data loss — R also allows midnight 24→0.x time wraps in validation.")
    w("")
    w("=" * 88)

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()