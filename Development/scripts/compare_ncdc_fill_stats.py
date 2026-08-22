#!/usr/bin/env python3
"""Compare Python filled NCDC vs R NCDC-Update Results for all stations.

Uses Python ``.npz`` slots 10/5/9 (pre-fill) and 18/15/17 (post-fill) against
R Daily / Precipitation / Storm CSVs and ``.cli`` monthly P headers.

Does not modify legacy Results.
"""

from __future__ import annotations

import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.export_store import read_npz_slots  # noqa: E402

PY_COMPLETE = ROOT / "Export" / "NCDC-Python-Fill" / "Complete"
PY_RERUN = ROOT / "Export" / "NCDC-Python-Fill" / "Rerun"
LEGACY = Path("/Users/home/Software/WEPPCLIFF/NCDC-Update/Results")
OUT_DIR = ROOT / "Export" / "NCDC-Python-Fill" / "Compare"
WORKERS = 8

DAILY_WX = ["MIN_TEMP", "MAX_TEMP", "SO_RAD", "W_VEL", "DP_TEMP"]
DAILY_PCP = ["PRECIP", "BPS", "ELEN", "EMAX", "EMIN"]
STORM_COLS = ["DEPTH", "DUR", "PEAKINT", "I30", "R2_KE", "R2_EI"]
SLOTS = {5, 9, 10, 15, 17, 18}


def _num(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)


def _ymd(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str[:8]


def _cli_monthly_p(path: Path) -> list[float]:
    try:
        lines = path.read_text(errors="replace").splitlines()
        return [float(x) for x in lines[12].split()]
    except (OSError, ValueError, IndexError):
        return []


def _finite_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _stats_pair(prefix: str, a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    aa, bb = _finite_pair(a, b)
    n = int(len(aa))
    out = {
        f"{prefix}_n": n,
        f"{prefix}_rmse": _rmse(aa, bb) if n else float("nan"),
        f"{prefix}_bias": float((aa - bb).mean()) if n else float("nan"),
        f"{prefix}_py_mean": float(aa.mean()) if n else float("nan"),
        f"{prefix}_r_mean": float(bb.mean()) if n else float("nan"),
        f"{prefix}_py_max": float(np.nanmax(a)) if np.isfinite(a).any() else float("nan"),
        f"{prefix}_r_max": float(np.nanmax(b)) if np.isfinite(b).any() else float("nan"),
        f"{prefix}_py_p99": float(np.nanpercentile(a[np.isfinite(a)], 99)) if np.isfinite(a).sum() else float("nan"),
        f"{prefix}_r_p99": float(np.nanpercentile(b[np.isfinite(b)], 99)) if np.isfinite(b).sum() else float("nan"),
        f"{prefix}_py_sum": float(np.nansum(a)),
        f"{prefix}_r_sum": float(np.nansum(b)),
    }
    rsum = out[f"{prefix}_r_sum"]
    out[f"{prefix}_sum_ratio"] = (out[f"{prefix}_py_sum"] / rsum) if rsum not in (0.0, float("nan")) and np.isfinite(rsum) else float("nan")
    return out


def _monthly_clim_rmse(ymd: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    months = np.array([int(x[4:6]) if isinstance(x, str) and len(x) >= 6 and x[4:6].isdigit() else 0 for x in ymd])
    vals = []
    for m in range(1, 13):
        sel = months == m
        aa, bb = _finite_pair(a[sel], b[sel])
        if len(aa) == 0:
            return float("nan")
        vals.append((aa.mean() - bb.mean()) ** 2)
    return float(np.sqrt(np.mean(vals)))


def _annual_max(ymd: np.ndarray, vals: np.ndarray) -> float:
    years = np.array([x[:4] if isinstance(x, str) and len(x) >= 4 else "" for x in ymd])
    finite = np.isfinite(vals)
    if not finite.any():
        return float("nan")
    s = pd.Series(vals[finite]).groupby(years[finite]).sum(min_count=1)
    return float(s.max()) if len(s) else float("nan")


def _storm_summary(df: pd.DataFrame | None, prefix: str) -> dict[str, float]:
    out = {f"{prefix}_n": 0.0}
    if df is None or df.empty:
        for c in STORM_COLS:
            out[f"{prefix}_{c}_sum"] = float("nan")
            out[f"{prefix}_{c}_mean"] = float("nan")
            out[f"{prefix}_{c}_max"] = float("nan")
        return out
    out[f"{prefix}_n"] = float(len(df))
    for c in STORM_COLS:
        if c not in df.columns:
            out[f"{prefix}_{c}_sum"] = float("nan")
            out[f"{prefix}_{c}_mean"] = float("nan")
            out[f"{prefix}_{c}_max"] = float("nan")
            continue
        v = _num(df[c])
        out[f"{prefix}_{c}_sum"] = float(np.nansum(v))
        out[f"{prefix}_{c}_mean"] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
        out[f"{prefix}_{c}_max"] = float(np.nanmax(v)) if np.isfinite(v).any() else float("nan")
    return out


def py_paths(stem: str) -> tuple[Path | None, Path | None]:
    for root in (PY_COMPLETE, PY_RERUN):
        npz = root / "NPZ" / f"{stem}.npz"
        cli = root / "WEPP" / f"{stem}.cli"
        if npz.exists():
            return npz, cli if cli.exists() else None
    return None, None


def compare_station(stem: str) -> dict[str, object]:
    row: dict[str, object] = {"station": stem, "error": ""}
    npz_path, py_cli = py_paths(stem)
    if npz_path is None:
        row["error"] = "missing_python_npz"
        return row
    try:
        slots = read_npz_slots(npz_path, SLOTS)
    except Exception as exc:
        row["error"] = f"npz:{type(exc).__name__}:{exc}"
        return row

    r_daily_p = LEGACY / "Daily" / f"{stem} DAILY DATA.csv"
    r_pcp_p = LEGACY / "Precipitation" / f"{stem} PRECIP DATA.csv"
    r_st_p = LEGACY / "Storm" / f"{stem} STORM DATA.csv"
    r_cli = LEGACY / "WEPP" / f"{stem}.cli"
    if not r_daily_p.exists():
        row["error"] = "missing_r_daily"
        return row

    r_daily = pd.read_csv(r_daily_p)
    r_daily["YYYYMMDD"] = _ymd(r_daily["YYYYMMDD"])
    r_pcp = pd.read_csv(r_pcp_p, usecols=lambda c: c in {"PRECIP", "INT", "DUR"}) if r_pcp_p.exists() else None
    r_storm = pd.read_csv(r_st_p) if r_st_p.exists() else None

    py_cli_p = _cli_monthly_p(py_cli) if py_cli else []
    r_cli_p = _cli_monthly_p(r_cli) if r_cli.exists() else []
    row["py_cli_p_ann"] = float(sum(py_cli_p)) if py_cli_p else float("nan")
    row["r_cli_p_ann"] = float(sum(r_cli_p)) if r_cli_p else float("nan")
    row["cli_p_ratio"] = (
        row["py_cli_p_ann"] / row["r_cli_p_ann"]
        if row["r_cli_p_ann"] not in (0, float("nan")) and np.isfinite(row["r_cli_p_ann"])
        else float("nan")
    )
    row["cli_p_rmse"] = _rmse(np.array(py_cli_p, dtype=float), np.array(r_cli_p, dtype=float)) if len(py_cli_p) == 12 and len(r_cli_p) == 12 else float("nan")

    for slot, tag in ((10, "pre"), (18, "post")):
        daily = slots.get(slot)
        if daily is None or "YYYYMMDD" not in daily.columns:
            continue
        daily = daily.copy()
        daily["YYYYMMDD"] = _ymd(daily["YYYYMMDD"])
        merged = daily.merge(r_daily, on="YYYYMMDD", suffixes=("_py", "_r"), how="inner")
        ymd = merged["YYYYMMDD"].to_numpy()
        for col in DAILY_WX + DAILY_PCP:
            py_col = f"{col}_py" if f"{col}_py" in merged.columns else col
            r_col = f"{col}_r" if f"{col}_r" in merged.columns else None
            if r_col is None or py_col not in merged.columns:
                continue
            a = _num(merged[py_col])
            b = _num(merged[r_col])
            st = _stats_pair(f"{tag}_{col}", a, b)
            row.update(st)
            row[f"{tag}_{col}_mon_rmse"] = _monthly_clim_rmse(ymd, a, b)
            if col == "PRECIP":
                row[f"{tag}_PRECIP_annmax_py"] = _annual_max(ymd, a)
                row[f"{tag}_PRECIP_annmax_r"] = _annual_max(ymd, b)

        py_p = _num(daily["PRECIP"]) if "PRECIP" in daily.columns else np.array([])
        row[f"{tag}_daily_p_sum"] = float(np.nansum(py_p))
        row[f"{tag}_daily_p_wet"] = int(np.nansum(py_p > 0))
        row[f"{tag}_daily_p_nafrac"] = float(np.mean(~np.isfinite(py_p))) if len(py_p) else float("nan")

    r_p = _num(r_daily["PRECIP"])
    row["r_daily_p_sum"] = float(np.nansum(r_p))
    row["r_daily_p_wet"] = int(np.nansum(r_p > 0))
    row["r_daily_p_nafrac"] = float(np.mean(~np.isfinite(r_p))) if len(r_p) else float("nan")

    py5 = _num(slots[5]["PRECIP"]) if 5 in slots and "PRECIP" in slots[5].columns else np.array([np.nan])
    py15 = _num(slots[15]["PRECIP"]) if 15 in slots and "PRECIP" in slots[15].columns else np.array([np.nan])
    row["py5_pcp_sum"] = float(np.nansum(py5))
    row["py15_pcp_sum"] = float(np.nansum(py15))
    row["py15_pcp_max"] = float(np.nanmax(py15)) if np.isfinite(py15).any() else float("nan")
    r_bp = _num(r_pcp["PRECIP"]) if r_pcp is not None and "PRECIP" in r_pcp.columns else np.array([np.nan])
    row["r_pcp_sum"] = float(np.nansum(r_bp))
    row["r_pcp_int_mean"] = float(np.nanmean(_num(r_pcp["INT"]))) if r_pcp is not None and "INT" in r_pcp.columns else float("nan")
    row["py5_int_mean"] = float(np.nanmean(_num(slots[5]["INT"]))) if 5 in slots and "INT" in slots[5].columns else float("nan")
    row["py15_int_mean"] = float(np.nanmean(_num(slots[15]["INT"]))) if 15 in slots and "INT" in slots[15].columns else float("nan")

    row["mb_pre_daily_vs_pcp"] = (
        row["pre_daily_p_sum"] / row["py5_pcp_sum"]
        if row.get("pre_daily_p_sum") and row["py5_pcp_sum"] not in (0, float("nan"))
        else float("nan")
    )
    row["mb_post_daily_vs_pcp"] = (
        row.get("post_daily_p_sum", float("nan")) / row["py15_pcp_sum"]
        if row.get("post_daily_p_sum") and row["py15_pcp_sum"] not in (0, float("nan"))
        else float("nan")
    )
    row["mb_r_daily_vs_pcp"] = (
        row["r_daily_p_sum"] / row["r_pcp_sum"]
        if row["r_pcp_sum"] not in (0, float("nan"))
        else float("nan")
    )
    row["mb_pre_vs_r_daily"] = (
        row.get("pre_daily_p_sum", float("nan")) / row["r_daily_p_sum"]
        if row["r_daily_p_sum"] not in (0, float("nan"))
        else float("nan")
    )
    row["mb_post_vs_r_daily"] = (
        row.get("post_daily_p_sum", float("nan")) / row["r_daily_p_sum"]
        if row["r_daily_p_sum"] not in (0, float("nan"))
        else float("nan")
    )

    row.update(_storm_summary(slots.get(9) if isinstance(slots.get(9), pd.DataFrame) else None, "pre_storm"))
    row.update(_storm_summary(slots.get(17) if isinstance(slots.get(17), pd.DataFrame) else None, "post_storm"))
    row.update(_storm_summary(r_storm, "r_storm"))
    for c in STORM_COLS:
        rsum = row.get(f"r_storm_{c}_sum", float("nan"))
        for tag in ("pre", "post"):
            psum = row.get(f"{tag}_storm_{c}_sum", float("nan"))
            row[f"{tag}_storm_{c}_sum_ratio"] = (
                psum / rsum if isinstance(rsum, float) and rsum not in (0.0, float("nan")) and np.isfinite(rsum) else float("nan")
            )

    row["exploded_post_precip"] = int(
        (isinstance(row.get("mb_post_vs_r_daily"), float) and row["mb_post_vs_r_daily"] > 10)
        or (isinstance(row.get("cli_p_ratio"), float) and row["cli_p_ratio"] > 10)
    )
    return row


def _quantile(vals: list[float], q: float) -> float:
    arr = np.array([v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def _fmt(v: object, nd: int = 3) -> str:
    if not isinstance(v, (int, float)) or not np.isfinite(v):
        return "NA"
    if abs(v) >= 1000:
        return f"{v:.3e}"
    return f"{v:.{nd}f}"


def summarize(rows: list[dict[str, object]]) -> str:
    ok = [r for r in rows if not r.get("error")]
    n = len(ok)
    exploded = sum(int(r.get("exploded_post_precip") or 0) for r in ok)
    keys = [
        ("pre_MIN_TEMP_rmse", "Pre-fill Tmin RMSE (°C)"),
        ("pre_MAX_TEMP_rmse", "Pre-fill Tmax RMSE (°C)"),
        ("pre_SO_RAD_rmse", "Pre-fill solar RMSE (Ly/d)"),
        ("pre_W_VEL_rmse", "Pre-fill wind RMSE (m/s)"),
        ("pre_DP_TEMP_rmse", "Pre-fill dewpoint RMSE (°C)"),
        ("post_MIN_TEMP_rmse", "Post-fill Tmin RMSE (°C)"),
        ("post_MAX_TEMP_rmse", "Post-fill Tmax RMSE (°C)"),
        ("post_SO_RAD_rmse", "Post-fill solar RMSE (Ly/d)"),
        ("mb_pre_vs_r_daily", "Mass ratio pre-fill daily P / R daily P"),
        ("mb_post_vs_r_daily", "Mass ratio post-fill daily P / R daily P"),
        ("cli_p_ratio", "CLI monthly-P annual sum Python/R"),
        ("mb_pre_daily_vs_pcp", "Python pre-fill daily P / breakpoint P"),
        ("mb_post_daily_vs_pcp", "Python post-fill daily P / breakpoint P"),
        ("mb_r_daily_vs_pcp", "R daily P / breakpoint P"),
        ("pre_PRECIP_rmse", "Pre-fill overlapping daily P RMSE (mm)"),
        ("post_PRECIP_rmse", "Post-fill overlapping daily P RMSE (mm)"),
        ("pre_storm_DEPTH_sum_ratio", "Pre-fill storm depth sum / R"),
        ("post_storm_DEPTH_sum_ratio", "Post-fill storm depth sum / R"),
        ("pre_storm_I30_mean", "Pre-fill mean I30"),
        ("r_storm_I30_mean", "R mean I30"),
        ("post_storm_I30_mean", "Post-fill mean I30"),
        ("pre_storm_R2_EI_sum_ratio", "Pre-fill R2_EI sum / R"),
        ("post_storm_R2_EI_sum_ratio", "Post-fill R2_EI sum / R"),
        ("pre_PRECIP_mon_rmse", "Pre-fill monthly-clim P RMSE (mm)"),
        ("post_PRECIP_mon_rmse", "Post-fill monthly-clim P RMSE (mm)"),
        ("pre_PRECIP_annmax_py", "Pre-fill max annual P (mm)"),
        ("pre_PRECIP_annmax_r", "R max annual P (mm)"),
        ("post_PRECIP_py_max", "Post-fill max daily P (mm)"),
        ("pre_PRECIP_py_max", "Pre-fill max daily P (mm)"),
        ("pre_PRECIP_r_max", "R max daily P (mm)"),
    ]
    lines = [
        "# NCDC Python fill vs R 1.7.1 Results",
        "",
        "Python: `Export/NCDC-Python-Fill/` (`-qc -id -pm -qi -ei -ed 2 -ef npz`).",
        "R: `~/Software/WEPPCLIFF/NCDC-Update/Results/`.",
        "",
        f"- Stations compared: **{n}** / {len(rows)} (errors: {len(rows) - n})",
        f"- Stations with exploded post-fill precip (ratio > 10): **{exploded}** / {n}",
        "",
        "## Interpretation",
        "",
        "Pre-fill daily tables (npz slot 10) match R observed precip mass on overlapping wet days.",
        "Post-fill tables (slot 18) and `.cli` monthly P are **not** usable: imputed-event rows",
        "wrote `YYYYMMDD` into `PRECIP` (e.g. 20170930 mm). That assignment is now fixed in",
        "`Source/impute.py`; this report describes the sweep that already ran.",
        "",
        "## Cross-station distribution (median [p10, p90])",
        "",
        "| Metric | p10 | median | p90 |",
        "|---|---:|---:|---:|",
    ]
    for key, label in keys:
        vals = [r.get(key) for r in ok]
        lines.append(
            f"| {label} | {_fmt(_quantile(vals, 0.1))} | {_fmt(_quantile(vals, 0.5))} | {_fmt(_quantile(vals, 0.9))} |"
        )

    lines += [
        "",
        "## Daily weather (post-fill slot 18 vs R, overlapping days)",
        "",
        "These fields were not poisoned by the date-as-precip bug.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    r_stems = {p.name.replace(" DAILY DATA.csv", "") for p in (LEGACY / "Daily").glob("*DAILY DATA.csv")}
    py_stems = {p.stem for p in (PY_COMPLETE / "NPZ").glob("*.npz")} | {
        p.stem for p in (PY_RERUN / "NPZ").glob("*.npz")
    }
    stems = sorted(r_stems & py_stems)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Comparing {len(stems)} stations with {WORKERS} workers", flush=True)
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(compare_station, s): s for s in stems}
        done = 0
        for fut in as_completed(futs):
            rows.append(fut.result())
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(stems)}", flush=True)
    rows.sort(key=lambda r: str(r.get("station")))
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "StationMetrics.csv"
    df.to_csv(csv_path, index=False)
    summary = summarize(rows)
    (OUT_DIR / "SUMMARY.md").write_text(summary, encoding="utf-8")
    print(f"Wrote {csv_path} ({len(df)} rows)")
    print(f"Wrote {OUT_DIR / 'SUMMARY.md'}")
    err = df[df["error"].astype(str) != ""]
    if len(err):
        print(f"Errors: {len(err)}")
        print(err[["station", "error"]].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
