#!/usr/bin/env python3
"""Compare Python filled NCDC daily precip to same-site GHCNd PRCP.

Primary bench is GHCNd, not R. R Results are scored as a second fill when
present. Does not modify R Results or NCDC-Python-2026.
"""

from __future__ import annotations

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
GHCN_DIR = ROOT / "Export" / "NCDC-Python-Fill" / "GHCNd"
LEGACY = Path("/Users/home/Software/WEPPCLIFF/NCDC-Update/Results")
OUT_DIR = ROOT / "Export" / "NCDC-Python-Fill" / "Compare"
WORKERS = 8


def _ymd(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str[:8]


def _num(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)


def _npz_path(stem: str) -> Path | None:
    for root in (PY_RERUN, PY_COMPLETE):
        path = root / "NPZ" / f"{stem}.npz"
        if path.exists():
            return path
    return None


def _daily_p(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "YYYYMMDD": _ymd(df["YYYYMMDD"]),
            "P": pd.to_numeric(df["PRECIP"], errors="coerce"),
        }
    )
    return out


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def compare_station(stem: str) -> dict[str, object]:
    row: dict[str, object] = {"station": stem, "error": ""}
    ghcn_path = GHCN_DIR / f"{stem}.parquet"
    npz_path = _npz_path(stem)
    if not ghcn_path.exists():
        row["error"] = "missing_ghcn"
        return row
    if npz_path is None:
        row["error"] = "missing_npz"
        return row
    try:
        ghcn = pd.read_parquet(ghcn_path, columns=["YYYYMMDD", "PRCP"])
        ghcn["YYYYMMDD"] = _ymd(ghcn["YYYYMMDD"])
        ghcn["G"] = pd.to_numeric(ghcn["PRCP"], errors="coerce")
        slots = read_npz_slots(npz_path, {10, 18})
        pre = _daily_p(slots[10]).rename(columns={"P": "PRE"})
        post = _daily_p(slots[18]).rename(columns={"P": "POST"})
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}:{exc}"
        return row

    merged = pre.merge(post, on="YYYYMMDD", how="outer").merge(
        ghcn[["YYYYMMDD", "G"]], on="YYYYMMDD", how="inner"
    )
    g = _num(merged["G"])
    pre_p = _num(merged["PRE"])
    post_p = _num(merged["POST"])
    row["n_overlap"] = int(len(merged))
    row["g_mass"] = float(np.nansum(g))
    row["g_wet"] = int(np.nansum(g > 0))
    row["pre_mass"] = float(np.nansum(pre_p))
    row["pre_wet"] = int(np.nansum(pre_p > 0))
    row["post_mass"] = float(np.nansum(post_p))
    row["post_wet"] = int(np.nansum(post_p > 0))
    row["pre_mass_vs_g"] = row["pre_mass"] / row["g_mass"] if row["g_mass"] else float("nan")
    row["post_mass_vs_g"] = row["post_mass"] / row["g_mass"] if row["g_mass"] else float("nan")
    row["pre_rmse"] = _rmse(pre_p, g)
    row["post_rmse"] = _rmse(post_p, g)

    g_dry = g <= 0
    pre_dry = ~(pre_p > 0)
    row["fp_mass"] = float(np.nansum(np.where(pre_dry & g_dry, np.nan_to_num(post_p, nan=0.0), 0.0)))
    row["fp_days"] = int(np.nansum(pre_dry & g_dry & (post_p > 0)))
    row["miss_mass"] = float(np.nansum(np.where(pre_dry & (g > 0), g, 0.0)))
    row["filled_miss_mass"] = float(
        np.nansum(np.where(pre_dry & (g > 0), np.nan_to_num(post_p, nan=0.0), 0.0))
    )
    row["filled_miss_frac"] = (
        row["filled_miss_mass"] / row["miss_mass"] if row["miss_mass"] else float("nan")
    )

    merged["YYYYMM"] = merged["YYYYMMDD"].str[:6]
    monthly = merged.groupby("YYYYMM", as_index=False).agg(
        g=("G", "sum"), pre=("PRE", "sum"), post=("POST", "sum"), n=("G", "size")
    )
    zero_mo = monthly[monthly["pre"].fillna(0) <= 0]
    if len(zero_mo):
        gsum = float(zero_mo["g"].sum())
        row["zero_mo_n"] = int(len(zero_mo))
        row["zero_mo_g"] = gsum
        row["zero_mo_post"] = float(zero_mo["post"].sum())
        row["zero_mo_post_vs_g"] = (
            row["zero_mo_post"] / gsum if gsum else float("nan")
        )
        row["zero_mo_rmse"] = _rmse(
            zero_mo["post"].to_numpy(dtype=float), zero_mo["g"].to_numpy(dtype=float)
        )
    else:
        row["zero_mo_n"] = 0
        row["zero_mo_g"] = 0.0
        row["zero_mo_post"] = 0.0
        row["zero_mo_post_vs_g"] = float("nan")
        row["zero_mo_rmse"] = float("nan")

    r_path = LEGACY / "Daily" / f"{stem} DAILY DATA.csv"
    if r_path.exists():
        r_daily = pd.read_csv(r_path, usecols=lambda c: c in {"YYYYMMDD", "PRECIP"})
        r_daily["YYYYMMDD"] = _ymd(r_daily["YYYYMMDD"])
        r_daily["R"] = pd.to_numeric(r_daily["PRECIP"], errors="coerce")
        with_r = merged.merge(r_daily[["YYYYMMDD", "R"]], on="YYYYMMDD", how="inner")
        r_p = _num(with_r["R"])
        g_r = _num(with_r["G"])
        row["r_mass"] = float(np.nansum(r_p))
        row["r_wet"] = int(np.nansum(r_p > 0))
        row["r_mass_vs_g"] = (
            row["r_mass"] / float(np.nansum(g_r)) if np.nansum(g_r) else float("nan")
        )
        row["r_rmse"] = _rmse(r_p, g_r)
    return row


def _quantile(vals: list[object], q: float) -> float:
    arr = np.array(
        [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)],
        dtype=float,
    )
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def summarize(rows: list[dict[str, object]]) -> str:
    ok = [r for r in rows if not r.get("error")]
    n = len(ok)
    keys = [
        ("post_mass_vs_g", "Post-fill daily mass / GHCN"),
        ("pre_mass_vs_g", "Pre-fill daily mass / GHCN"),
        ("r_mass_vs_g", "R daily mass / GHCN"),
        ("post_rmse", "Post-fill daily RMSE vs GHCN (mm)"),
        ("r_rmse", "R daily RMSE vs GHCN (mm)"),
        ("filled_miss_frac", "Fill mass on 15-min-dry GHCN-wet days / GHCN"),
        ("zero_mo_post_vs_g", "15-min-zero month mass / GHCN"),
        ("fp_mass", "Invented mass on GHCN-dry 15-min-dry days (mm)"),
    ]
    lines = [
        "# NCDC Python fill vs same-site GHCNd",
        "",
        "Primary bench is GHCNd daily PRCP (same station ID), not R.",
        "Python: `Export/NCDC-Python-Fill/` after month-of-year gap detection.",
        "",
        f"- Stations compared: **{n}** / {len(rows)} "
        f"(errors: {len(rows) - n})",
        "",
        "| Metric | p10 | median | p90 |",
        "|---|---:|---:|---:|",
    ]
    for key, label in keys:
        vals = [r.get(key) for r in ok]
        lines.append(
            f"| {label} | {_quantile(vals, 0.1):.3f} | "
            f"{_quantile(vals, 0.5):.3f} | {_quantile(vals, 0.9):.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    stems = sorted({p.stem for p in GHCN_DIR.glob("*.parquet")})
    if not stems:
        raise SystemExit(f"No GHCNd parquet files in {GHCN_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Comparing {len(stems)} stations to GHCNd", flush=True)
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
    csv_path = OUT_DIR / "GhcnStationMetrics.csv"
    df.to_csv(csv_path, index=False)
    summary = summarize(rows)
    (OUT_DIR / "GHCN-SUMMARY.md").write_text(summary, encoding="utf-8")
    print(f"Wrote {csv_path} ({len(df)} rows)")
    print(summary)
    err = df[df["error"].astype(str) != ""]
    if len(err):
        print(f"Errors: {len(err)}")
        print(err["error"].value_counts().head().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
