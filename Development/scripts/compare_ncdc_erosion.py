#!/usr/bin/env python3
"""Compare erosion metrics for the NCDC Python fill.

GHCN has no I30/EI. This scores:
1. Observed-storm preservation (pre vs post, matched timestamps).
2. New (filled) storms vs that station's observed storm climatology.
3. R storm tables as a second WEPP processor, not as truth.

Does not modify R Results or NCDC-Python-2026.
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
LEGACY = Path("/Users/home/Software/WEPPCLIFF/NCDC-Update/Results")
OUT_DIR = ROOT / "Export" / "NCDC-Python-Fill" / "Compare"
WORKERS = 8
STORM_COLS = ["DEPTH", "DUR", "PEAKINT", "I30", "R2_KE", "R2_EI"]


def _npz_path(stem: str) -> Path | None:
    for root in (PY_RERUN, PY_COMPLETE):
        path = root / "NPZ" / f"{stem}.npz"
        if path.exists():
            return path
    return None


def _storm_frame(obj: object) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, list):
        frames = [f for f in obj if isinstance(f, pd.DataFrame) and not f.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
    if isinstance(obj, pd.DataFrame):
        return obj
    return pd.DataFrame()


def _num(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)


def _keys(dt: pd.Series) -> pd.Series:
    return pd.to_datetime(dt, errors="coerce").dt.round("min")


def _ratio(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b) & (np.abs(a) > 1e-12)
    if not mask.any():
        return float("nan")
    return float(np.nanmedian(b[mask] / a[mask]))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def _med(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else float("nan")


def compare_station(stem: str) -> dict[str, object]:
    row: dict[str, object] = {"station": stem, "error": ""}
    npz = _npz_path(stem)
    if npz is None:
        row["error"] = "missing_npz"
        return row
    try:
        slots = read_npz_slots(npz, {9, 17})
        pre = _storm_frame(slots.get(9))
        post = _storm_frame(slots.get(17))
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}:{exc}"
        return row

    row["pre_n"] = int(len(pre))
    row["post_n"] = int(len(post))
    if pre.empty and post.empty:
        row["error"] = "no_storms"
        return row

    if not pre.empty and "DT" in pre.columns:
        pre = pre.copy()
        pre["key"] = _keys(pre["DT"])
    else:
        pre = pd.DataFrame(columns=["key", *STORM_COLS])
    if not post.empty and "DT" in post.columns:
        post = post.copy()
        post["key"] = _keys(post["DT"])
    else:
        post = pd.DataFrame(columns=["key", *STORM_COLS])

    matched = pre.merge(post, on="key", suffixes=("_pre", "_post"), how="inner")
    row["n_matched"] = int(len(matched))
    row["n_unmatched_pre"] = int(len(pre) - len(matched))
    new = post.loc[~post["key"].isin(set(pre["key"].dropna()))].copy()
    row["n_new"] = int(len(new))

    for col in STORM_COLS:
        pre_v = _num(pre[col]) if col in pre.columns else np.array([])
        post_v = _num(post[col]) if col in post.columns else np.array([])
        new_v = _num(new[col]) if col in new.columns else np.array([])
        row[f"pre_{col}_sum"] = float(np.nansum(pre_v))
        row[f"post_{col}_sum"] = float(np.nansum(post_v))
        row[f"pre_{col}_med"] = _med(pre_v)
        row[f"post_{col}_med"] = _med(post_v)
        row[f"pre_{col}_mean"] = float(np.nanmean(pre_v)) if np.isfinite(pre_v).any() else float("nan")
        row[f"post_{col}_mean"] = float(np.nanmean(post_v)) if np.isfinite(post_v).any() else float("nan")
        row[f"pre_{col}_max"] = float(np.nanmax(pre_v)) if np.isfinite(pre_v).any() else float("nan")
        row[f"post_{col}_max"] = float(np.nanmax(post_v)) if np.isfinite(post_v).any() else float("nan")
        row[f"new_{col}_med"] = _med(new_v)
        row[f"new_{col}_mean"] = float(np.nanmean(new_v)) if np.isfinite(new_v).any() else float("nan")
        row[f"new_{col}_sum"] = float(np.nansum(new_v))
        if col in matched.columns or f"{col}_pre" in matched.columns:
            a = _num(matched[f"{col}_pre"])
            b = _num(matched[f"{col}_post"])
            row[f"keep_{col}_ratio"] = _ratio(a, b)
            row[f"keep_{col}_rmse"] = _rmse(a, b)
        else:
            row[f"keep_{col}_ratio"] = float("nan")
            row[f"keep_{col}_rmse"] = float("nan")
        row[f"new_vs_obs_{col}_med_ratio"] = (
            row[f"new_{col}_med"] / row[f"pre_{col}_med"]
            if row[f"pre_{col}_med"] not in (0, float("nan"))
            and np.isfinite(row[f"pre_{col}_med"])
            and np.isfinite(row[f"new_{col}_med"])
            else float("nan")
        )

    r_path = LEGACY / "Storm" / f"{stem} STORM DATA.csv"
    if r_path.exists():
        try:
            rst = pd.read_csv(r_path)
        except Exception as exc:
            row["r_error"] = f"{type(exc).__name__}:{exc}"
            rst = pd.DataFrame()
        row["r_n"] = int(len(rst))
        for col in STORM_COLS:
            if col not in rst.columns:
                continue
            rv = _num(rst[col])
            row[f"r_{col}_sum"] = float(np.nansum(rv))
            row[f"r_{col}_med"] = _med(rv)
            row[f"r_{col}_mean"] = float(np.nanmean(rv)) if np.isfinite(rv).any() else float("nan")
            row[f"r_{col}_max"] = float(np.nanmax(rv)) if np.isfinite(rv).any() else float("nan")
            rsum = row[f"r_{col}_sum"]
            row[f"post_{col}_sum_vs_r"] = (
                row[f"post_{col}_sum"] / rsum
                if rsum not in (0, float("nan")) and np.isfinite(rsum)
                else float("nan")
            )
            row[f"pre_{col}_sum_vs_r"] = (
                row[f"pre_{col}_sum"] / rsum
                if rsum not in (0, float("nan")) and np.isfinite(rsum)
                else float("nan")
            )
    return row


def _quantile(vals: list[object], q: float) -> float:
    arr = np.array(
        [v for v in vals if isinstance(v, (int, float)) and np.isfinite(v)],
        dtype=float,
    )
    if arr.size == 0:
        return float("nan")
    return float(np.quantile(arr, q))


def _fmt(v: float, nd: int = 3) -> str:
    if not np.isfinite(v):
        return "NA"
    return f"{v:.{nd}f}"


def summarize(rows: list[dict[str, object]]) -> str:
    ok = [r for r in rows if not r.get("error")]
    keys = [
        ("keep_I30_ratio", "Matched observed I30 post/pre"),
        ("keep_R2_EI_ratio", "Matched observed R2_EI post/pre"),
        ("keep_DEPTH_ratio", "Matched observed DEPTH post/pre"),
        ("keep_DUR_ratio", "Matched observed DUR post/pre"),
        ("keep_PEAKINT_ratio", "Matched observed PEAKINT post/pre"),
        ("keep_I30_rmse", "Matched observed I30 RMSE (mm/h)"),
        ("keep_R2_EI_rmse", "Matched observed R2_EI RMSE"),
        ("keep_DUR_rmse", "Matched observed DUR RMSE (min)"),
        ("new_vs_obs_I30_med_ratio", "New-storm median I30 / observed"),
        ("new_vs_obs_R2_EI_med_ratio", "New-storm median R2_EI / observed"),
        ("new_vs_obs_DEPTH_med_ratio", "New-storm median DEPTH / observed"),
        ("new_vs_obs_DUR_med_ratio", "New-storm median DUR / observed"),
        ("new_vs_obs_PEAKINT_med_ratio", "New-storm median PEAKINT / observed"),
        ("post_R2_EI_sum_vs_r", "Post R2_EI sum / R"),
        ("post_I30_sum_vs_r", "Post I30 sum / R"),
        ("post_DEPTH_sum_vs_r", "Post DEPTH sum / R"),
        ("pre_R2_EI_sum_vs_r", "Pre R2_EI sum / R"),
    ]
    n_new = sum(int(r.get("n_new") or 0) for r in ok)
    n_keep = sum(int(r.get("n_matched") or 0) for r in ok)
    lines = [
        "# NCDC Python fill — erosion metrics",
        "",
        "GHCN has no I30/EI. Observed storms are the preservation bench.",
        "Filled storms are scored against the same station's observed storms.",
        "R 1.7.1 storm tables are a second processor, not truth.",
        "",
        f"- Stations: **{len(ok)}** / {len(rows)} (errors: {len(rows)-len(ok)})",
        f"- Matched observed storms: **{n_keep}**",
        f"- New (filled) storms: **{n_new}**",
        "",
        "| Metric | p10 | median | p90 |",
        "|---|---:|---:|---:|",
    ]
    for key, label in keys:
        vals = [r.get(key) for r in ok]
        lines.append(
            f"| {label} | {_fmt(_quantile(vals, 0.1))} | "
            f"{_fmt(_quantile(vals, 0.5))} | {_fmt(_quantile(vals, 0.9))} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    stems = sorted(
        {p.stem for p in (PY_COMPLETE / "NPZ").glob("*.npz")}
        | {p.stem for p in (PY_RERUN / "NPZ").glob("*.npz")}
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Erosion compare on {len(stems)} stations", flush=True)
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
    csv_path = OUT_DIR / "ErosionStationMetrics.csv"
    df.to_csv(csv_path, index=False)
    summary = summarize(rows)
    (OUT_DIR / "EROSION-SUMMARY.md").write_text(summary, encoding="utf-8")
    print(f"Wrote {csv_path} ({len(df)} rows)")
    print(summary)
    err = df[df["error"].astype(str) != ""]
    if len(err):
        print(f"Errors: {len(err)}")
        print(err["error"].value_counts().head().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
