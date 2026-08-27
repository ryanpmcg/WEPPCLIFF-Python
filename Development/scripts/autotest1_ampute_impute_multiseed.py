#!/usr/bin/env python3
"""
Repeat AutoTest.1 ampute → impute evaluation across many random seeds.

Uses climate-aware scoring from ``autotest1_ampute_impute_analysis`` (not
day-matched precip/wind RMSE). Aggregates mean / std / median over N seeds
for paper tables.

Usage:
  python Development/scripts/autotest1_ampute_impute_multiseed.py
  python Development/scripts/autotest1_ampute_impute_multiseed.py --n 30 --workers 4
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Development" / "scripts"))

import autotest1_ampute_impute_analysis as aa  # noqa: E402
from Source.weppcliff_impute.adapter import ImputeConfig, impute_by_cluster  # noqa: E402
from Source.weppcliff_impute.validation import (  # noqa: E402
    WeatherScenario,
    apply_long_gap_all,
    apply_short_gaps,
    enforce_temperature_order,
)

OUT_DIR = ROOT / "Autotest" / "Output" / "AmputeImpute" / "multiseed"
PAPER_RESULTS = ROOT / "JOSS-PAPER" / "results"

# Key climate / pointwise metrics for the paper tables
PRECIP_METRICS = (
    "wet_dry_accuracy",
    "wet_f1",
    "monthly_mass_bias",
    "holdout_mass_ratio",
    "wet_intensity_ks",
)
WIND_VEL_METRICS = (
    "speed_ks",
    "speed_mean_ratio",
    "monthly_mean_speed_bias",
)
WIND_DIR_METRICS = (
    "sector_tvd",
    "monthly_circular_mean_error_deg",
)
POINTWISE_VARS = ("MIN_TEMP", "MAX_TEMP", "DP_TEMP", "SO_RAD")

SCENARIOS = (
    WeatherScenario.MCAR_AMPUTE,
    WeatherScenario.SHORT_GAPS,
    WeatherScenario.LONG_GAP_ALL,
)


def _run_one_seed_scenario(
    seed: int,
    scenario_value: str,
    panel_dict: dict,
    value_cols: list[str],
    io_iters: int,
    lat: float,
) -> dict:
    """Worker: ampute + impute + score one (seed, scenario). Returns flat metrics."""
    panel = pd.DataFrame(panel_dict)
    value_cols = list(value_cols)
    scenario = WeatherScenario(scenario_value)

    if scenario == WeatherScenario.MCAR_AMPUTE:
        masked = aa.apply_mcar_all_variables(panel, value_cols, prop=0.12, seed=seed)
    elif scenario == WeatherScenario.SHORT_GAPS:
        masked = apply_short_gaps(panel, value_cols, seed=seed)
    elif scenario == WeatherScenario.LONG_GAP_ALL:
        # Long gap is deterministic in location; still re-impute with seed-varying FCS
        masked = apply_long_gap_all(panel, value_cols)
    else:
        raise ValueError(scenario_value)

    cfg = ImputeConfig(
        pm="t", io=io_iters, qi="f", seed=seed, lat=lat, kg="Dfb"
    )
    work = masked.amputed[["CLUSTER", *value_cols]].copy()
    imputed = enforce_temperature_order(impute_by_cluster(work, "midastouch", cfg))
    pw, climate = aa.score_all(
        panel,
        imputed,
        masked.missing_mask,
        value_cols,
        scenario=scenario_value,
        panel="daily",
    )

    row: dict = {"seed": seed, "scenario": scenario_value}

    # Pointwise per-variable RMSE + mean
    rmses = []
    for var in POINTWISE_VARS:
        sub = pw[pw["variable"] == var]
        val = float(sub["rmse"].iloc[0]) if len(sub) and np.isfinite(sub["rmse"].iloc[0]) else np.nan
        row[f"pw_rmse_{var}"] = val
        if np.isfinite(val):
            rmses.append(val)
    row["pw_rmse_mean"] = float(np.mean(rmses)) if rmses else np.nan

    maes = []
    for var in POINTWISE_VARS:
        sub = pw[pw["variable"] == var]
        val = float(sub["mae"].iloc[0]) if len(sub) and np.isfinite(sub["mae"].iloc[0]) else np.nan
        row[f"pw_mae_{var}"] = val
        if np.isfinite(val):
            maes.append(val)
    row["pw_mae_mean"] = float(np.mean(maes)) if maes else np.nan

    for metric in PRECIP_METRICS:
        row[f"precip_{metric}"] = aa._metric_value(climate, "PRECIP", metric)
    for metric in WIND_VEL_METRICS:
        row[f"wvel_{metric}"] = aa._metric_value(climate, "W_VEL", metric)
    for metric in WIND_DIR_METRICS:
        row[f"wdir_{metric}"] = aa._metric_value(climate, "W_DIR", metric)

    return row


def _worker(args: tuple) -> dict:
    seed, scenario_value, panel_dict, value_cols, io_iters, lat = args
    t0 = time.perf_counter()
    try:
        row = _run_one_seed_scenario(
            seed, scenario_value, panel_dict, value_cols, io_iters, lat
        )
        row["status"] = "ok"
        row["elapsed_s"] = time.perf_counter() - t0
        return row
    except Exception as exc:  # noqa: BLE001 — collect worker failures
        return {
            "seed": seed,
            "scenario": scenario_value,
            "status": f"error: {exc}",
            "elapsed_s": time.perf_counter() - t0,
        }


def aggregate_table(replicates: pd.DataFrame) -> pd.DataFrame:
    """Mean / std / median / p05 / p95 by scenario for numeric metric columns."""
    ok = replicates[replicates["status"] == "ok"].copy()
    metric_cols = [
        c
        for c in ok.columns
        if c not in ("seed", "scenario", "status", "elapsed_s")
        and pd.api.types.is_numeric_dtype(ok[c])
    ]
    rows = []
    for scenario, g in ok.groupby("scenario"):
        for col in metric_cols:
            vals = g[col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            rows.append(
                {
                    "scenario": scenario,
                    "metric": col,
                    "n": int(vals.size),
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                    "median": float(np.median(vals)),
                    "p05": float(np.percentile(vals, 5)),
                    "p95": float(np.percentile(vals, 95)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
            )
    return pd.DataFrame(rows)


def paper_summary_wide(agg: pd.DataFrame) -> pd.DataFrame:
    """Compact wide table: scenario × selected metrics as mean±std."""
    keep = [
        "pw_rmse_mean",
        "pw_rmse_MIN_TEMP",
        "pw_rmse_MAX_TEMP",
        "pw_rmse_DP_TEMP",
        "pw_rmse_SO_RAD",
        "precip_wet_dry_accuracy",
        "precip_wet_f1",
        "precip_monthly_mass_bias",
        "precip_holdout_mass_ratio",
        "precip_wet_intensity_ks",
        "wvel_speed_ks",
        "wvel_speed_mean_ratio",
        "wvel_monthly_mean_speed_bias",
        "wdir_sector_tvd",
        "wdir_monthly_circular_mean_error_deg",
    ]
    rows = []
    for scenario in agg["scenario"].unique():
        row: dict = {"scenario": scenario}
        for metric in keep:
            sub = agg[(agg["scenario"] == scenario) & (agg["metric"] == metric)]
            if sub.empty:
                row[metric] = ""
                row[f"{metric}_std"] = ""
                continue
            row[metric] = float(sub["mean"].iloc[0])
            row[f"{metric}_std"] = float(sub["std"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def format_mean_std_table(wide: pd.DataFrame) -> pd.DataFrame:
    """Human-readable mean ± std columns for markdown/paper."""
    display_map = {
        "pw_rmse_mean": "Pointwise RMSE (mean of 4 vars)",
        "pw_rmse_MIN_TEMP": "MIN_TEMP RMSE (°C)",
        "pw_rmse_MAX_TEMP": "MAX_TEMP RMSE (°C)",
        "pw_rmse_DP_TEMP": "DP_TEMP RMSE (°C)",
        "pw_rmse_SO_RAD": "SO_RAD RMSE (Ly)",
        "precip_wet_dry_accuracy": "PRECIP wet/dry accuracy",
        "precip_wet_f1": "PRECIP wet-day F1",
        "precip_monthly_mass_bias": "PRECIP monthly mass bias",
        "precip_holdout_mass_ratio": "PRECIP holdout mass ratio",
        "precip_wet_intensity_ks": "PRECIP wet-day intensity KS",
        "wvel_speed_ks": "W_VEL speed KS",
        "wvel_speed_mean_ratio": "W_VEL mean ratio",
        "wvel_monthly_mean_speed_bias": "W_VEL monthly mean bias",
        "wdir_sector_tvd": "W_DIR sector TVD",
        "wdir_monthly_circular_mean_error_deg": "W_DIR monthly circ. mean err (°)",
    }
    out_rows = []
    for _, wrow in wide.iterrows():
        scen = wrow["scenario"]
        for key, label in display_map.items():
            m = wrow.get(key)
            s = wrow.get(f"{key}_std")
            if m == "" or m is None or (isinstance(m, float) and not np.isfinite(m)):
                cell = "—"
            else:
                cell = f"{float(m):.3f} ± {float(s):.3f}"
            out_rows.append({"scenario": scen, "metric": label, "mean_pm_std": cell})
    return pd.DataFrame(out_rows)


def to_markdown_pivot(fmt: pd.DataFrame) -> str:
    """Pivot scenario columns for a compact paper table."""
    pivot = fmt.pivot(index="metric", columns="scenario", values="mean_pm_std")
    # Prefer column order
    order = [c for c in ("mcar_ampute", "short_gaps", "long_gap_all") if c in pivot.columns]
    pivot = pivot[order]
    lines = [
        "| Metric | " + " | ".join(order) + " |",
        "| --- | " + " | ".join("---" for _ in order) + " |",
    ]
    for metric, row in pivot.iterrows():
        cells = [str(row[c]) if pd.notna(row[c]) else "—" for c in order]
        lines.append("| " + str(metric) + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=30, help="Number of random seeds")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument(
        "--io",
        type=int,
        default=20,
        help="PyMICE iteration override (paper multi-seed default 20)",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Disable process pool (debug)",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_RESULTS.mkdir(parents=True, exist_ok=True)

    print(f"Loading daily panel from {aa.DAILY_CSV} ...")
    truth = aa.load_daily_panel()
    value_cols = [c for c in aa.DAILY_VALUE_COLS if c in truth.columns]
    panel = aa.attach_clusters(truth, value_cols, lat=aa.LAT)
    # Serialize for process workers
    panel_dict = panel.to_dict(orient="list")

    seeds = list(range(args.seed_start, args.seed_start + args.n))
    jobs = [
        (seed, scen.value, panel_dict, value_cols, args.io, aa.LAT)
        for seed in seeds
        for scen in SCENARIOS
    ]
    print(
        f"Running {len(jobs)} jobs "
        f"({args.n} seeds × {len(SCENARIOS)} scenarios, io={args.io}, "
        f"workers={1 if args.serial else args.workers}) ..."
    )
    t0 = time.perf_counter()
    results: list[dict] = []

    if args.serial:
        for i, job in enumerate(jobs, 1):
            row = _worker(job)
            results.append(row)
            print(
                f"  [{i}/{len(jobs)}] seed={row['seed']} {row['scenario']} "
                f"{row['status']} ({row.get('elapsed_s', 0):.1f}s)"
            )
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, job): job for job in jobs}
            for fut in as_completed(futures):
                row = fut.result()
                results.append(row)
                done += 1
                if done % 5 == 0 or done == len(jobs):
                    print(
                        f"  progress {done}/{len(jobs)} "
                        f"(last seed={row['seed']} {row['scenario']} "
                        f"{row['status']})"
                    )

    replicates = pd.DataFrame(results).sort_values(["scenario", "seed"])
    rep_path = OUT_DIR / "replicates.csv"
    replicates.to_csv(rep_path, index=False)
    replicates.to_csv(PAPER_RESULTS / "multiseed_replicates.csv", index=False)
    print(f"Wrote {rep_path}")

    n_ok = int((replicates["status"] == "ok").sum())
    n_err = int(len(replicates) - n_ok)
    print(f"Completed: {n_ok} ok, {n_err} errors in {time.perf_counter() - t0:.1f}s")

    agg = aggregate_table(replicates)
    agg_path = OUT_DIR / "aggregate_mean_std.csv"
    agg.to_csv(agg_path, index=False)
    agg.to_csv(PAPER_RESULTS / "multiseed_aggregate.csv", index=False)

    wide = paper_summary_wide(agg)
    wide.to_csv(OUT_DIR / "paper_wide_mean_std.csv", index=False)
    wide.to_csv(PAPER_RESULTS / "paper_wide_mean_std.csv", index=False)

    fmt = format_mean_std_table(wide)
    fmt.to_csv(OUT_DIR / "paper_table_mean_pm_std.csv", index=False)
    fmt.to_csv(PAPER_RESULTS / "paper_table_mean_pm_std.csv", index=False)

    md = to_markdown_pivot(fmt)
    md_path = OUT_DIR / "TABLE_multiseed_summary.md"
    md_full = (
        f"# AutoTest.1 multiseed ampute/impute (N={args.n})\n\n"
        f"Seeds {seeds[0]}–{seeds[-1]}; PyMICE midastouch; io={args.io}; "
        f"climate-aware scoring for PRECIP/wind.\n\n"
        f"{md}\n\n"
        f"Full replicates: `{rep_path.relative_to(ROOT)}`\n"
    )
    md_path.write_text(md_full, encoding="utf-8")
    (PAPER_RESULTS / "TABLE_multiseed_summary.md").write_text(md_full, encoding="utf-8")
    print(md)
    print(f"Wrote {md_path}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
