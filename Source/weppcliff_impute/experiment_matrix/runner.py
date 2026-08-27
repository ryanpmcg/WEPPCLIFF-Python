"""Execute experiment matrix jobs and collect dual-scope metrics."""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Source.weppcliff_impute.experiment_matrix.catalog import (
    EXPERIMENT_CATALOG,
    ExperimentSpec,
    resolve_seed_count,
)
from Source.weppcliff_impute.experiment_matrix.engines import (
    EngineName,
    attach_clusters,
    run_panel_engine,
    run_workflow_engine,
)
from Source.weppcliff_impute.experiment_matrix.scenarios import apply_matrix_scenario
from Source.weppcliff_impute.experiment_matrix.scoring import (
    aggregate_metric_frame,
    score_breakpoint_panel,
    score_dual_panel,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DAILY = ROOT / "Autotest" / "Output" / "AutoTest.1-Daily-Time-Series.csv"
DEFAULT_PRECIP = ROOT / "Autotest" / "Output" / "AutoTest.1-Precipitation.csv"
DEFAULT_KMQE = ROOT / "Input" / "ASOS_BPT_KMQE.csv"
DEFAULT_OUT = ROOT / "Autotest" / "Output" / "ExperimentMatrix"

DAILY_COMPLETE = ("MIN_TEMP", "MAX_TEMP", "SO_RAD", "W_VEL", "W_DIR", "DP_TEMP")
DAILY_PARTIAL = ("PRECIP", "BPS", "ELEN", "EMAX", "EMIN")
DAILY_VALUE_COLS = DAILY_COMPLETE + DAILY_PARTIAL
BP_VALUE_COLS = ("PRECIP", "DUR", "INT", "ANT", "BTW")


@dataclass
class MatrixRunConfig:
    """Runtime knobs for matrix execution."""

    experiments: tuple[str, ...] = ("A", "B", "C", "D", "E", "G")
    profile: str = "default"  # smoke | default | paper
    seed_start: int = 0
    n_seeds_override: int | None = None
    workers: int = 2
    serial: bool = False
    io: int = 15
    lat: float = 42.2
    kg: str = "Dfb"
    daily_csv: Path = DEFAULT_DAILY
    precip_csv: Path = DEFAULT_PRECIP
    kmqe_csv: Path = DEFAULT_KMQE
    out_dir: Path = DEFAULT_OUT
    include_workflow: bool = False
    """If True, also run experiment F (slow full fill_workflow)."""
    max_breakpoint_rows: int = 8000
    gates: bool = True
    """Emit CI gate pass/fail table after aggregation."""
    engines_override: tuple[str, ...] | None = None
    """If set, replace each experiment's engine list (for fast smoke)."""
    scenarios_override: tuple[str, ...] | None = None
    """If set, replace each experiment's scenario list."""


def load_daily_panel(path: Path = DEFAULT_DAILY) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing daily export {path}. Run AutoTest.1 with -ed 6 first."
        )
    df = pd.read_csv(path)
    if "Omitted-Storms" in df.columns:
        df = df.drop(columns=["Omitted-Storms"])
    df["YYYYMMDD"] = df["YYYYMMDD"].astype(str).str.replace(r"\.0$", "", regex=True)
    return df.reset_index(drop=True)


def load_precip_panel(path: Path = DEFAULT_PRECIP, max_rows: int = 8000) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing precip export {path}")
    df = pd.read_csv(path)
    df = df.head(max_rows).copy()
    df["YYYYMMDD"] = pd.to_datetime(df["DT_1"], errors="coerce").dt.strftime("%Y%m%d")
    return df.reset_index(drop=True)


def _value_cols_present(df: pd.DataFrame, cols: tuple[str, ...]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _run_daily_job(payload: dict[str, Any]) -> list[dict]:
    """Worker: one (experiment, scenario, engine, seed) on daily panel."""
    t0 = time.perf_counter()
    experiment = payload["experiment"]
    scenario = payload["scenario"]
    engine = payload["engine"]
    seed = int(payload["seed"])
    io = int(payload["io"])
    lat = float(payload["lat"])
    kg = payload["kg"]
    value_cols = list(payload["value_cols"])
    panel = pd.DataFrame(payload["panel_dict"])
    try:
        truth = panel
        applied = apply_matrix_scenario(
            truth,
            scenario,
            value_cols,
            seed=seed,
            complete_cols=DAILY_COMPLETE,
        )
        # Ensure CLUSTER on truth for pre-cluster engines
        if "CLUSTER" not in truth.columns:
            truth = attach_clusters(truth, value_cols, lat=lat)
        amputed = applied.amputed.copy()
        if "CLUSTER" not in amputed.columns:
            amputed["CLUSTER"] = truth["CLUSTER"].to_numpy()

        eng = run_panel_engine(
            engine,
            truth,
            amputed,
            value_cols,
            seed=seed,
            lat=lat,
            io=io,
            kg=kg,
        )
        metrics = score_dual_panel(
            truth,
            eng.imputed,
            applied.missing_mask,
            applied.hole_day,
            value_cols,
            experiment=experiment,
            engine=engine,
            scenario=scenario,
            seed=seed,
            panel="daily",
        )
        rows = metrics.to_dict(orient="records")
        # Job meta row
        rows.append(
            {
                "experiment": experiment,
                "engine": engine,
                "scenario": scenario,
                "seed": seed,
                "variable": "_JOB",
                "metric": "status",
                "value": 1.0,
                "scope": "meta",
                "period": "job",
                "metric_family": "meta",
                "n": 1,
                "detail": "ok",
                "panel": "daily",
                "elapsed_s": time.perf_counter() - t0,
                "hole_day_frac": float(np.mean(applied.hole_day)) if len(applied.hole_day) else 0.0,
                "engine_notes": ";".join(eng.notes),
            }
        )
        return rows
    except Exception as exc:  # noqa: BLE001
        return [
            {
                "experiment": experiment,
                "engine": engine,
                "scenario": scenario,
                "seed": seed,
                "variable": "_JOB",
                "metric": "status",
                "value": 0.0,
                "scope": "meta",
                "period": "job",
                "metric_family": "meta",
                "n": 0,
                "detail": f"error: {exc}",
                "panel": "daily",
                "elapsed_s": time.perf_counter() - t0,
                "hole_day_frac": float("nan"),
                "engine_notes": "",
            }
        ]


def _run_breakpoint_job(payload: dict[str, Any]) -> list[dict]:
    t0 = time.perf_counter()
    experiment = payload["experiment"]
    scenario = payload["scenario"]
    engine = payload["engine"]
    seed = int(payload["seed"])
    io = int(payload["io"])
    lat = float(payload["lat"])
    kg = payload["kg"]
    value_cols = list(payload["value_cols"])
    panel = pd.DataFrame(payload["panel_dict"])
    try:
        applied = apply_matrix_scenario(panel, scenario, value_cols, seed=seed)
        # CLUSTER by date bucket for breakpoint panel
        work = applied.amputed.copy()
        if "YYYYMMDD" in work.columns:
            # coarse cluster: month
            work["CLUSTER"] = work["YYYYMMDD"].astype(str).str.slice(4, 6).astype(int)
        else:
            work["CLUSTER"] = 1
        truth = panel.copy()
        if "CLUSTER" not in truth.columns:
            truth["CLUSTER"] = work["CLUSTER"].to_numpy()
        eng = run_panel_engine(
            engine,
            truth,
            work,
            value_cols,
            seed=seed,
            lat=lat,
            io=io,
            kg=kg,
        )
        metrics = score_breakpoint_panel(
            truth if "DT_1" in truth.columns else panel,
            eng.imputed,
            applied.missing_mask,
            value_cols,
            experiment=experiment,
            engine=engine,
            scenario=scenario,
            seed=seed,
        )
        # Also dual mass on daily-aggregated PRECIP if possible
        rows = metrics
        rows.append(
            {
                "experiment": experiment,
                "engine": engine,
                "scenario": scenario,
                "seed": seed,
                "variable": "_JOB",
                "metric": "status",
                "value": 1.0,
                "scope": "meta",
                "period": "job",
                "metric_family": "meta",
                "n": 1,
                "detail": "ok",
                "panel": "breakpoint",
                "elapsed_s": time.perf_counter() - t0,
                "hole_day_frac": float(np.mean(applied.hole_day)) if len(applied.hole_day) else 0.0,
                "engine_notes": ";".join(eng.notes),
            }
        )
        return rows
    except Exception as exc:  # noqa: BLE001
        return [
            {
                "experiment": experiment,
                "engine": engine,
                "scenario": scenario,
                "seed": seed,
                "variable": "_JOB",
                "metric": "status",
                "value": 0.0,
                "scope": "meta",
                "period": "job",
                "metric_family": "meta",
                "n": 0,
                "detail": f"error: {exc}",
                "panel": "breakpoint",
                "elapsed_s": time.perf_counter() - t0,
                "hole_day_frac": float("nan"),
                "engine_notes": "",
            }
        ]


def _run_workflow_job(payload: dict[str, Any]) -> list[dict]:
    t0 = time.perf_counter()
    experiment = payload["experiment"]
    scenario = payload["scenario"]
    engine = payload["engine"]
    seed = int(payload["seed"])
    input_path = Path(payload["kmqe"])
    try:
        flat = run_workflow_engine(
            engine,
            input_path,
            seed=seed,
            holdout_frac=float(payload.get("holdout_frac", 0.10)),
            scenario=scenario,
        )
        rows = []
        for key, val in flat.items():
            if key in ("engine", "scenario", "seed", "status", "notes", "winners"):
                continue
            if isinstance(val, (int, float, np.floating)):
                rows.append(
                    {
                        "experiment": experiment,
                        "engine": engine,
                        "scenario": scenario,
                        "seed": seed,
                        "variable": key.split("_", 1)[-1] if key.startswith("rmse_") else "WORKFLOW",
                        "metric": key,
                        "value": float(val) if np.isfinite(float(val)) else float("nan"),
                        "scope": "holdout_cells",
                        "period": "daily",
                        "metric_family": "workflow",
                        "n": int(flat.get("n_holdout_days", 0) or 0),
                        "detail": flat.get("notes", ""),
                        "panel": "workflow",
                    }
                )
        rows.append(
            {
                "experiment": experiment,
                "engine": engine,
                "scenario": scenario,
                "seed": seed,
                "variable": "_JOB",
                "metric": "status",
                "value": 1.0 if flat.get("status") == "ok" else 0.0,
                "scope": "meta",
                "period": "job",
                "metric_family": "meta",
                "n": 1,
                "detail": flat.get("notes", "ok"),
                "panel": "workflow",
                "elapsed_s": time.perf_counter() - t0,
                "hole_day_frac": float("nan"),
                "engine_notes": "",
            }
        )
        return rows
    except Exception as exc:  # noqa: BLE001
        return [
            {
                "experiment": experiment,
                "engine": engine,
                "scenario": scenario,
                "seed": seed,
                "variable": "_JOB",
                "metric": "status",
                "value": 0.0,
                "scope": "meta",
                "period": "job",
                "metric_family": "meta",
                "n": 0,
                "detail": f"error: {exc}",
                "panel": "workflow",
                "elapsed_s": time.perf_counter() - t0,
                "hole_day_frac": float("nan"),
                "engine_notes": "",
            }
        ]


def _build_jobs(
    config: MatrixRunConfig,
    daily: pd.DataFrame | None,
    precip: pd.DataFrame | None,
) -> list[tuple[str, dict[str, Any]]]:
    """Return list of (job_type, payload)."""
    jobs: list[tuple[str, dict[str, Any]]] = []
    exp_ids = list(config.experiments)
    if config.include_workflow and "F" not in exp_ids:
        exp_ids.append("F")

    for eid in exp_ids:
        if eid not in EXPERIMENT_CATALOG:
            raise ValueError(f"unknown experiment {eid}")
        spec = EXPERIMENT_CATALOG[eid]
        n = (
            config.n_seeds_override
            if config.n_seeds_override is not None
            else resolve_seed_count(spec, config.profile)
        )
        if n <= 0:
            continue
        seeds = list(range(config.seed_start, config.seed_start + n))

        engines = config.engines_override or spec.engines
        scenarios = config.scenarios_override or spec.scenarios

        if spec.panel == "daily":
            assert daily is not None
            value_cols = _value_cols_present(daily, DAILY_VALUE_COLS)
            truth = attach_clusters(daily, value_cols, lat=config.lat)
            panel_dict = truth.to_dict(orient="list")
            for seed in seeds:
                for scenario in scenarios:
                    for engine in engines:
                        jobs.append(
                            (
                                "daily",
                                {
                                    "experiment": eid,
                                    "scenario": scenario,
                                    "engine": engine,
                                    "seed": seed,
                                    "io": config.io,
                                    "lat": config.lat,
                                    "kg": config.kg,
                                    "value_cols": value_cols,
                                    "panel_dict": panel_dict,
                                },
                            )
                        )
        elif spec.panel == "breakpoint":
            if precip is None:
                continue
            value_cols = _value_cols_present(precip, BP_VALUE_COLS)
            panel_dict = precip.to_dict(orient="list")
            for seed in seeds:
                for scenario in scenarios:
                    for engine in engines:
                        jobs.append(
                            (
                                "breakpoint",
                                {
                                    "experiment": eid,
                                    "scenario": scenario,
                                    "engine": engine,
                                    "seed": seed,
                                    "io": config.io,
                                    "lat": config.lat,
                                    "kg": config.kg,
                                    "value_cols": value_cols,
                                    "panel_dict": panel_dict,
                                },
                            )
                        )
        elif spec.panel == "workflow":
            if not config.kmqe_csv.exists():
                continue
            wf_scen = config.scenarios_override or spec.workflow_scenarios
            for seed in seeds:
                for scenario in wf_scen:
                    for engine in engines:
                        jobs.append(
                            (
                                "workflow",
                                {
                                    "experiment": eid,
                                    "scenario": scenario,
                                    "engine": engine,
                                    "seed": seed,
                                    "kmqe": str(config.kmqe_csv),
                                    "holdout_frac": 0.10,
                                },
                            )
                        )
    return jobs


def _dispatch(job_type: str, payload: dict[str, Any]) -> list[dict]:
    if job_type == "daily":
        return _run_daily_job(payload)
    if job_type == "breakpoint":
        return _run_breakpoint_job(payload)
    if job_type == "workflow":
        return _run_workflow_job(payload)
    raise ValueError(job_type)


def evaluate_gates(agg: pd.DataFrame, replicates: pd.DataFrame) -> pd.DataFrame:
    """
    CI / assessment gates (recommended thresholds).

    Returns a DataFrame of gate_id, description, passed, observed, threshold.
    """
    gates: list[dict] = []

    def _mean(experiment: str, engine: str, scenario: str, variable: str, metric: str, scope: str, period: str) -> float:
        if agg.empty:
            return float("nan")
        mask = pd.Series(True, index=agg.index)
        for col, val in (
            ("experiment", experiment),
            ("engine", engine),
            ("scenario", scenario),
            ("variable", variable),
            ("metric", metric),
            ("scope", scope),
            ("period", period),
        ):
            if col not in agg.columns:
                return float("nan")
            mask &= agg[col] == val
        sub = agg.loc[mask]
        if sub.empty:
            return float("nan")
        return float(sub["mean"].iloc[0])

    # Gate A1: MCAR wet F1
    v = _mean("A", "midastouch", "mcar_ampute", "PRECIP", "wet_f1", "holdout_cells", "daily")
    gates.append(
        {
            "gate_id": "A1_mcar_wet_f1",
            "description": "MCAR midastouch wet F1 ≥ 0.95",
            "observed": v,
            "threshold": 0.95,
            "passed": bool(np.isfinite(v) and v >= 0.95),
        }
    )
    # Gate A2: MCAR hole mass ratio near 1
    v = _mean("A", "midastouch", "mcar_ampute", "PRECIP", "mass_ratio", "hole", "daily")
    gates.append(
        {
            "gate_id": "A2_mcar_hole_mass",
            "description": "MCAR midastouch hole daily mass ratio in [0.75, 1.25]",
            "observed": v,
            "threshold": "0.75–1.25",
            "passed": bool(np.isfinite(v) and 0.75 <= v <= 1.25),
        }
    )
    # Gate C1: long-gap EMAX mean_ratio not explosive under midastouch (document failure if fails)
    for scen in ("long_gap_DJF", "long_gap_MAM", "long_gap_JJA", "long_gap_SON"):
        v = _mean("C", "midastouch", scen, "EMAX", "mean_ratio", "holdout_cells", "sample")
        gates.append(
            {
                "gate_id": f"C_emax_{scen}",
                "description": f"EMAX mean_ratio < 5 for {scen} (known weak; track)",
                "observed": v,
                "threshold": 5.0,
                "passed": bool(not np.isfinite(v) or v < 5.0),
            }
        )
    # Gate D1: SO_RAD variable-absent RMSE should be finite (engine runs)
    v = _mean("D", "midastouch", "variable_absent:SO_RAD", "SO_RAD", "rmse", "holdout_cells", "daily")
    gates.append(
        {
            "gate_id": "D1_sorad_absent_runs",
            "description": "SO_RAD variable-absent produces finite RMSE",
            "observed": v,
            "threshold": "finite",
            "passed": bool(np.isfinite(v)),
        }
    )
    # Job success rate
    if not replicates.empty and "metric" in replicates.columns:
        jobs = replicates[(replicates["variable"] == "_JOB") & (replicates["metric"] == "status")]
        if len(jobs):
            rate = float(jobs["value"].mean())
            gates.append(
                {
                    "gate_id": "Z_job_success_rate",
                    "description": "Job success rate ≥ 0.95",
                    "observed": rate,
                    "threshold": 0.95,
                    "passed": rate >= 0.95,
                }
            )
    return pd.DataFrame(gates)


def run_experiment(
    experiment_id: str,
    config: MatrixRunConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a single experiment id; returns (replicates_long, aggregate)."""
    cfg = config or MatrixRunConfig(experiments=(experiment_id,))
    cfg.experiments = (experiment_id,)
    return run_matrix(cfg)[:2]


def run_matrix(
    config: MatrixRunConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    """
    Run the configured experiment matrix.

    Returns (replicates, aggregate, gates, out_dir).
    """
    cfg = config or MatrixRunConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    daily = None
    precip = None
    need_daily = any(
        EXPERIMENT_CATALOG[e].panel == "daily"
        for e in cfg.experiments
        if e in EXPERIMENT_CATALOG
    )
    need_bp = any(
        EXPERIMENT_CATALOG[e].panel == "breakpoint"
        for e in cfg.experiments
        if e in EXPERIMENT_CATALOG
    ) or (cfg.include_workflow is False and "E" in cfg.experiments)

    if need_daily or any(e in cfg.experiments for e in "ABCDG"):
        daily = load_daily_panel(cfg.daily_csv)
    if need_bp or "E" in cfg.experiments:
        try:
            precip = load_precip_panel(cfg.precip_csv, max_rows=cfg.max_breakpoint_rows)
        except FileNotFoundError:
            precip = None

    jobs = _build_jobs(cfg, daily, precip)
    if not jobs:
        empty = pd.DataFrame()
        return empty, empty, empty, cfg.out_dir

    all_rows: list[dict] = []
    t0 = time.perf_counter()
    if cfg.serial or cfg.workers <= 1:
        for i, (jtype, payload) in enumerate(jobs, 1):
            rows = _dispatch(jtype, payload)
            all_rows.extend(rows)
            meta = [r for r in rows if r.get("variable") == "_JOB"]
            status = meta[0].get("detail", "") if meta else ""
            print(
                f"  [{i}/{len(jobs)}] {payload.get('experiment')} "
                f"{payload.get('scenario')} {payload.get('engine')} "
                f"seed={payload.get('seed')} {status}",
                flush=True,
            )
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            futs = {
                pool.submit(_dispatch, jtype, payload): (jtype, payload)
                for jtype, payload in jobs
            }
            for fut in as_completed(futs):
                rows = fut.result()
                all_rows.extend(rows)
                done += 1
                if done % 5 == 0 or done == len(jobs):
                    print(f"  progress {done}/{len(jobs)}", flush=True)

    replicates = pd.DataFrame(all_rows)
    rep_path = cfg.out_dir / "replicates_long.csv"
    replicates.to_csv(rep_path, index=False)

    # Drop meta for aggregation of science metrics
    sci = replicates[replicates.get("metric_family", pd.Series(dtype=str)) != "meta"].copy() if not replicates.empty else replicates
    if not replicates.empty and "metric_family" in replicates.columns:
        sci = replicates[replicates["metric_family"] != "meta"].copy()
    agg = aggregate_metric_frame(sci)
    agg_path = cfg.out_dir / "aggregate_mean_std.csv"
    agg.to_csv(agg_path, index=False)

    gates = evaluate_gates(agg, replicates) if cfg.gates else pd.DataFrame()
    if not gates.empty:
        gates.to_csv(cfg.out_dir / "gates.csv", index=False)

    elapsed = time.perf_counter() - t0
    print(f"Matrix complete: {len(jobs)} jobs, {len(replicates)} metric rows, {elapsed:.1f}s")
    print(f"Wrote {rep_path}")
    return replicates, agg, gates, cfg.out_dir
