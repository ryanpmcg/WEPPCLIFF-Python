"""Imputation engines used by the experiment matrix."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd

from Source.weppcliff_impute.adapter import ImputeConfig, impute_by_cluster
from Source.weppcliff_impute.cluster import assign_cluster_labels
from Source.weppcliff_impute.validation import enforce_temperature_order


class EngineName(str, Enum):
    MEAN = "mean"
    MIDASTOUCH = "midastouch"
    CART = "cart"
    MIDASTOUCH_POST_CLUSTER = "midastouch_post_cluster"
    """Cluster labels assigned *after* amputation (realistic donor structure)."""
    JOMO = "jomo"
    STORM_TEMPLATE_FCS = "storm_template_fcs"
    """Sub-daily FCS path with storm templates when available."""
    WORKFLOW_LEGACY = "workflow_legacy"
    WORKFLOW_PHIGI = "workflow_phigi"


@dataclass
class EngineResult:
    imputed: pd.DataFrame
    engine: str
    notes: tuple[str, ...] = ()
    cluster_mode: str = "pre"  # pre | post | none | workflow


def _numeric_cols(df: pd.DataFrame, value_cols: list[str]) -> list[str]:
    return [c for c in value_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def attach_clusters(
    df: pd.DataFrame,
    value_cols: list[str],
    *,
    lat: float = 42.2,
) -> pd.DataFrame:
    """Assign CLUSTER from observed (median-filled) matrix — pre-ampute optimistic mode."""
    out = df.copy()
    ymd = out["YYYYMMDD"].astype(str).tolist()
    cols = _numeric_cols(out, value_cols)
    matrix = np.array(out[cols].to_numpy(dtype=np.float64), copy=True)
    col_med = np.nanmedian(matrix, axis=0)
    for j in range(matrix.shape[1]):
        miss = np.isnan(matrix[:, j])
        if miss.any():
            matrix[miss, j] = col_med[j] if np.isfinite(col_med[j]) else 0.0
    out["CLUSTER"] = assign_cluster_labels(ymd, matrix, lat=lat, n_vars=len(cols))
    return out


def attach_clusters_post_ampute(
    amputed: pd.DataFrame,
    value_cols: list[str],
    *,
    lat: float = 42.2,
) -> pd.DataFrame:
    """
    Assign CLUSTER using only information available after amputation.

    Median fill is still required for the clustering matrix, but the amputed
    holes remain in the working frame for imputation — labels reflect incomplete data.
    """
    return attach_clusters(amputed, value_cols, lat=lat)


def run_panel_engine(
    engine: str | EngineName,
    truth_with_clusters: pd.DataFrame,
    amputed: pd.DataFrame,
    value_cols: list[str],
    *,
    seed: int = 0,
    lat: float = 42.2,
    io: int = 20,
    kg: str = "Dfb",
) -> EngineResult:
    """
    Run a daily-panel engine.

    ``truth_with_clusters`` is the complete panel with pre-assigned CLUSTER (for
    pre-cluster engines). ``amputed`` has introduced NAs.
    """
    name = engine.value if isinstance(engine, EngineName) else str(engine)
    cols = _numeric_cols(amputed, value_cols)

    if name == EngineName.MEAN.value:
        work = amputed.copy()
        if "CLUSTER" not in work.columns:
            if "CLUSTER" in truth_with_clusters.columns:
                work["CLUSTER"] = truth_with_clusters["CLUSTER"].to_numpy()
            elif "YYYYMMDD" in work.columns:
                work = attach_clusters(work, value_cols, lat=lat)
            else:
                work["CLUSTER"] = 1
        work = work[["CLUSTER", *cols]].copy()
        cfg = ImputeConfig(pm="f", io=io, seed=seed, lat=lat, kg=kg)
        out = enforce_temperature_order(impute_by_cluster(work, "midastouch", cfg))
        merged = amputed.copy()
        for c in cols:
            merged[c] = out[c].to_numpy()
        return EngineResult(imputed=merged, engine=name, cluster_mode="pre", notes=("pm=f mean fill",))

    if name in (
        EngineName.MIDASTOUCH.value,
        EngineName.CART.value,
        EngineName.JOMO.value,
        EngineName.MIDASTOUCH_POST_CLUSTER.value,
    ):
        method = "cart" if name == EngineName.CART.value else "midastouch"
        post = name == EngineName.MIDASTOUCH_POST_CLUSTER.value
        if post:
            work_base = attach_clusters_post_ampute(amputed, value_cols, lat=lat)
            cluster_mode = "post"
        else:
            work_base = amputed.copy()
            if "CLUSTER" not in work_base.columns:
                work_base["CLUSTER"] = truth_with_clusters["CLUSTER"].to_numpy()
            cluster_mode = "pre"
        work = work_base[["CLUSTER", *cols]].copy()
        cfg = ImputeConfig(
            pm="t",
            io=io,
            seed=seed,
            lat=lat,
            kg=kg,
            jm="t" if name == EngineName.JOMO.value else "f",
            st="f",
        )
        out = enforce_temperature_order(impute_by_cluster(work, method, cfg))
        merged = amputed.copy()
        for c in cols:
            merged[c] = out[c].to_numpy()
        notes = (f"method={method}", f"io={io}", f"jm={cfg.jm}")
        return EngineResult(imputed=merged, engine=name, cluster_mode=cluster_mode, notes=notes)

    if name == EngineName.STORM_TEMPLATE_FCS.value:
        # Sub-daily: try storm-template path when DT_1 present; else midastouch
        work_base = amputed.copy()
        if "CLUSTER" not in work_base.columns and "YYYYMMDD" in work_base.columns:
            work_base = attach_clusters_post_ampute(work_base, value_cols, lat=lat)
        elif "CLUSTER" not in work_base.columns and "CLUSTER" in truth_with_clusters.columns:
            work_base["CLUSTER"] = truth_with_clusters["CLUSTER"].to_numpy()
        work = work_base[["CLUSTER", *cols]].copy() if "CLUSTER" in work_base.columns else work_base[cols].copy()
        if "CLUSTER" not in work.columns:
            work.insert(0, "CLUSTER", 1)
        cfg = ImputeConfig(pm="t", io=io, seed=seed, lat=lat, kg=kg, st="t")
        try:
            from Source.weppcliff_impute.subdaily import impute_subdaily_timeseries

            out = impute_subdaily_timeseries(work_base, cols, cfg)
            notes = ("subdaily+storm_template",)
        except Exception:
            out = enforce_temperature_order(impute_by_cluster(work, "midastouch", cfg))
            notes = ("fallback_midastouch_st_flag",)
        merged = amputed.copy()
        for c in cols:
            if c in out.columns:
                merged[c] = out[c].to_numpy()
        return EngineResult(imputed=merged, engine=name, cluster_mode="post", notes=notes)

    raise ValueError(f"unknown panel engine: {name}")


def run_workflow_engine(
    engine: str | EngineName,
    input_path: Path,
    *,
    seed: int = 0,
    holdout_frac: float = 0.10,
    scenario: str = "holdout_random",
) -> dict:
    """
    Full ``fill_workflow`` compare path (experiment F / optional PHIGI).

    Returns a flat metric dict from ``fair_compare`` (not a daily panel).
    """
    name = engine.value if isinstance(engine, EngineName) else str(engine)
    from Source.phigi.fair_compare import (
        CompareScenario,
        compare_engines_on_kmqe,
    )

    scen_map = {
        "holdout_random": CompareScenario.HOLDOUT_RANDOM,
        "sorad_corrupt_summer": CompareScenario.SORAD_CORRUPT_SUMMER,
        "seasonal_gap": CompareScenario.SEASONAL_GAP,
        "sparse_donors": CompareScenario.SPARSE_DONORS,
    }
    compare_scen = scen_map.get(scenario, CompareScenario.HOLDOUT_RANDOM)
    # compare_engines_on_kmqe always runs both engines; we extract the requested side
    result = compare_engines_on_kmqe(
        input_path,
        compare_scen,
        fraction=holdout_frac,
        seed=seed,
    )
    score = result.phigi if name == EngineName.WORKFLOW_PHIGI.value else result.legacy
    flat: dict = {
        "engine": name,
        "scenario": result.scenario,
        "seed": seed,
        "n_holdout_days": score.n_holdout_days,
        "n_wet_holdout_days": score.n_wet_holdout_days,
        "i30_mean_rel_bias": score.i30_mean_rel_bias,
        "ei_mean_rel_bias": score.ei_mean_rel_bias,
        "temp_order_violations": score.temp_order_violations,
        "winners": result.winners,
    }
    for var, rmse in score.daily_rmse.items():
        flat[f"rmse_{var}"] = rmse
    for k, v in score.precip.items():
        flat[f"precip_{k}"] = v
    flat["status"] = "ok"
    flat["notes"] = ";".join(score.notes + result.notes)
    return flat


# Registry for discovery / docs
PANEL_ENGINES: tuple[str, ...] = (
    EngineName.MEAN.value,
    EngineName.MIDASTOUCH.value,
    EngineName.CART.value,
    EngineName.MIDASTOUCH_POST_CLUSTER.value,
    EngineName.JOMO.value,
    EngineName.STORM_TEMPLATE_FCS.value,
)

WORKFLOW_ENGINES: tuple[str, ...] = (
    EngineName.WORKFLOW_LEGACY.value,
    EngineName.WORKFLOW_PHIGI.value,
)

ENGINE_DESCRIPTIONS: dict[str, str] = {
    EngineName.MEAN.value: "Cluster-wise mean fill (`-pm` off)",
    EngineName.MIDASTOUCH.value: "PyMICE midastouch with pre-ampute clusters (paper baseline)",
    EngineName.CART.value: "PyMICE cart FCS ablation",
    EngineName.MIDASTOUCH_POST_CLUSTER.value: "midastouch with clusters assigned after amputation",
    EngineName.JOMO.value: "Daily JOMO/PAN block (`-jm`)",
    EngineName.STORM_TEMPLATE_FCS.value: "Sub-daily path with storm templates (`-st`)",
    EngineName.WORKFLOW_LEGACY.value: "Full fill_workflow `-gf legacy`",
    EngineName.WORKFLOW_PHIGI.value: "Full fill_workflow `-gf phigi`",
}
