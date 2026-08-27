"""Unit + smoke tests for the ampute–impute experiment matrix."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Source.weppcliff_impute.experiment_matrix.catalog import EXPERIMENT_CATALOG, list_experiments
from Source.weppcliff_impute.experiment_matrix.engines import attach_clusters, run_panel_engine
from Source.weppcliff_impute.experiment_matrix.scenarios import (
    apply_long_gap_season,
    apply_matrix_scenario,
    apply_powered_short_gaps,
)
from Source.weppcliff_impute.experiment_matrix.scoring import (
    aggregate_metric_frame,
    score_dual_panel,
)
from Source.weppcliff_impute.experiment_matrix.runner import MatrixRunConfig, run_matrix
from Source.weppcliff_impute.validation import synthetic_weather_panel

ROOT = Path(__file__).resolve().parents[2]
DAILY = ROOT / "Autotest" / "Output" / "AutoTest.1-Daily-Time-Series.csv"


def _tiny_panel(n_years: int = 2, seed: int = 0) -> pd.DataFrame:
    """Synthetic daily panel with PRECIP-like columns for dual scoring."""
    base = synthetic_weather_panel(n_years=n_years, seed=seed, lat=42.0)
    # Map generic columns to WEPPCLIFF names
    df = pd.DataFrame(
        {
            "YYYYMMDD": base["YYYYMMDD"],
            "MIN_TEMP": base["MIN_TEMP"],
            "MAX_TEMP": base["MAX_TEMP"],
            "SO_RAD": base["SOLAR"] * 20.0,
            "W_VEL": base["WIND"],
            "W_DIR": (base["WIND"] * 40.0) % 360.0,
            "DP_TEMP": base["MIN_TEMP"] - 2.0,
            "PRECIP": np.clip(base["WIND"] - 3.5, 0, None) * 2.0,
            "BPS": 3.0,
            "ELEN": 1.0,
            "EMAX": 5.0,
            "EMIN": 0.5,
        }
    )
    # Dry many days
    rng = np.random.default_rng(seed)
    dry = rng.random(len(df)) > 0.35
    df.loc[dry, ["PRECIP", "BPS", "ELEN", "EMAX", "EMIN"]] = np.nan
    return df


def test_catalog_has_a_through_g():
    ids = {s.id for s in list_experiments()}
    assert {"A", "B", "C", "D", "E", "F", "G"}.issubset(ids)
    assert EXPERIMENT_CATALOG["A"].paper_n_seeds == 30


def test_powered_short_gaps_cover_fraction():
    df = _tiny_panel()
    cols = ["MIN_TEMP", "MAX_TEMP", "PRECIP"]
    applied = apply_powered_short_gaps(df, cols, target_frac=0.10, gap_len=4, seed=1, min_gaps=5)
    frac = float(applied.hole_day.mean())
    assert frac >= 0.05
    assert applied.missing_mask.to_numpy().any()


def test_long_gap_season_masks_correct_months():
    df = _tiny_panel(n_years=3)
    cols = ["MIN_TEMP", "MAX_TEMP", "SO_RAD"]
    applied = apply_long_gap_season(df, cols, season="JJA")
    months = df.loc[applied.hole_day, "YYYYMMDD"].astype(str).str.slice(4, 6).astype(int)
    assert set(months.unique()).issubset({6, 7, 8})
    assert applied.hole_day.sum() > 50


def test_dual_score_emits_whole_and_hole_periods():
    df = _tiny_panel()
    cols = ["MIN_TEMP", "MAX_TEMP", "SO_RAD", "DP_TEMP", "W_VEL", "W_DIR", "PRECIP"]
    truth = attach_clusters(df, cols, lat=42.0)
    applied = apply_matrix_scenario(truth, "mcar_ampute", cols, seed=0)
    amputed = applied.amputed.copy()
    amputed["CLUSTER"] = truth["CLUSTER"].to_numpy()
    eng = run_panel_engine("mean", truth, amputed, cols, seed=0, io=5, lat=42.0)
    metrics = score_dual_panel(
        truth,
        eng.imputed,
        applied.missing_mask,
        applied.hole_day,
        cols,
        experiment="A",
        engine="mean",
        scenario="mcar_ampute",
        seed=0,
    )
    assert not metrics.empty
    scopes = set(metrics["scope"].unique())
    assert "whole" in scopes
    assert "hole" in scopes
    assert "holdout_cells" in scopes
    periods = set(metrics["period"].unique())
    assert "daily" in periods
    assert "monthly" in periods or "seasonal" in periods or "annual" in periods
    # PRECIP mass_ratio exists for both scopes
    precip = metrics[(metrics["variable"] == "PRECIP") & (metrics["metric"] == "mass_ratio")]
    assert set(precip["scope"]).intersection({"whole", "hole"})


def test_post_cluster_engine_differs_from_pre_on_long_gap():
    df = _tiny_panel(n_years=2)
    cols = ["MIN_TEMP", "MAX_TEMP", "SO_RAD", "DP_TEMP", "W_VEL", "W_DIR", "PRECIP"]
    truth = attach_clusters(df, cols, lat=42.0)
    applied = apply_long_gap_season(truth, cols, season="DJF")
    amputed = applied.amputed.copy()
    amputed["CLUSTER"] = truth["CLUSTER"].to_numpy()
    pre = run_panel_engine("midastouch", truth, amputed, cols, seed=1, io=8, lat=42.0)
    post = run_panel_engine(
        "midastouch_post_cluster", truth, amputed, cols, seed=1, io=8, lat=42.0
    )
    # Both fill
    assert pre.imputed[cols].notna().all().all()
    assert post.imputed[cols].notna().all().all()
    assert pre.cluster_mode == "pre"
    assert post.cluster_mode == "post"


def test_aggregate_metric_frame_groups():
    rows = [
        {
            "experiment": "A",
            "engine": "midastouch",
            "scenario": "mcar_ampute",
            "seed": 0,
            "variable": "PRECIP",
            "metric": "wet_f1",
            "value": 0.9,
            "scope": "holdout_cells",
            "period": "daily",
            "metric_family": "precip_climate",
            "panel": "daily",
        },
        {
            "experiment": "A",
            "engine": "midastouch",
            "scenario": "mcar_ampute",
            "seed": 1,
            "variable": "PRECIP",
            "metric": "wet_f1",
            "value": 1.0,
            "scope": "holdout_cells",
            "period": "daily",
            "metric_family": "precip_climate",
            "panel": "daily",
        },
    ]
    agg = aggregate_metric_frame(pd.DataFrame(rows))
    assert len(agg) == 1
    assert agg.iloc[0]["mean"] == pytest.approx(0.95)


def test_smoke_matrix_on_synthetic_export(tmp_path: Path):
    """End-to-end matrix path on a tiny synthetic daily CSV (mean engine only)."""
    df = _tiny_panel(n_years=2, seed=3)
    daily_path = tmp_path / "daily.csv"
    df.to_csv(daily_path, index=False)
    cfg = MatrixRunConfig(
        experiments=("A", "B"),
        profile="smoke",
        n_seeds_override=1,
        serial=True,
        workers=1,
        io=5,
        out_dir=tmp_path / "matrix",
        include_workflow=False,
        daily_csv=daily_path,
        engines_override=("mean",),
        scenarios_override=None,
    )
    replicates, agg, gates, out = run_matrix(cfg)
    assert out.exists()
    assert (out / "replicates_long.csv").exists()
    assert not replicates.empty
    jobs = replicates[(replicates["variable"] == "_JOB") & (replicates["metric"] == "status")]
    assert float(jobs["value"].mean()) == 1.0
    sci = replicates[replicates["metric_family"] != "meta"]
    assert "hole" in set(sci["scope"])
    assert "whole" in set(sci["scope"])
    assert not agg.empty
