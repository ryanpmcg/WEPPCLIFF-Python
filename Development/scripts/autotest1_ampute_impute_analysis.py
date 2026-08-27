#!/usr/bin/env python3
"""
Ampute → impute validation for AutoTest.1 using standard WEPPCLIFF exports.

Evaluation philosophy
---------------------
* **Continuous smooth fields** (temps, dewpoint, solar): day-matched hold-out
  RMSE / MAE / R² — a held-out Tuesday should look like that Tuesday.
* **Precipitation**: never score day-matched mm amounts. Score wet/dry
  occurrence, monthly / seasonal mass, and wet-day intensity *distributions*.
* **Wind speed / direction**: score climatology and distributions (monthly
  means, sector frequencies, KS / quantile match). Day-matched degree or
  m/s error is diagnostic only, not the primary test.

Inputs (WEPPCLIFF ``-ed 6`` / ``-ed 3``):
  Autotest/Output/AutoTest.1-Daily-Time-Series.csv
  Autotest/Output/AutoTest.1-Precipitation.csv   (optional)

Writes tables and plots under:
  Autotest/Output/AmputeImpute/
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.weppcliff_impute.adapter import ImputeConfig, impute_by_cluster  # noqa: E402
from Source.weppcliff_impute.cluster import assign_cluster_labels  # noqa: E402
from pymice.ampute import ampute  # noqa: E402
from Source.weppcliff_impute.missingness import MissingnessKind  # noqa: E402
from Source.weppcliff_impute.validation import (  # noqa: E402
    ScenarioResult,
    WeatherScenario,
    apply_long_gap_all,
    apply_short_gaps,
    apply_variable_absent,
    enforce_temperature_order,
)


def _mask_frame(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(False, index=df.index, columns=value_cols)


OUT_DIR = ROOT / "Autotest" / "Output" / "AmputeImpute"
DAILY_CSV = ROOT / "Autotest" / "Output" / "AutoTest.1-Daily-Time-Series.csv"
PRECIP_CSV = ROOT / "Autotest" / "Output" / "AutoTest.1-Precipitation.csv"
PKL_PATH = ROOT / "Autotest" / "Output" / "AutoTest.1.pkl"

DAILY_COMPLETE_COLS: tuple[str, ...] = (
    "MIN_TEMP",
    "MAX_TEMP",
    "SO_RAD",
    "W_VEL",
    "W_DIR",
    "DP_TEMP",
)
DAILY_PARTIAL_COLS: tuple[str, ...] = (
    "PRECIP",
    "BPS",
    "ELEN",
    "EMAX",
    "EMIN",
)
DAILY_VALUE_COLS: tuple[str, ...] = DAILY_COMPLETE_COLS + DAILY_PARTIAL_COLS

# Day-matched error is a fair primary metric only for these.
POINTWISE_VARS: frozenset[str] = frozenset(
    {"MIN_TEMP", "MAX_TEMP", "DP_TEMP", "SO_RAD"}
)
PRECIP_VARS: frozenset[str] = frozenset({"PRECIP", "BPS", "ELEN", "EMAX", "EMIN"})
WIND_VARS: frozenset[str] = frozenset({"W_VEL", "W_DIR"})

PRECIP_VALUE_COLS: tuple[str, ...] = ("PRECIP", "DUR", "INT", "ANT", "BTW")

SCENARIOS: tuple[WeatherScenario, ...] = (
    WeatherScenario.MCAR_AMPUTE,
    WeatherScenario.SHORT_GAPS,
    WeatherScenario.LONG_GAP_ALL,
    WeatherScenario.VARIABLE_ABSENT,
)

SEED = 42
LAT = 42.2
IO_ITERS = 30
WET_MM = 0.25  # daily wet-day threshold (mm)
N_WIND_SECTORS = 8


# ---------------------------------------------------------------------------
# Load / ampute helpers
# ---------------------------------------------------------------------------


def load_daily_panel() -> pd.DataFrame:
    """Load complete daily panel from standard WEPPCLIFF CSV (or pkl fallback)."""
    if DAILY_CSV.exists():
        df = pd.read_csv(DAILY_CSV)
    elif PKL_PATH.exists() or PKL_PATH.with_suffix(".npz").exists():
        from Source.export_store import load_exported_store

        path = PKL_PATH if PKL_PATH.exists() else PKL_PATH.with_suffix(".npz")
        store = load_exported_store(path)
        df = store[10].copy()
    else:
        raise FileNotFoundError(
            f"Missing AutoTest.1 daily export. Run:\n"
            f"  python weppcliff.py -f ASOS_BPT_KMQE.csv -fn AutoTest.1 "
            f"-o Autotest/Output -e Autotest/Output -la y -ed 6"
        )

    if "Omitted-Storms" in df.columns:
        df = df.drop(columns=["Omitted-Storms"])
    df["YYYYMMDD"] = df["YYYYMMDD"].astype(str).str.replace(r"\.0$", "", regex=True)
    return df.reset_index(drop=True)


def load_precip_panel(max_rows: int = 8000) -> pd.DataFrame | None:
    if not PRECIP_CSV.exists():
        return None
    df = pd.read_csv(PRECIP_CSV)
    if "DT_1" not in df.columns:
        return None
    df = df.head(max_rows).copy()
    df["YYYYMMDD"] = pd.to_datetime(df["DT_1"], errors="coerce").dt.strftime("%Y%m%d")
    return df.reset_index(drop=True)


def apply_mcar_all_variables(
    df: pd.DataFrame,
    value_cols: list[str],
    *,
    prop: float = 0.12,
    seed: int = 0,
) -> ScenarioResult:
    """MCAR amputation for mixed complete/partial panels."""
    out = df.copy()
    mask = _mask_frame(df, value_cols)
    complete = [c for c in value_cols if c in DAILY_COMPLETE_COLS and c in df.columns]
    partial = [c for c in value_cols if c not in complete]

    if complete:
        matrix = np.array(df[complete].to_numpy(dtype=np.float64), copy=True)
        for j in range(matrix.shape[1]):
            miss = np.isnan(matrix[:, j])
            if miss.any():
                med = np.nanmedian(matrix[:, j])
                matrix[miss, j] = med if np.isfinite(med) else 0.0
        result = ampute(matrix, prop=prop, seed=seed)
        if result.amp is not None:
            out[complete] = result.amp
            mask[complete] = np.isnan(result.amp)

    rng = np.random.default_rng(seed + 17)
    for col in partial:
        observed = df[col].notna().to_numpy()
        idx = np.flatnonzero(observed)
        if len(idx) == 0:
            continue
        n_drop = max(1, int(round(prop * len(idx))))
        drop = rng.choice(idx, size=min(n_drop, len(idx)), replace=False)
        out.loc[drop, col] = np.nan
        mask.loc[drop, col] = True

    return ScenarioResult(
        amputed=out,
        missing_mask=mask,
        expected_kind=MissingnessKind.SHORT_GAPS,
        scenario=WeatherScenario.MCAR_AMPUTE,
    )


def attach_clusters(df: pd.DataFrame, value_cols: list[str], lat: float = LAT) -> pd.DataFrame:
    out = df.copy()
    ymd = out["YYYYMMDD"].astype(str).tolist()
    matrix = np.array(out[value_cols].to_numpy(dtype=np.float64), copy=True)
    col_med = np.nanmedian(matrix, axis=0)
    for j in range(matrix.shape[1]):
        miss = np.isnan(matrix[:, j])
        if miss.any():
            matrix[miss, j] = col_med[j] if np.isfinite(col_med[j]) else 0.0
    out["CLUSTER"] = assign_cluster_labels(ymd, matrix, lat=lat, n_vars=len(value_cols))
    return out


def _eval_mask(
    truth: pd.DataFrame, imputed: pd.DataFrame, missing_mask: pd.DataFrame, col: str
) -> np.ndarray:
    """Held-out cells that had finite truth and a finite impute."""
    return (
        missing_mask[col].to_numpy(dtype=bool)
        & truth[col].notna().to_numpy()
        & imputed[col].notna().to_numpy()
    )


def _reconstruct(
    truth: pd.DataFrame, imputed: pd.DataFrame, missing_mask: pd.DataFrame, col: str
) -> np.ndarray:
    """Full series: truth outside ampute, imputed inside ampute."""
    y = truth[col].to_numpy(dtype=float).copy()
    m = missing_mask[col].to_numpy(dtype=bool)
    yhat = imputed[col].to_numpy(dtype=float)
    filled = m & np.isfinite(yhat)
    y[filled] = yhat[filled]
    return y


def _circular_mean_deg(angles: np.ndarray) -> float:
    a = np.asarray(angles, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    rad = np.deg2rad(a)
    mean = np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))
    deg = float(np.rad2deg(mean))
    return deg % 360.0


def _circular_diff_deg(a: float, b: float) -> float:
    """Smallest absolute angular difference in degrees."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 5 or b.size < 5:
        return float("nan")
    return float(stats.ks_2samp(a, b).statistic)


def _quantile_mae(a: np.ndarray, b: np.ndarray, qs: np.ndarray | None = None) -> float:
    """MAE of empirical quantiles (distributional match, not day-paired)."""
    if qs is None:
        qs = np.linspace(0.1, 0.9, 9)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 5 or b.size < 5:
        return float("nan")
    qa = np.quantile(a, qs)
    qb = np.quantile(b, qs)
    return float(np.mean(np.abs(qa - qb)))


# ---------------------------------------------------------------------------
# Primary scoring
# ---------------------------------------------------------------------------


def score_pointwise(
    truth: pd.DataFrame,
    imputed: pd.DataFrame,
    missing_mask: pd.DataFrame,
    value_cols: list[str],
    *,
    scenario: str,
    panel: str,
) -> pd.DataFrame:
    """Day-matched hold-out metrics for smooth continuous variables only."""
    rows: list[dict] = []
    for col in value_cols:
        if col not in POINTWISE_VARS:
            continue
        filled = _eval_mask(truth, imputed, missing_mask, col)
        n_miss = int(
            (missing_mask[col].to_numpy() & truth[col].notna().to_numpy()).sum()
        )
        n_fill = int(filled.sum())
        base = {
            "panel": panel,
            "scenario": scenario,
            "variable": col,
            "metric_family": "pointwise",
            "n_amputed": n_miss,
            "n_filled": n_fill,
            "fill_rate": n_fill / n_miss if n_miss else np.nan,
        }
        if n_fill == 0:
            rows.append({**base, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "r2": np.nan})
            continue
        y = truth.loc[filled, col].to_numpy(dtype=float)
        yhat = imputed.loc[filled, col].to_numpy(dtype=float)
        resid = yhat - y
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        rows.append(
            {
                **base,
                "rmse": float(np.sqrt(np.mean(resid**2))),
                "mae": float(np.mean(np.abs(resid))),
                "bias": float(np.mean(resid)),
                "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def score_precip_climate(
    truth: pd.DataFrame,
    imputed: pd.DataFrame,
    missing_mask: pd.DataFrame,
    *,
    scenario: str,
    panel: str = "daily",
    col: str = "PRECIP",
    wet_mm: float = WET_MM,
) -> pd.DataFrame:
    """
    Climatological / occurrence metrics for daily precipitation.

    Does **not** use day-matched amount RMSE. Primary quantities:
      - wet/dry agreement on held-out days
      - monthly mass bias on the reconstructed series
      - wet-day intensity distribution match (KS, quantile MAE, mean intensity bias)
    """
    if col not in truth.columns:
        return pd.DataFrame()

    ymd = truth["YYYYMMDD"].astype(str).to_numpy()
    had = truth[col].notna().to_numpy()
    miss = missing_mask[col].to_numpy(dtype=bool) & had
    filled = miss & imputed[col].notna().to_numpy()
    n_miss = int(miss.sum())
    n_fill = int(filled.sum())

    rows: list[dict] = []
    base = {
        "panel": panel,
        "scenario": scenario,
        "variable": col,
        "metric_family": "precip_climate",
        "n_amputed": n_miss,
        "n_filled": n_fill,
        "fill_rate": n_fill / n_miss if n_miss else np.nan,
    }

    # --- wet/dry occurrence on held-out days ---
    if n_fill > 0:
        yt = truth.loc[filled, col].to_numpy(dtype=float)
        yh = imputed.loc[filled, col].to_numpy(dtype=float)
        wet_t = yt >= wet_mm
        wet_h = yh >= wet_mm
        acc = float(np.mean(wet_t == wet_h))
        # F1 for wet class
        tp = float(np.sum(wet_t & wet_h))
        fp = float(np.sum(~wet_t & wet_h))
        fn = float(np.sum(wet_t & ~wet_h))
        prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1 = (
            2 * prec * rec / (prec + rec)
            if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0
            else np.nan
        )
        wet_freq_truth = float(np.mean(wet_t))
        wet_freq_imp = float(np.mean(wet_h))
        rows.append(
            {
                **base,
                "metric": "wet_dry_accuracy",
                "value": acc,
                "detail": f"wet_mm={wet_mm}",
            }
        )
        rows.append({**base, "metric": "wet_f1", "value": f1, "detail": ""})
        rows.append(
            {
                **base,
                "metric": "wet_freq_truth",
                "value": wet_freq_truth,
                "detail": "held-out days",
            }
        )
        rows.append(
            {
                **base,
                "metric": "wet_freq_imputed",
                "value": wet_freq_imp,
                "detail": "held-out days",
            }
        )
        rows.append(
            {
                **base,
                "metric": "wet_freq_bias",
                "value": wet_freq_imp - wet_freq_truth,
                "detail": "imputed − truth",
            }
        )

        # Wet-day intensity: compare distributions of amounts on days that were
        # truly wet (imputed wet-day amount vs truth wet-day amount as sets).
        # Not paired day-to-day — two samples of wet-day depths.
        wet_truth_amt = yt[wet_t]
        wet_imp_amt = yh[wet_h]
        rows.append(
            {
                **base,
                "metric": "wet_intensity_ks",
                "value": _ks_stat(wet_truth_amt, wet_imp_amt),
                "detail": "KS(truth wet, imputed wet); lower better",
            }
        )
        rows.append(
            {
                **base,
                "metric": "wet_intensity_qmae",
                "value": _quantile_mae(wet_truth_amt, wet_imp_amt),
                "detail": "quantile MAE of wet-day mm",
            }
        )
        mean_t = float(np.mean(wet_truth_amt)) if wet_truth_amt.size else np.nan
        mean_h = float(np.mean(wet_imp_amt)) if wet_imp_amt.size else np.nan
        rows.append(
            {
                **base,
                "metric": "wet_mean_intensity_bias",
                "value": (mean_h - mean_t) if np.isfinite(mean_t) and np.isfinite(mean_h) else np.nan,
                "detail": "mm; imputed mean wet − truth mean wet",
            }
        )
        # Total mass on held-out days (aggregate, not per-day)
        sum_t = float(np.sum(yt))
        sum_h = float(np.sum(yh))
        rows.append(
            {
                **base,
                "metric": "holdout_mass_ratio",
                "value": sum_h / sum_t if sum_t > 1e-9 else np.nan,
                "detail": "sum(imputed)/sum(truth) on held-out; 1.0 ideal",
            }
        )
    else:
        for m in (
            "wet_dry_accuracy",
            "wet_f1",
            "wet_freq_bias",
            "wet_intensity_ks",
            "wet_intensity_qmae",
            "holdout_mass_ratio",
        ):
            rows.append({**base, "metric": m, "value": np.nan, "detail": "no fills"})

    # --- monthly mass bias on reconstructed full series ---
    recon = _reconstruct(truth, imputed, missing_mask, col)
    truth_arr = truth[col].to_numpy(dtype=float)
    # Fill NaN dry days as 0 for mass accounting when PRECIP is partial
    t_mass = np.where(np.isfinite(truth_arr), truth_arr, 0.0)
    r_mass = np.where(np.isfinite(recon), recon, 0.0)
    # Where truth was naturally NaN (dry), reconstructed dry is fine as 0
    months = np.array([d[:6] if len(d) >= 6 else "000000" for d in ymd])
    frame = pd.DataFrame({"YYYYMM": months, "truth": t_mass, "recon": r_mass})
    monthly = frame.groupby("YYYYMM", as_index=False)[["truth", "recon"]].sum()
    denom = np.maximum(np.abs(monthly["truth"].to_numpy()), 1e-6)
    rel = np.abs(monthly["recon"].to_numpy() - monthly["truth"].to_numpy()) / denom
    rows.append(
        {
            **base,
            "metric": "monthly_mass_bias",
            "value": float(np.nanmean(rel)),
            "detail": "mean |monthly recon−truth|/|truth|; lower better",
        }
    )
    # Seasonal mass bias
    seasons = {
        "DJF": {12, 1, 2},
        "MAM": {3, 4, 5},
        "JJA": {6, 7, 8},
        "SON": {9, 10, 11},
    }
    mon = np.array([int(d[4:6]) if len(d) >= 6 else 0 for d in ymd])
    for name, mset in seasons.items():
        sel = np.isin(mon, list(mset))
        st = float(np.sum(t_mass[sel]))
        sr = float(np.sum(r_mass[sel]))
        rows.append(
            {
                **base,
                "metric": f"seasonal_mass_ratio_{name}",
                "value": sr / st if st > 1e-9 else np.nan,
                "detail": "recon/truth seasonal total; 1.0 ideal",
            }
        )

    return pd.DataFrame(rows)


def score_storm_distribution(
    truth: pd.DataFrame,
    imputed: pd.DataFrame,
    missing_mask: pd.DataFrame,
    cols: list[str],
    *,
    scenario: str,
    panel: str = "daily",
) -> pd.DataFrame:
    """Distributional metrics for storm attributes (not day-matched)."""
    rows: list[dict] = []
    for col in cols:
        if col not in truth.columns or col == "PRECIP":
            continue
        filled = _eval_mask(truth, imputed, missing_mask, col)
        n_miss = int(
            (missing_mask[col].to_numpy() & truth[col].notna().to_numpy()).sum()
        )
        n_fill = int(filled.sum())
        base = {
            "panel": panel,
            "scenario": scenario,
            "variable": col,
            "metric_family": "storm_distribution",
            "n_amputed": n_miss,
            "n_filled": n_fill,
            "fill_rate": n_fill / n_miss if n_miss else np.nan,
        }
        if n_fill < 5:
            rows.append({**base, "metric": "ks", "value": np.nan, "detail": "too few"})
            continue
        yt = truth.loc[filled, col].to_numpy(dtype=float)
        yh = imputed.loc[filled, col].to_numpy(dtype=float)
        rows.append(
            {
                **base,
                "metric": "ks",
                "value": _ks_stat(yt, yh),
                "detail": "KS(truth held-out, imputed); lower better",
            }
        )
        rows.append(
            {
                **base,
                "metric": "quantile_mae",
                "value": _quantile_mae(yt, yh),
                "detail": "empirical quantile MAE",
            }
        )
        mt, mh = float(np.mean(yt)), float(np.mean(yh))
        rows.append(
            {
                **base,
                "metric": "mean_ratio",
                "value": mh / mt if abs(mt) > 1e-12 else np.nan,
                "detail": "mean(imputed)/mean(truth); 1.0 ideal",
            }
        )
    return pd.DataFrame(rows)


def score_wind_climate(
    truth: pd.DataFrame,
    imputed: pd.DataFrame,
    missing_mask: pd.DataFrame,
    *,
    scenario: str,
    panel: str = "daily",
) -> pd.DataFrame:
    """
    Climatological / distributional wind metrics.

    Primary tests are *not* day-matched m/s or degree error:
      W_VEL — KS, quantile match, monthly mean speed bias
      W_DIR — sector histogram distance, monthly circular-mean error
    """
    rows: list[dict] = []
    ymd = truth["YYYYMMDD"].astype(str).to_numpy()
    months = np.array([d[:6] if len(d) >= 6 else "000000" for d in ymd])

    # ---- W_VEL ----
    if "W_VEL" in truth.columns:
        col = "W_VEL"
        filled = _eval_mask(truth, imputed, missing_mask, col)
        n_miss = int(
            (missing_mask[col].to_numpy() & truth[col].notna().to_numpy()).sum()
        )
        n_fill = int(filled.sum())
        base = {
            "panel": panel,
            "scenario": scenario,
            "variable": col,
            "metric_family": "wind_climate",
            "n_amputed": n_miss,
            "n_filled": n_fill,
            "fill_rate": n_fill / n_miss if n_miss else np.nan,
        }
        if n_fill >= 5:
            yt = truth.loc[filled, col].to_numpy(dtype=float)
            yh = imputed.loc[filled, col].to_numpy(dtype=float)
            rows.append(
                {
                    **base,
                    "metric": "speed_ks",
                    "value": _ks_stat(yt, yh),
                    "detail": "KS distribution match; lower better",
                }
            )
            rows.append(
                {
                    **base,
                    "metric": "speed_quantile_mae",
                    "value": _quantile_mae(yt, yh),
                    "detail": "quantile MAE (m/s or site units)",
                }
            )
            mt, mh = float(np.mean(yt)), float(np.mean(yh))
            st, sh = float(np.std(yt)), float(np.std(yh))
            rows.append(
                {
                    **base,
                    "metric": "speed_mean_ratio",
                    "value": mh / mt if mt > 1e-9 else np.nan,
                    "detail": "1.0 ideal",
                }
            )
            rows.append(
                {
                    **base,
                    "metric": "speed_std_ratio",
                    "value": sh / st if st > 1e-9 else np.nan,
                    "detail": "1.0 ideal",
                }
            )
        else:
            for m in ("speed_ks", "speed_quantile_mae", "speed_mean_ratio"):
                rows.append({**base, "metric": m, "value": np.nan, "detail": ""})

        recon = _reconstruct(truth, imputed, missing_mask, col)
        t_arr = truth[col].to_numpy(dtype=float)
        frame = pd.DataFrame({"YYYYMM": months, "truth": t_arr, "recon": recon})
        monthly = frame.groupby("YYYYMM", as_index=False)[["truth", "recon"]].mean()
        mt = monthly["truth"].to_numpy()
        mr = monthly["recon"].to_numpy()
        ok = np.isfinite(mt) & np.isfinite(mr)
        if ok.any():
            denom = np.maximum(np.abs(mt[ok]), 1e-6)
            rel = np.abs(mr[ok] - mt[ok]) / denom
            rows.append(
                {
                    **base,
                    "metric": "monthly_mean_speed_bias",
                    "value": float(np.mean(rel)),
                    "detail": "mean |monthly recon−truth|/|truth|; lower better",
                }
            )
        else:
            rows.append(
                {
                    **base,
                    "metric": "monthly_mean_speed_bias",
                    "value": np.nan,
                    "detail": "",
                }
            )

    # ---- W_DIR ----
    if "W_DIR" in truth.columns:
        col = "W_DIR"
        filled = _eval_mask(truth, imputed, missing_mask, col)
        n_miss = int(
            (missing_mask[col].to_numpy() & truth[col].notna().to_numpy()).sum()
        )
        n_fill = int(filled.sum())
        base = {
            "panel": panel,
            "scenario": scenario,
            "variable": col,
            "metric_family": "wind_climate",
            "n_amputed": n_miss,
            "n_filled": n_fill,
            "fill_rate": n_fill / n_miss if n_miss else np.nan,
        }
        if n_fill >= 5:
            yt = truth.loc[filled, col].to_numpy(dtype=float) % 360.0
            yh = imputed.loc[filled, col].to_numpy(dtype=float) % 360.0
            # Sector histogram total variation distance
            edges = np.linspace(0, 360, N_WIND_SECTORS + 1)
            ht, _ = np.histogram(yt, bins=edges, density=True)
            hh, _ = np.histogram(yh, bins=edges, density=True)
            # density * bin_width → probability mass
            width = 360.0 / N_WIND_SECTORS
            pt = ht * width
            ph = hh * width
            pt = pt / pt.sum() if pt.sum() > 0 else pt
            ph = ph / ph.sum() if ph.sum() > 0 else ph
            tvd = 0.5 * float(np.sum(np.abs(pt - ph)))
            rows.append(
                {
                    **base,
                    "metric": "sector_tvd",
                    "value": tvd,
                    "detail": f"{N_WIND_SECTORS}-sector total variation; 0 ideal, ≤0.5",
                }
            )
            # Circular means of the two *samples* (not paired days)
            cm_t = _circular_mean_deg(yt)
            cm_h = _circular_mean_deg(yh)
            rows.append(
                {
                    **base,
                    "metric": "holdout_circular_mean_error_deg",
                    "value": _circular_diff_deg(cm_t, cm_h),
                    "detail": "|circmean(truth sample)−circmean(imputed sample)|",
                }
            )
        else:
            for m in ("sector_tvd", "holdout_circular_mean_error_deg"):
                rows.append({**base, "metric": m, "value": np.nan, "detail": ""})

        # Monthly circular-mean climatology on reconstructed series
        recon = _reconstruct(truth, imputed, missing_mask, col)
        t_arr = truth[col].to_numpy(dtype=float)
        errs = []
        for ym in np.unique(months):
            sel = months == ym
            ct = _circular_mean_deg(t_arr[sel])
            cr = _circular_mean_deg(recon[sel])
            e = _circular_diff_deg(ct, cr)
            if np.isfinite(e):
                errs.append(e)
        rows.append(
            {
                **base,
                "metric": "monthly_circular_mean_error_deg",
                "value": float(np.mean(errs)) if errs else np.nan,
                "detail": "mean |monthly circmean recon−truth|; lower better",
            }
        )

    return pd.DataFrame(rows)


def score_all(
    truth: pd.DataFrame,
    imputed: pd.DataFrame,
    missing_mask: pd.DataFrame,
    value_cols: list[str],
    *,
    scenario: str,
    panel: str = "daily",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pointwise_wide, climate_long) metric frames."""
    pw = score_pointwise(
        truth, imputed, missing_mask, value_cols, scenario=scenario, panel=panel
    )
    climate_parts = [
        score_precip_climate(
            truth, imputed, missing_mask, scenario=scenario, panel=panel
        ),
        score_storm_distribution(
            truth,
            imputed,
            missing_mask,
            [c for c in value_cols if c in PRECIP_VARS],
            scenario=scenario,
            panel=panel,
        ),
        score_wind_climate(
            truth, imputed, missing_mask, scenario=scenario, panel=panel
        ),
    ]
    climate = pd.concat([p for p in climate_parts if not p.empty], ignore_index=True)
    return pw, climate


def score_precip_panel_distribution(
    truth: pd.DataFrame,
    imputed: pd.DataFrame,
    missing_mask: pd.DataFrame,
    value_cols: list[str],
    *,
    scenario: str,
) -> pd.DataFrame:
    """Breakpoint panel: distributional match only (no day-paired mm RMSE)."""
    rows: list[dict] = []
    for col in value_cols:
        filled = _eval_mask(truth, imputed, missing_mask, col)
        n_miss = int(
            (missing_mask[col].to_numpy() & truth[col].notna().to_numpy()).sum()
        )
        n_fill = int(filled.sum())
        base = {
            "panel": "precip_breakpoints",
            "scenario": scenario,
            "variable": col,
            "metric_family": "breakpoint_distribution",
            "n_amputed": n_miss,
            "n_filled": n_fill,
            "fill_rate": n_fill / n_miss if n_miss else np.nan,
        }
        if n_fill < 5:
            rows.append({**base, "metric": "ks", "value": np.nan, "detail": ""})
            continue
        yt = truth.loc[filled, col].to_numpy(dtype=float)
        yh = imputed.loc[filled, col].to_numpy(dtype=float)
        rows.append({**base, "metric": "ks", "value": _ks_stat(yt, yh), "detail": ""})
        rows.append(
            {
                **base,
                "metric": "quantile_mae",
                "value": _quantile_mae(yt, yh),
                "detail": "",
            }
        )
        mt, mh = float(np.mean(yt)), float(np.mean(yh))
        rows.append(
            {
                **base,
                "metric": "mean_ratio",
                "value": mh / mt if abs(mt) > 1e-12 else np.nan,
                "detail": "1.0 ideal",
            }
        )
        if col == "PRECIP":
            # mass on held-out breakpoints
            st, sh = float(np.sum(yt)), float(np.sum(yh))
            rows.append(
                {
                    **base,
                    "metric": "holdout_mass_ratio",
                    "value": sh / st if st > 1e-9 else np.nan,
                    "detail": "sum ratio; 1.0 ideal",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------


def run_daily_scenarios(
    truth: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    value_cols = [c for c in DAILY_VALUE_COLS if c in truth.columns]
    panel = attach_clusters(truth, value_cols)
    cfg = ImputeConfig(pm="t", io=IO_ITERS, qi="f", seed=SEED, lat=LAT, kg="Dfb")
    pointwise_all: list[pd.DataFrame] = []
    climate_all: list[pd.DataFrame] = []
    payloads: dict = {}

    for scenario in SCENARIOS:
        t0 = time.perf_counter()
        if scenario == WeatherScenario.VARIABLE_ABSENT:
            for col in value_cols:
                masked = apply_variable_absent(panel, value_cols, column=col)
                work = masked.amputed[["CLUSTER", *value_cols]].copy()
                imputed = enforce_temperature_order(
                    impute_by_cluster(work, "midastouch", cfg)
                )
                pw, cl = score_all(
                    panel,
                    imputed,
                    masked.missing_mask,
                    value_cols,
                    scenario=f"variable_absent:{col}",
                    panel="daily",
                )
                pointwise_all.append(pw)
                climate_all.append(cl)
                if col in ("MIN_TEMP", "PRECIP", "W_VEL", "W_DIR"):
                    payloads[f"variable_absent:{col}"] = {
                        "truth": panel,
                        "imputed": imputed,
                        "mask": masked.missing_mask,
                        "value_cols": value_cols,
                    }
            print(
                f"  {scenario.value}: {time.perf_counter() - t0:.1f}s (per-variable absent)"
            )
            continue

        if scenario == WeatherScenario.MCAR_AMPUTE:
            masked = apply_mcar_all_variables(panel, value_cols, prop=0.12, seed=SEED)
        elif scenario == WeatherScenario.SHORT_GAPS:
            masked = apply_short_gaps(panel, value_cols, seed=SEED)
        elif scenario == WeatherScenario.LONG_GAP_ALL:
            masked = apply_long_gap_all(panel, value_cols)
        else:
            raise ValueError(f"unhandled scenario {scenario}")

        work = masked.amputed[["CLUSTER", *value_cols]].copy()
        imputed = enforce_temperature_order(impute_by_cluster(work, "midastouch", cfg))
        pw, cl = score_all(
            panel,
            imputed,
            masked.missing_mask,
            value_cols,
            scenario=scenario.value,
            panel="daily",
        )
        pointwise_all.append(pw)
        climate_all.append(cl)
        payloads[scenario.value] = {
            "truth": panel,
            "imputed": imputed,
            "mask": masked.missing_mask,
            "value_cols": value_cols,
        }
        # Brief console: pointwise mean RMSE + key climate metrics
        pw_rmse = float(pw["rmse"].mean(skipna=True)) if not pw.empty else np.nan
        cl_sub = cl[cl["scenario"] == scenario.value] if not cl.empty else cl
        precip_acc = _metric_value(cl_sub, "PRECIP", "wet_dry_accuracy")
        precip_mass = _metric_value(cl_sub, "PRECIP", "monthly_mass_bias")
        wind_ks = _metric_value(cl_sub, "W_VEL", "speed_ks")
        dir_tvd = _metric_value(cl_sub, "W_DIR", "sector_tvd")
        print(
            f"  {scenario.value}: pointwise RMSE={pw_rmse:.3f} | "
            f"PRECIP wet_acc={precip_acc:.3f} mass_bias={precip_mass:.3f} | "
            f"W_VEL KS={wind_ks:.3f} W_DIR sector_tvd={dir_tvd:.3f} "
            f"({time.perf_counter() - t0:.1f}s)"
        )

    return (
        pd.concat(pointwise_all, ignore_index=True) if pointwise_all else pd.DataFrame(),
        pd.concat(climate_all, ignore_index=True) if climate_all else pd.DataFrame(),
        payloads,
    )


def _metric_value(climate: pd.DataFrame, variable: str, metric: str) -> float:
    if climate.empty:
        return float("nan")
    sub = climate[(climate["variable"] == variable) & (climate["metric"] == metric)]
    if sub.empty:
        return float("nan")
    return float(sub["value"].iloc[0])


def run_precip_mcar(precip: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    value_cols = [c for c in PRECIP_VALUE_COLS if c in precip.columns]
    panel = attach_clusters(precip, value_cols)
    cfg = ImputeConfig(pm="t", io=IO_ITERS, qi="t", seed=SEED, lat=LAT)
    matrix = np.array(panel[value_cols].to_numpy(dtype=np.float64), copy=True)
    for j in range(matrix.shape[1]):
        miss = np.isnan(matrix[:, j])
        if miss.any():
            med = np.nanmedian(matrix[:, j])
            matrix[miss, j] = med if np.isfinite(med) else 0.0
    result = ampute(matrix, prop=0.12, seed=SEED)
    out = panel.copy()
    mask = _mask_frame(panel, value_cols)
    if result.amp is not None:
        out[value_cols] = result.amp
        mask[value_cols] = np.isnan(result.amp)
    work = out[["CLUSTER", *value_cols]].copy()
    imputed = impute_by_cluster(work, "cart", cfg)
    metrics = score_precip_panel_distribution(
        panel, imputed, mask, value_cols, scenario="mcar_ampute"
    )
    payloads = {
        "mcar_ampute": {
            "truth": panel,
            "imputed": imputed,
            "mask": mask,
            "value_cols": value_cols,
        }
    }
    return metrics, payloads


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_pointwise_bars(pointwise: pd.DataFrame, out_path: Path) -> None:
    if pointwise.empty:
        return
    daily = pointwise[pointwise["panel"] == "daily"].copy()
    daily["scenario_plot"] = daily["scenario"].apply(
        lambda s: "variable_absent" if str(s).startswith("variable_absent") else s
    )
    va = daily["scenario"].astype(str).str.startswith("variable_absent:")
    if va.any():
        keep = [
            row["variable"] == str(row["scenario"]).split(":", 1)[1]
            for _, row in daily[va].iterrows()
        ]
        daily = pd.concat([daily[~va], daily[va].loc[keep]], ignore_index=True)

    scenarios = daily["scenario_plot"].unique().tolist()
    variables = [c for c in sorted(POINTWISE_VARS) if c in daily["variable"].values]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    x = np.arange(len(variables))
    width = 0.8 / max(len(scenarios), 1)
    for ax, metric, title in zip(axes, ("rmse", "mae"), ("RMSE", "MAE")):
        for i, scen in enumerate(scenarios):
            sub = daily[daily["scenario_plot"] == scen].set_index("variable")
            vals = [sub.loc[v, metric] if v in sub.index else np.nan for v in variables]
            ax.bar(x + i * width, vals, width, label=scen)
        ax.set_ylabel(title)
        ax.set_title(f"Pointwise hold-out {title} (temps / dew / solar only)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
    axes[-1].set_xticks(x + width * (len(scenarios) - 1) / 2)
    axes[-1].set_xticklabels(variables, rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_precip_climate(climate: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of key PRECIP climate metrics by scenario."""
    if climate.empty:
        return
    keys = [
        "wet_dry_accuracy",
        "wet_f1",
        "monthly_mass_bias",
        "holdout_mass_ratio",
        "wet_intensity_ks",
    ]
    sub = climate[
        (climate["variable"] == "PRECIP")
        & (climate["metric"].isin(keys))
        & (~climate["scenario"].astype(str).str.startswith("variable_absent:"))
    ].copy()
    if sub.empty:
        return
    scenarios = sub["scenario"].unique().tolist()
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 4), sharey=False)
    if len(keys) == 1:
        axes = [axes]
    for ax, metric in zip(axes, keys):
        vals = []
        for scen in scenarios:
            row = sub[(sub["scenario"] == scen) & (sub["metric"] == metric)]
            vals.append(float(row["value"].iloc[0]) if len(row) else np.nan)
        ax.bar(range(len(scenarios)), vals, color="C0", alpha=0.85)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, rotation=35, ha="right", fontsize=8)
        ax.set_title(metric, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        if metric in ("wet_dry_accuracy", "wet_f1", "holdout_mass_ratio"):
            ax.axhline(1.0 if metric == "holdout_mass_ratio" else 0.0, color="k", lw=0.5)
            if metric != "holdout_mass_ratio":
                ax.set_ylim(0, 1.05)
            else:
                ax.axhline(1.0, color="C3", ls="--", lw=1)
    fig.suptitle("PRECIP evaluation (occurrence / mass / wet-day distribution — not day-matched mm)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_wind_climate(climate: pd.DataFrame, out_path: Path) -> None:
    if climate.empty:
        return
    keys_vel = ["speed_ks", "speed_mean_ratio", "monthly_mean_speed_bias"]
    keys_dir = ["sector_tvd", "monthly_circular_mean_error_deg", "holdout_circular_mean_error_deg"]
    main = climate[~climate["scenario"].astype(str).str.startswith("variable_absent:")].copy()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, metric, var in zip(
        axes[0],
        keys_vel,
        ["W_VEL"] * 3,
    ):
        sub = main[(main["variable"] == var) & (main["metric"] == metric)]
        scenarios = sub["scenario"].tolist()
        vals = sub["value"].tolist()
        ax.bar(range(len(scenarios)), vals, color="C2", alpha=0.85)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{var}: {metric}", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        if "ratio" in metric:
            ax.axhline(1.0, color="C3", ls="--", lw=1)
    for ax, metric in zip(axes[1], keys_dir):
        sub = main[(main["variable"] == "W_DIR") & (main["metric"] == metric)]
        scenarios = sub["scenario"].tolist()
        vals = sub["value"].tolist()
        ax.bar(range(len(scenarios)), vals, color="C1", alpha=0.85)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"W_DIR: {metric}", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Wind evaluation (distributions / climatology — not day-matched m/s or °)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_precip_monthly_scatter(payload: dict, scenario: str, out_path: Path) -> None:
    """Monthly reconstructed vs truth precip totals."""
    truth = payload["truth"]
    imputed = payload["imputed"]
    mask = payload["mask"]
    if "PRECIP" not in truth.columns:
        return
    recon = _reconstruct(truth, imputed, mask, "PRECIP")
    t = truth["PRECIP"].to_numpy(dtype=float)
    t = np.where(np.isfinite(t), t, 0.0)
    r = np.where(np.isfinite(recon), recon, 0.0)
    ymd = truth["YYYYMMDD"].astype(str)
    ym = ymd.str[:6]
    frame = pd.DataFrame({"ym": ym, "t": t, "r": r}).groupby("ym", as_index=False).sum()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(frame["t"], frame["r"], s=18, alpha=0.6, edgecolors="none")
    lo = float(min(frame["t"].min(), frame["r"].min()))
    hi = float(max(frame["t"].max(), frame["r"].max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Truth monthly PRECIP (mm)")
    ax.set_ylabel("Reconstructed monthly PRECIP (mm)")
    ax.set_title(f"Monthly precip mass — {scenario}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_precip_wetday_qq(payload: dict, scenario: str, out_path: Path) -> None:
    """Q–Q of wet-day amounts on held-out days (distributional, not paired)."""
    truth = payload["truth"]
    imputed = payload["imputed"]
    mask = payload["mask"]
    if "PRECIP" not in truth.columns:
        return
    filled = _eval_mask(truth, imputed, mask, "PRECIP")
    if filled.sum() < 10:
        return
    yt = truth.loc[filled, "PRECIP"].to_numpy(dtype=float)
    yh = imputed.loc[filled, "PRECIP"].to_numpy(dtype=float)
    wet_t = yt[yt >= WET_MM]
    wet_h = yh[yh >= WET_MM]
    if wet_t.size < 5 or wet_h.size < 5:
        return
    qs = np.linspace(0.05, 0.95, 19)
    qt = np.quantile(wet_t, qs)
    qh = np.quantile(wet_h, qs)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].plot(qt, qh, "o", ms=5)
    lo = float(min(qt.min(), qh.min()))
    hi = float(max(qt.max(), qh.max()))
    axes[0].plot([lo, hi], [lo, hi], "k--")
    axes[0].set_xlabel("Truth wet-day quantiles (mm)")
    axes[0].set_ylabel("Imputed wet-day quantiles (mm)")
    axes[0].set_title("Wet-day intensity Q–Q")
    axes[0].grid(alpha=0.3)
    # Contingency-style: wet frequency comparison is text; show amount hist
    axes[1].hist(wet_t, bins=30, alpha=0.55, label="truth wet", density=True)
    axes[1].hist(wet_h, bins=30, alpha=0.55, label="imputed wet", density=True)
    axes[1].set_xlabel("Wet-day amount (mm)")
    axes[1].set_ylabel("Density")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Wet-day amount distribution")
    axes[1].grid(alpha=0.3)
    fig.suptitle(f"PRECIP wet-day climate — {scenario}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_wind_sectors(payload: dict, scenario: str, out_path: Path) -> None:
    """Sector frequency: held-out truth vs imputed (not day-paired)."""
    truth = payload["truth"]
    imputed = payload["imputed"]
    mask = payload["mask"]
    if "W_DIR" not in truth.columns:
        return
    filled = _eval_mask(truth, imputed, mask, "W_DIR")
    if filled.sum() < 10:
        return
    yt = truth.loc[filled, "W_DIR"].to_numpy(dtype=float) % 360.0
    yh = imputed.loc[filled, "W_DIR"].to_numpy(dtype=float) % 360.0
    edges = np.linspace(0, 360, N_WIND_SECTORS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ht, _ = np.histogram(yt, bins=edges, density=False)
    hh, _ = np.histogram(yh, bins=edges, density=False)
    pt = ht / ht.sum()
    ph = hh / hh.sum()
    width = 0.35 * (360 / N_WIND_SECTORS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(centers - width / 2, pt, width=width, label="truth", alpha=0.8)
    axes[0].bar(centers + width / 2, ph, width=width, label="imputed", alpha=0.8)
    axes[0].set_xlabel("Direction sector center (°)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Wind direction sectors (held-out sample)")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    if "W_VEL" in truth.columns:
        filled_v = _eval_mask(truth, imputed, mask, "W_VEL")
        vt = truth.loc[filled_v, "W_VEL"].to_numpy(dtype=float)
        vh = imputed.loc[filled_v, "W_VEL"].to_numpy(dtype=float)
        qs = np.linspace(0.05, 0.95, 19)
        axes[1].plot(np.quantile(vt, qs), np.quantile(vh, qs), "o", ms=5)
        lo = float(min(vt.min(), vh.min()))
        hi = float(max(vt.max(), vh.max()))
        axes[1].plot([lo, hi], [lo, hi], "k--")
        axes[1].set_xlabel("Truth speed quantiles")
        axes[1].set_ylabel("Imputed speed quantiles")
        axes[1].set_title("W_VEL Q–Q (held-out sample)")
        axes[1].grid(alpha=0.3)
    fig.suptitle(f"Wind climate — {scenario}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pointwise_scatter(payload: dict, scenario: str, out_path: Path) -> None:
    """Truth vs imputed only for pointwise variables."""
    truth = payload["truth"]
    imputed = payload["imputed"]
    mask = payload["mask"]
    cols = [c for c in sorted(POINTWISE_VARS) if c in truth.columns]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(3.5 * len(cols), 3.5))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        m = _eval_mask(truth, imputed, mask, col)
        if not m.any():
            ax.set_title(f"{col} (empty)")
            continue
        y = truth.loc[m, col].to_numpy()
        yhat = imputed.loc[m, col].to_numpy()
        ax.scatter(y, yhat, s=8, alpha=0.35, edgecolors="none")
        lo = float(min(y.min(), yhat.min()))
        hi = float(max(y.max(), yhat.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("Truth")
        ax.set_ylabel("Imputed")
        ax.set_title(col)
        ax.grid(alpha=0.25)
    fig.suptitle(f"Pointwise truth vs imputed — {scenario}", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries_excerpt(
    payload: dict,
    scenario: str,
    out_path: Path,
    *,
    cols: tuple[str, ...] = ("MIN_TEMP", "MAX_TEMP", "SO_RAD", "DP_TEMP"),
    start: int = 200,
    length: int = 120,
) -> None:
    truth = payload["truth"]
    imputed = payload["imputed"]
    mask = payload["mask"]
    cols = [c for c in cols if c in truth.columns]
    sl = slice(start, start + length)
    dates = pd.to_datetime(truth["YYYYMMDD"].astype(str).iloc[sl], format="%Y%m%d")
    fig, axes = plt.subplots(len(cols), 1, figsize=(12, 2.2 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        y_true = truth[col].iloc[sl].to_numpy()
        y_imp = imputed[col].iloc[sl].to_numpy()
        m = mask[col].iloc[sl].to_numpy()
        ax.plot(dates, y_true, color="0.35", lw=1.2, label="truth")
        ax.plot(dates, np.where(m, y_imp, np.nan), "o", ms=3.5, color="C3", label="imputed")
        ax.set_ylabel(col)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"Gap-fill excerpt (pointwise vars) — {scenario}")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------


def _df_to_md(df: pd.DataFrame, float_cols: set[str] | None = None) -> str:
    if df.empty:
        return "_(empty)_"
    float_cols = float_cols or {
        "rmse",
        "mae",
        "bias",
        "r2",
        "fill_rate",
        "value",
        "n_amputed",
        "n_filled",
    }
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in float_cols and isinstance(v, (float, np.floating)) and np.isfinite(v):
                cells.append(f"{float(v):.3f}")
            elif c in ("n_amputed", "n_filled") and isinstance(v, (int, float, np.integer, np.floating)):
                cells.append(str(int(v)))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def write_summary_markdown(
    pointwise: pd.DataFrame,
    climate: pd.DataFrame,
    precip_bp: pd.DataFrame,
    out_path: Path,
    pymice_info: str,
) -> None:
    lines = [
        "# AutoTest.1 Ampute / Impute Validation",
        "",
        f"**PyMICE:** {pymice_info}",
        "",
        "Inputs: standard WEPPCLIFF exports "
        "(`AutoTest.1-Daily-Time-Series.csv`, `AutoTest.1-Precipitation.csv`).",
        "",
        "## How scoring works",
        "",
        "Imputation is **not** judged the same way for every variable.",
        "",
        "| Family | Variables | Primary test | What we deliberately avoid |",
        "| --- | --- | --- | --- |",
        "| **Pointwise** | MIN_TEMP, MAX_TEMP, DP_TEMP, SO_RAD | Day-matched RMSE / MAE / R² on held-out cells | — |",
        "| **Precip climate** | PRECIP | Wet/dry accuracy & F1; monthly & seasonal mass; wet-day intensity KS / quantiles | Day-matched mm RMSE |",
        "| **Storm distribution** | BPS, ELEN, EMAX, EMIN | KS & quantile match of held-out samples | Day-paired storm identity |",
        "| **Wind climate** | W_VEL, W_DIR | Speed KS / mean ratio / monthly mean bias; direction sector TVD & monthly circular-mean error | Day-matched m/s or degree error |",
        "",
        "Method: cluster labels on complete data → ampute → "
        "`impute_by_cluster(..., midastouch|cart)` → score as above.",
        "",
        "## Pointwise (temps / dew / solar)",
        "",
    ]
    main_pw = pointwise[
        ~pointwise["scenario"].astype(str).str.startswith("variable_absent:")
    ] if not pointwise.empty else pointwise
    if not main_pw.empty:
        show = main_pw[
            ["scenario", "variable", "n_filled", "fill_rate", "rmse", "mae", "bias", "r2"]
        ]
        lines.append(_df_to_md(show))
        lines.append("")

    lines.append("## PRECIP — occurrence, mass, wet-day distribution")
    lines.append("")
    if not climate.empty:
        precip = climate[
            (climate["variable"] == "PRECIP")
            & (~climate["scenario"].astype(str).str.startswith("variable_absent:"))
        ]
        # Wide-ish summary of key metrics
        key_metrics = [
            "wet_dry_accuracy",
            "wet_f1",
            "wet_freq_bias",
            "monthly_mass_bias",
            "holdout_mass_ratio",
            "wet_intensity_ks",
            "wet_intensity_qmae",
            "wet_mean_intensity_bias",
            "seasonal_mass_ratio_DJF",
            "seasonal_mass_ratio_MAM",
            "seasonal_mass_ratio_JJA",
            "seasonal_mass_ratio_SON",
        ]
        pivot_rows = []
        for scen in precip["scenario"].unique():
            row = {"scenario": scen}
            for m in key_metrics:
                v = precip[(precip["scenario"] == scen) & (precip["metric"] == m)]
                row[m] = float(v["value"].iloc[0]) if len(v) else np.nan
            pivot_rows.append(row)
        if pivot_rows:
            lines.append(_df_to_md(pd.DataFrame(pivot_rows), float_cols=set(key_metrics)))
            lines.append("")
        lines.append(
            "Interpretation: `wet_dry_accuracy` / `wet_f1` near 1 = occurrence skill; "
            "`monthly_mass_bias` near 0 = good monthly totals; "
            "`holdout_mass_ratio` near 1 = aggregate mass conserved; "
            "`wet_intensity_ks` near 0 = wet-day depth distribution matches."
        )
        lines.append("")

    lines.append("## Wind — speed distribution & direction climatology")
    lines.append("")
    if not climate.empty:
        wind = climate[
            (climate["variable"].isin(["W_VEL", "W_DIR"]))
            & (~climate["scenario"].astype(str).str.startswith("variable_absent:"))
        ]
        show_w = wind[
            ["scenario", "variable", "metric", "value", "n_filled", "detail"]
        ]
        lines.append(_df_to_md(show_w, float_cols={"value", "n_filled"}))
        lines.append("")
        lines.append(
            "Interpretation: `speed_ks` / `sector_tvd` near 0 = distribution match; "
            "`speed_mean_ratio` near 1 = mean wind conserved; "
            "`monthly_circular_mean_error_deg` small = monthly prevailing direction OK."
        )
        lines.append("")

    lines.append("## Storm attributes (distributional)")
    lines.append("")
    if not climate.empty:
        storm = climate[
            (climate["metric_family"] == "storm_distribution")
            & (~climate["scenario"].astype(str).str.startswith("variable_absent:"))
        ]
        if not storm.empty:
            lines.append(
                _df_to_md(
                    storm[["scenario", "variable", "metric", "value", "n_filled"]],
                    float_cols={"value", "n_filled"},
                )
            )
            lines.append("")

    if not precip_bp.empty:
        lines.append("## Precipitation breakpoint panel (distributional MCAR)")
        lines.append("")
        lines.append(
            _df_to_md(
                precip_bp[["variable", "metric", "value", "n_filled", "fill_rate"]],
                float_cols={"value", "n_filled", "fill_rate"},
            )
        )
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- PRECIP and storm fields are naturally missing on dry days; amputation "
        "only scores cells that were observed in the complete export."
    )
    lines.append(
        "- Day-matched PRECIP/W_VEL/W_DIR RMSE is intentionally **not** reported as "
        "a primary metric: weather generators and gap-fills are judged on climate "
        "fidelity, not replaying the exact held-out day."
    )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import pymice

    pymice_info = (
        f"{getattr(pymice, '__version__', 'unknown')} "
        f"({Path(pymice.__file__).resolve()})"
    )
    print(f"PyMICE: {pymice_info}")
    print(f"Output: {OUT_DIR}")
    print(
        "Scoring: pointwise temps/solar; PRECIP occurrence+mass+wet-day dist; "
        "wind distribution+climatology (no day-matched precip/wind RMSE)."
    )

    truth = load_daily_panel()
    print(f"Daily panel: {len(truth)} days, cols={list(truth.columns)}")
    print("Running daily ampute/impute scenarios...")
    pointwise, climate, daily_payloads = run_daily_scenarios(truth)

    precip_bp = pd.DataFrame()
    precip_payloads: dict = {}
    precip = load_precip_panel()
    if precip is not None:
        print(f"Precip panel sample: {len(precip)} rows")
        print("Running precip MCAR ampute/impute...")
        t0 = time.perf_counter()
        precip_bp, precip_payloads = run_precip_mcar(precip)
        print(f"  precip mcar: {time.perf_counter() - t0:.1f}s")

    pointwise.to_csv(OUT_DIR / "metrics_pointwise.csv", index=False)
    climate.to_csv(OUT_DIR / "metrics_climate.csv", index=False)
    if not precip_bp.empty:
        precip_bp.to_csv(OUT_DIR / "metrics_precip_breakpoints.csv", index=False)
    print(f"Wrote metrics_*.csv under {OUT_DIR}")

    write_summary_markdown(
        pointwise, climate, precip_bp, OUT_DIR / "SUMMARY.md", pymice_info
    )

    plot_pointwise_bars(pointwise, OUT_DIR / "fig_pointwise_rmse_mae.png")
    plot_precip_climate(climate, OUT_DIR / "fig_precip_climate.png")
    plot_wind_climate(climate, OUT_DIR / "fig_wind_climate.png")

    for scen_key in ("mcar_ampute", "short_gaps", "long_gap_all"):
        if scen_key not in daily_payloads:
            continue
        plot_pointwise_scatter(
            daily_payloads[scen_key],
            scen_key,
            OUT_DIR / f"fig_scatter_pointwise_{scen_key}.png",
        )
        plot_timeseries_excerpt(
            daily_payloads[scen_key],
            scen_key,
            OUT_DIR / f"fig_timeseries_{scen_key}.png",
        )
        plot_precip_monthly_scatter(
            daily_payloads[scen_key],
            scen_key,
            OUT_DIR / f"fig_precip_monthly_{scen_key}.png",
        )
        plot_precip_wetday_qq(
            daily_payloads[scen_key],
            scen_key,
            OUT_DIR / f"fig_precip_wetday_{scen_key}.png",
        )
        plot_wind_sectors(
            daily_payloads[scen_key],
            scen_key,
            OUT_DIR / f"fig_wind_sectors_{scen_key}.png",
        )

    if "mcar_ampute" in precip_payloads:
        # Distributional Q–Q for breakpoint PRECIP
        p = precip_payloads["mcar_ampute"]
        filled = _eval_mask(p["truth"], p["imputed"], p["mask"], "PRECIP")
        if filled.sum() >= 10:
            yt = p["truth"].loc[filled, "PRECIP"].to_numpy(dtype=float)
            yh = p["imputed"].loc[filled, "PRECIP"].to_numpy(dtype=float)
            qs = np.linspace(0.05, 0.95, 19)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.plot(np.quantile(yt, qs), np.quantile(yh, qs), "o", ms=5)
            lo = float(min(yt.min(), yh.min()))
            hi = float(max(yt.max(), yh.max()))
            ax.plot([lo, hi], [lo, hi], "k--")
            ax.set_xlabel("Truth PRECIP quantiles (breakpoints)")
            ax.set_ylabel("Imputed PRECIP quantiles")
            ax.set_title("Breakpoint PRECIP Q–Q (MCAR held-out sample)")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUT_DIR / "fig_precip_bp_qq_mcar.png", dpi=150)
            plt.close(fig)

    print("Done. See SUMMARY.md for interpretation.")
    # Console highlight table
    if not climate.empty:
        print("\n--- PRECIP key metrics ---")
        for scen in climate.loc[
            climate["variable"] == "PRECIP", "scenario"
        ].unique():
            if str(scen).startswith("variable_absent"):
                continue
            acc = _metric_value(climate[climate["scenario"] == scen], "PRECIP", "wet_dry_accuracy")
            mass = _metric_value(climate[climate["scenario"] == scen], "PRECIP", "monthly_mass_bias")
            ratio = _metric_value(climate[climate["scenario"] == scen], "PRECIP", "holdout_mass_ratio")
            ks = _metric_value(climate[climate["scenario"] == scen], "PRECIP", "wet_intensity_ks")
            print(
                f"  {scen}: wet_acc={acc:.3f} monthly_mass_bias={mass:.3f} "
                f"holdout_mass_ratio={ratio:.3f} wet_KS={ks:.3f}"
            )
        print("\n--- Wind key metrics ---")
        for scen in climate["scenario"].unique():
            if str(scen).startswith("variable_absent"):
                continue
            sks = _metric_value(climate[climate["scenario"] == scen], "W_VEL", "speed_ks")
            smr = _metric_value(climate[climate["scenario"] == scen], "W_VEL", "speed_mean_ratio")
            tvd = _metric_value(climate[climate["scenario"] == scen], "W_DIR", "sector_tvd")
            mce = _metric_value(
                climate[climate["scenario"] == scen], "W_DIR", "monthly_circular_mean_error_deg"
            )
            print(
                f"  {scen}: speed_KS={sks:.3f} speed_mean_ratio={smr:.3f} "
                f"sector_tvd={tvd:.3f} monthly_dir_err={mce:.1f}°"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
