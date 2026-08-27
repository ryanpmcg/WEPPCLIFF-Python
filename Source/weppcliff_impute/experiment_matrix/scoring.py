"""Dual-scope climate scoring: whole series vs hole-only, multi-period.

Every mass/mean climate statistic is emitted twice:

* **scope=whole** — reconstructed full series vs truth (dilution by intact
  periods is expected; answers “does the export still look like climate?”)
* **scope=hole** — only calendar days in the amputed hole (answers “did we
  reconstruct the missing period?”)

Periods: daily (holdout cell aggregates), monthly, seasonal, annual.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

WET_MM = 0.25
POINTWISE_VARS = frozenset({"MIN_TEMP", "MAX_TEMP", "DP_TEMP", "SO_RAD"})
STORM_VARS = frozenset({"BPS", "ELEN", "EMAX", "EMIN"})
WIND_VARS = frozenset({"W_VEL", "W_DIR"})
N_WIND_SECTORS = 8
SEASON_MONTHS = {
    "DJF": {12, 1, 2},
    "MAM": {3, 4, 5},
    "JJA": {6, 7, 8},
    "SON": {9, 10, 11},
}


def _eval_mask(
    truth: pd.DataFrame, imputed: pd.DataFrame, missing_mask: pd.DataFrame, col: str
) -> np.ndarray:
    return (
        missing_mask[col].to_numpy(dtype=bool)
        & truth[col].notna().to_numpy()
        & imputed[col].notna().to_numpy()
    )


def _reconstruct(
    truth: pd.DataFrame, imputed: pd.DataFrame, missing_mask: pd.DataFrame, col: str
) -> np.ndarray:
    y = truth[col].to_numpy(dtype=float).copy()
    m = missing_mask[col].to_numpy(dtype=bool)
    yhat = imputed[col].to_numpy(dtype=float)
    filled = m & np.isfinite(yhat)
    y[filled] = yhat[filled]
    return y


def _ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 5 or b.size < 5:
        return float("nan")
    return float(stats.ks_2samp(a, b).statistic)


def _quantile_mae(a: np.ndarray, b: np.ndarray) -> float:
    qs = np.linspace(0.1, 0.9, 9)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 5 or b.size < 5:
        return float("nan")
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def _circular_mean_deg(angles: np.ndarray) -> float:
    a = np.asarray(angles, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    rad = np.deg2rad(a)
    mean = np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))
    return float(np.rad2deg(mean)) % 360.0


def _circular_diff_deg(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _ymd_parts(ymd: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = np.asarray(ymd, dtype=str)
    years = np.array([int(x[:4]) if len(x) >= 4 and x[:4].isdigit() else 0 for x in s])
    months = np.array([int(x[4:6]) if len(x) >= 6 and x[4:6].isdigit() else 0 for x in s])
    yyyymm = np.array([x[:6] if len(x) >= 6 else "000000" for x in s])
    return years, months, yyyymm


def _season_label(month: int) -> str:
    for name, mset in SEASON_MONTHS.items():
        if month in mset:
            return name
    return "UNK"


def _row(
    *,
    experiment: str,
    engine: str,
    scenario: str,
    seed: int,
    variable: str,
    metric: str,
    value: float,
    scope: str,
    period: str,
    metric_family: str,
    n: int = 0,
    detail: str = "",
) -> dict:
    return {
        "experiment": experiment,
        "engine": engine,
        "scenario": scenario,
        "seed": seed,
        "variable": variable,
        "metric": metric,
        "value": value,
        "scope": scope,  # whole | hole | holdout_cells
        "period": period,  # daily | monthly | seasonal | annual | sample
        "metric_family": metric_family,
        "n": n,
        "detail": detail,
    }


def _period_mass_metrics(
    *,
    experiment: str,
    engine: str,
    scenario: str,
    seed: int,
    variable: str,
    ymd: np.ndarray,
    truth_mass: np.ndarray,
    recon_mass: np.ndarray,
    day_sel: np.ndarray,
    scope: str,
    metric_family: str = "mass_climate",
) -> list[dict]:
    """Mass ratio + relative bias at monthly / seasonal / annual under day_sel."""
    rows: list[dict] = []
    if not np.any(day_sel):
        for period in ("monthly", "seasonal", "annual"):
            rows.append(
                _row(
                    experiment=experiment,
                    engine=engine,
                    scenario=scenario,
                    seed=seed,
                    variable=variable,
                    metric="mass_ratio",
                    value=float("nan"),
                    scope=scope,
                    period=period,
                    metric_family=metric_family,
                    n=0,
                    detail="empty selection",
                )
            )
        return rows

    years, months, yyyymm = _ymd_parts(ymd)
    t = np.where(np.isfinite(truth_mass), truth_mass, 0.0)
    r = np.where(np.isfinite(recon_mass), recon_mass, 0.0)

    # Daily aggregate on selection (sum)
    st = float(np.sum(t[day_sel]))
    sr = float(np.sum(r[day_sel]))
    rows.append(
        _row(
            experiment=experiment,
            engine=engine,
            scenario=scenario,
            seed=seed,
            variable=variable,
            metric="mass_ratio",
            value=sr / st if st > 1e-9 else float("nan"),
            scope=scope,
            period="daily",
            metric_family=metric_family,
            n=int(day_sel.sum()),
            detail="sum(recon)/sum(truth) over selected days",
        )
    )
    rows.append(
        _row(
            experiment=experiment,
            engine=engine,
            scenario=scenario,
            seed=seed,
            variable=variable,
            metric="mass_bias_abs",
            value=abs(sr - st),
            scope=scope,
            period="daily",
            metric_family=metric_family,
            n=int(day_sel.sum()),
            detail="|sum recon − sum truth|",
        )
    )

    # Monthly
    frame = pd.DataFrame(
        {
            "YYYYMM": yyyymm[day_sel],
            "truth": t[day_sel],
            "recon": r[day_sel],
        }
    )
    monthly = frame.groupby("YYYYMM", as_index=False)[["truth", "recon"]].sum()
    if len(monthly):
        mt_arr = monthly["truth"].to_numpy(dtype=float)
        mr_arr = monthly["recon"].to_numpy(dtype=float)
        # Skip near-dry months for relative bias (avoids 1e6 blow-ups when truth≈0)
        wet_m = np.abs(mt_arr) > 0.5
        if wet_m.any():
            rel = np.abs(mr_arr[wet_m] - mt_arr[wet_m]) / np.abs(mt_arr[wet_m])
            rows.append(
                _row(
                    experiment=experiment,
                    engine=engine,
                    scenario=scenario,
                    seed=seed,
                    variable=variable,
                    metric="mass_rel_bias",
                    value=float(np.nanmean(rel)),
                    scope=scope,
                    period="monthly",
                    metric_family=metric_family,
                    n=int(wet_m.sum()),
                    detail="mean |m_recon−m_truth|/|m_truth| over months with |truth|>0.5",
                )
            )
        else:
            rows.append(
                _row(
                    experiment=experiment,
                    engine=engine,
                    scenario=scenario,
                    seed=seed,
                    variable=variable,
                    metric="mass_rel_bias",
                    value=float("nan"),
                    scope=scope,
                    period="monthly",
                    metric_family=metric_family,
                    n=0,
                    detail="no months with |truth|>0.5",
                )
            )
        mt, mr = float(mt_arr.sum()), float(mr_arr.sum())
        rows.append(
            _row(
                experiment=experiment,
                engine=engine,
                scenario=scenario,
                seed=seed,
                variable=variable,
                metric="mass_ratio",
                value=mr / mt if mt > 1e-9 else float("nan"),
                scope=scope,
                period="monthly",
                metric_family=metric_family,
                n=len(monthly),
                detail="total over months in scope",
            )
        )

    # Seasonal
    seas = np.array([_season_label(int(m)) for m in months])
    for name in SEASON_MONTHS:
        sel = day_sel & (seas == name)
        if not np.any(sel):
            continue
        st_s = float(np.sum(t[sel]))
        sr_s = float(np.sum(r[sel]))
        rows.append(
            _row(
                experiment=experiment,
                engine=engine,
                scenario=scenario,
                seed=seed,
                variable=variable,
                metric=f"mass_ratio_{name}",
                value=sr_s / st_s if st_s > 1e-9 else float("nan"),
                scope=scope,
                period="seasonal",
                metric_family=metric_family,
                n=int(sel.sum()),
                detail=f"season {name}",
            )
        )

    # Annual
    frame_y = pd.DataFrame({"YYYY": years[day_sel], "truth": t[day_sel], "recon": r[day_sel]})
    annual = frame_y.groupby("YYYY", as_index=False)[["truth", "recon"]].sum()
    if len(annual):
        at_arr = annual["truth"].to_numpy(dtype=float)
        ar_arr = annual["recon"].to_numpy(dtype=float)
        wet_y = np.abs(at_arr) > 0.5
        if wet_y.any():
            rel = np.abs(ar_arr[wet_y] - at_arr[wet_y]) / np.abs(at_arr[wet_y])
            rows.append(
                _row(
                    experiment=experiment,
                    engine=engine,
                    scenario=scenario,
                    seed=seed,
                    variable=variable,
                    metric="mass_rel_bias",
                    value=float(np.nanmean(rel)),
                    scope=scope,
                    period="annual",
                    metric_family=metric_family,
                    n=int(wet_y.sum()),
                    detail="mean |a_recon−a_truth|/|a_truth| over years with |truth|>0.5",
                )
            )
        at, ar = float(at_arr.sum()), float(ar_arr.sum())
        rows.append(
            _row(
                experiment=experiment,
                engine=engine,
                scenario=scenario,
                seed=seed,
                variable=variable,
                metric="mass_ratio",
                value=ar / at if at > 1e-9 else float("nan"),
                scope=scope,
                period="annual",
                metric_family=metric_family,
                n=len(annual),
                detail="total over years in scope",
            )
        )
    return rows


def _period_mean_metrics(
    *,
    experiment: str,
    engine: str,
    scenario: str,
    seed: int,
    variable: str,
    ymd: np.ndarray,
    truth_arr: np.ndarray,
    recon_arr: np.ndarray,
    day_sel: np.ndarray,
    scope: str,
    metric_family: str = "mean_climate",
) -> list[dict]:
    """Mean-field relative bias at monthly / seasonal / annual (temps, wind speed)."""
    rows: list[dict] = []
    if not np.any(day_sel):
        return rows
    years, months, yyyymm = _ymd_parts(ymd)
    t = truth_arr.astype(float)
    r = recon_arr.astype(float)

    # Holdout-style daily RMSE on selection where both finite
    both = day_sel & np.isfinite(t) & np.isfinite(r)
    if both.any():
        resid = r[both] - t[both]
        rows.append(
            _row(
                experiment=experiment,
                engine=engine,
                scenario=scenario,
                seed=seed,
                variable=variable,
                metric="rmse",
                value=float(np.sqrt(np.mean(resid**2))),
                scope=scope,
                period="daily",
                metric_family=metric_family,
                n=int(both.sum()),
            )
        )
        rows.append(
            _row(
                experiment=experiment,
                engine=engine,
                scenario=scenario,
                seed=seed,
                variable=variable,
                metric="mae",
                value=float(np.mean(np.abs(resid))),
                scope=scope,
                period="daily",
                metric_family=metric_family,
                n=int(both.sum()),
            )
        )
        rows.append(
            _row(
                experiment=experiment,
                engine=engine,
                scenario=scenario,
                seed=seed,
                variable=variable,
                metric="bias",
                value=float(np.mean(resid)),
                scope=scope,
                period="daily",
                metric_family=metric_family,
                n=int(both.sum()),
            )
        )

    frame = pd.DataFrame(
        {"YYYYMM": yyyymm[day_sel], "truth": t[day_sel], "recon": r[day_sel]}
    )
    monthly = frame.groupby("YYYYMM", as_index=False)[["truth", "recon"]].mean()
    if len(monthly):
        mt = monthly["truth"].to_numpy()
        mr = monthly["recon"].to_numpy()
        ok = np.isfinite(mt) & np.isfinite(mr)
        if ok.any():
            denom = np.maximum(np.abs(mt[ok]), 1e-6)
            rel = np.abs(mr[ok] - mt[ok]) / denom
            rows.append(
                _row(
                    experiment=experiment,
                    engine=engine,
                    scenario=scenario,
                    seed=seed,
                    variable=variable,
                    metric="mean_rel_bias",
                    value=float(np.mean(rel)),
                    scope=scope,
                    period="monthly",
                    metric_family=metric_family,
                    n=int(ok.sum()),
                )
            )

    seas = np.array([_season_label(int(m)) for m in months])
    for name in SEASON_MONTHS:
        sel = day_sel & (seas == name)
        both_s = sel & np.isfinite(t) & np.isfinite(r)
        if both_s.sum() < 3:
            continue
        mt, mr = float(np.nanmean(t[both_s])), float(np.nanmean(r[both_s]))
        rows.append(
            _row(
                experiment=experiment,
                engine=engine,
                scenario=scenario,
                seed=seed,
                variable=variable,
                metric=f"mean_ratio_{name}",
                value=mr / mt if abs(mt) > 1e-9 else float("nan"),
                scope=scope,
                period="seasonal",
                metric_family=metric_family,
                n=int(both_s.sum()),
            )
        )

    frame_y = pd.DataFrame({"YYYY": years[day_sel], "truth": t[day_sel], "recon": r[day_sel]})
    annual = frame_y.groupby("YYYY", as_index=False)[["truth", "recon"]].mean()
    if len(annual):
        mt = annual["truth"].to_numpy()
        mr = annual["recon"].to_numpy()
        ok = np.isfinite(mt) & np.isfinite(mr)
        if ok.any():
            denom = np.maximum(np.abs(mt[ok]), 1e-6)
            rel = np.abs(mr[ok] - mt[ok]) / denom
            rows.append(
                _row(
                    experiment=experiment,
                    engine=engine,
                    scenario=scenario,
                    seed=seed,
                    variable=variable,
                    metric="mean_rel_bias",
                    value=float(np.mean(rel)),
                    scope=scope,
                    period="annual",
                    metric_family=metric_family,
                    n=int(ok.sum()),
                )
            )
    return rows


def score_pointwise_holdout(
    truth: pd.DataFrame,
    imputed: pd.DataFrame,
    missing_mask: pd.DataFrame,
    cols: Iterable[str],
    *,
    experiment: str,
    engine: str,
    scenario: str,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    for col in cols:
        if col not in POINTWISE_VARS or col not in truth.columns:
            continue
        filled = _eval_mask(truth, imputed, missing_mask, col)
        n_miss = int((missing_mask[col].to_numpy() & truth[col].notna().to_numpy()).sum())
        n_fill = int(filled.sum())
        if n_fill == 0:
            for m in ("rmse", "mae", "bias", "r2"):
                rows.append(
                    _row(
                        experiment=experiment,
                        engine=engine,
                        scenario=scenario,
                        seed=seed,
                        variable=col,
                        metric=m,
                        value=float("nan"),
                        scope="holdout_cells",
                        period="daily",
                        metric_family="pointwise",
                        n=0,
                    )
                )
            continue
        y = truth.loc[filled, col].to_numpy(dtype=float)
        yhat = imputed.loc[filled, col].to_numpy(dtype=float)
        resid = yhat - y
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        metrics = {
            "rmse": float(np.sqrt(np.mean(resid**2))),
            "mae": float(np.mean(np.abs(resid))),
            "bias": float(np.mean(resid)),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "fill_rate": n_fill / n_miss if n_miss else float("nan"),
        }
        for m, v in metrics.items():
            rows.append(
                _row(
                    experiment=experiment,
                    engine=engine,
                    scenario=scenario,
                    seed=seed,
                    variable=col,
                    metric=m,
                    value=v,
                    scope="holdout_cells",
                    period="daily",
                    metric_family="pointwise",
                    n=n_fill,
                )
            )
    return rows


def score_precip_holdout_and_climate(
    truth: pd.DataFrame,
    imputed: pd.DataFrame,
    missing_mask: pd.DataFrame,
    hole_day: np.ndarray,
    *,
    experiment: str,
    engine: str,
    scenario: str,
    seed: int,
    col: str = "PRECIP",
    wet_mm: float = WET_MM,
) -> list[dict]:
    rows: list[dict] = []
    if col not in truth.columns:
        return rows
    ymd = truth["YYYYMMDD"].astype(str).to_numpy()
    filled = _eval_mask(truth, imputed, missing_mask, col)
    n_fill = int(filled.sum())

    if n_fill > 0:
        yt = truth.loc[filled, col].to_numpy(dtype=float)
        yh = imputed.loc[filled, col].to_numpy(dtype=float)
        wet_t = yt >= wet_mm
        wet_h = yh >= wet_mm
        acc = float(np.mean(wet_t == wet_h))
        tp = float(np.sum(wet_t & wet_h))
        fp = float(np.sum(~wet_t & wet_h))
        fn = float(np.sum(wet_t & ~wet_h))
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (
            2 * prec * rec / (prec + rec)
            if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0
            else float("nan")
        )
        for m, v in (
            ("wet_dry_accuracy", acc),
            ("wet_f1", f1),
            ("wet_freq_bias", float(np.mean(wet_h) - np.mean(wet_t))),
            ("wet_intensity_ks", _ks_stat(yt[wet_t], yh[wet_h])),
            ("wet_intensity_qmae", _quantile_mae(yt[wet_t], yh[wet_h])),
            ("holdout_mass_ratio", float(np.sum(yh)) / float(np.sum(yt)) if float(np.sum(yt)) > 1e-9 else float("nan")),
            ("n_holdout_wet_truth", float(wet_t.sum())),
            ("n_holdout_wet_imputed", float(wet_h.sum())),
        ):
            rows.append(
                _row(
                    experiment=experiment,
                    engine=engine,
                    scenario=scenario,
                    seed=seed,
                    variable=col,
                    metric=m,
                    value=v,
                    scope="holdout_cells",
                    period="daily",
                    metric_family="precip_climate",
                    n=n_fill,
                )
            )

    recon = _reconstruct(truth, imputed, missing_mask, col)
    truth_arr = truth[col].to_numpy(dtype=float)
    whole = np.ones(len(truth), dtype=bool)
    hole = np.asarray(hole_day, dtype=bool)
    if hole.shape[0] != len(truth):
        hole = missing_mask.to_numpy(dtype=bool).any(axis=1)

    for scope, sel in (("whole", whole), ("hole", hole)):
        rows.extend(
            _period_mass_metrics(
                experiment=experiment,
                engine=engine,
                scenario=scenario,
                seed=seed,
                variable=col,
                ymd=ymd,
                truth_mass=truth_arr,
                recon_mass=recon,
                day_sel=sel,
                scope=scope,
            )
        )
    return rows


def score_storm_holdout(
    truth: pd.DataFrame,
    imputed: pd.DataFrame,
    missing_mask: pd.DataFrame,
    cols: Iterable[str],
    *,
    experiment: str,
    engine: str,
    scenario: str,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    for col in cols:
        if col not in STORM_VARS or col not in truth.columns:
            continue
        filled = _eval_mask(truth, imputed, missing_mask, col)
        n_fill = int(filled.sum())
        if n_fill < 5:
            for m in ("ks", "mean_ratio", "quantile_mae"):
                rows.append(
                    _row(
                        experiment=experiment,
                        engine=engine,
                        scenario=scenario,
                        seed=seed,
                        variable=col,
                        metric=m,
                        value=float("nan"),
                        scope="holdout_cells",
                        period="sample",
                        metric_family="storm_distribution",
                        n=n_fill,
                        detail="too few",
                    )
                )
            continue
        yt = truth.loc[filled, col].to_numpy(dtype=float)
        yh = imputed.loc[filled, col].to_numpy(dtype=float)
        mt, mh = float(np.mean(yt)), float(np.mean(yh))
        for m, v in (
            ("ks", _ks_stat(yt, yh)),
            ("quantile_mae", _quantile_mae(yt, yh)),
            ("mean_ratio", mh / mt if abs(mt) > 1e-12 else float("nan")),
            ("max_ratio", float(np.max(yh)) / float(np.max(yt)) if float(np.max(yt)) > 1e-12 else float("nan")),
        ):
            rows.append(
                _row(
                    experiment=experiment,
                    engine=engine,
                    scenario=scenario,
                    seed=seed,
                    variable=col,
                    metric=m,
                    value=v,
                    scope="holdout_cells",
                    period="sample",
                    metric_family="storm_distribution",
                    n=n_fill,
                )
            )
    return rows


def score_wind_dual(
    truth: pd.DataFrame,
    imputed: pd.DataFrame,
    missing_mask: pd.DataFrame,
    hole_day: np.ndarray,
    *,
    experiment: str,
    engine: str,
    scenario: str,
    seed: int,
) -> list[dict]:
    rows: list[dict] = []
    ymd = truth["YYYYMMDD"].astype(str).to_numpy()
    whole = np.ones(len(truth), dtype=bool)
    hole = np.asarray(hole_day, dtype=bool)
    if hole.shape[0] != len(truth):
        hole = missing_mask.to_numpy(dtype=bool).any(axis=1)

    if "W_VEL" in truth.columns:
        col = "W_VEL"
        filled = _eval_mask(truth, imputed, missing_mask, col)
        n_fill = int(filled.sum())
        if n_fill >= 5:
            yt = truth.loc[filled, col].to_numpy(dtype=float)
            yh = imputed.loc[filled, col].to_numpy(dtype=float)
            mt, mh = float(np.mean(yt)), float(np.mean(yh))
            for m, v in (
                ("speed_ks", _ks_stat(yt, yh)),
                ("speed_mean_ratio", mh / mt if mt > 1e-9 else float("nan")),
                ("speed_quantile_mae", _quantile_mae(yt, yh)),
            ):
                rows.append(
                    _row(
                        experiment=experiment,
                        engine=engine,
                        scenario=scenario,
                        seed=seed,
                        variable=col,
                        metric=m,
                        value=v,
                        scope="holdout_cells",
                        period="sample",
                        metric_family="wind_climate",
                        n=n_fill,
                    )
                )
        recon = _reconstruct(truth, imputed, missing_mask, col)
        t_arr = truth[col].to_numpy(dtype=float)
        for scope, sel in (("whole", whole), ("hole", hole)):
            rows.extend(
                _period_mean_metrics(
                    experiment=experiment,
                    engine=engine,
                    scenario=scenario,
                    seed=seed,
                    variable=col,
                    ymd=ymd,
                    truth_arr=t_arr,
                    recon_arr=recon,
                    day_sel=sel,
                    scope=scope,
                )
            )

    if "W_DIR" in truth.columns:
        col = "W_DIR"
        filled = _eval_mask(truth, imputed, missing_mask, col)
        n_fill = int(filled.sum())
        if n_fill >= 5:
            yt = truth.loc[filled, col].to_numpy(dtype=float) % 360.0
            yh = imputed.loc[filled, col].to_numpy(dtype=float) % 360.0
            edges = np.linspace(0, 360, N_WIND_SECTORS + 1)
            ht, _ = np.histogram(yt, bins=edges, density=True)
            hh, _ = np.histogram(yh, bins=edges, density=True)
            width = 360.0 / N_WIND_SECTORS
            pt, ph = ht * width, hh * width
            pt = pt / pt.sum() if pt.sum() > 0 else pt
            ph = ph / ph.sum() if ph.sum() > 0 else ph
            tvd = 0.5 * float(np.sum(np.abs(pt - ph)))
            rows.append(
                _row(
                    experiment=experiment,
                    engine=engine,
                    scenario=scenario,
                    seed=seed,
                    variable=col,
                    metric="sector_tvd",
                    value=tvd,
                    scope="holdout_cells",
                    period="sample",
                    metric_family="wind_climate",
                    n=n_fill,
                )
            )
            rows.append(
                _row(
                    experiment=experiment,
                    engine=engine,
                    scenario=scenario,
                    seed=seed,
                    variable=col,
                    metric="holdout_circular_mean_error_deg",
                    value=_circular_diff_deg(_circular_mean_deg(yt), _circular_mean_deg(yh)),
                    scope="holdout_cells",
                    period="sample",
                    metric_family="wind_climate",
                    n=n_fill,
                )
            )

        recon = _reconstruct(truth, imputed, missing_mask, col)
        t_arr = truth[col].to_numpy(dtype=float)
        _, _, yyyymm = _ymd_parts(ymd)
        for scope, sel in (("whole", whole), ("hole", hole)):
            errs = []
            for ym in np.unique(yyyymm[sel]):
                msel = sel & (yyyymm == ym)
                e = _circular_diff_deg(_circular_mean_deg(t_arr[msel]), _circular_mean_deg(recon[msel]))
                if np.isfinite(e):
                    errs.append(e)
            rows.append(
                _row(
                    experiment=experiment,
                    engine=engine,
                    scenario=scenario,
                    seed=seed,
                    variable=col,
                    metric="circular_mean_error_deg",
                    value=float(np.mean(errs)) if errs else float("nan"),
                    scope=scope,
                    period="monthly",
                    metric_family="wind_climate",
                    n=len(errs),
                )
            )
    return rows


def score_breakpoint_panel(
    truth: pd.DataFrame,
    imputed: pd.DataFrame,
    missing_mask: pd.DataFrame,
    value_cols: list[str],
    *,
    experiment: str,
    engine: str,
    scenario: str,
    seed: int,
) -> list[dict]:
    """Distributional metrics for breakpoint fields (PRECIP, DUR, INT, ANT, BTW)."""
    rows: list[dict] = []
    for col in value_cols:
        if col not in truth.columns:
            continue
        filled = _eval_mask(truth, imputed, missing_mask, col)
        n_fill = int(filled.sum())
        if n_fill < 5:
            continue
        yt = truth.loc[filled, col].to_numpy(dtype=float)
        yh = imputed.loc[filled, col].to_numpy(dtype=float)
        mt, mh = float(np.mean(yt)), float(np.mean(yh))
        for m, v in (
            ("ks", _ks_stat(yt, yh)),
            ("quantile_mae", _quantile_mae(yt, yh)),
            ("mean_ratio", mh / mt if abs(mt) > 1e-12 else float("nan")),
            (
                "holdout_mass_ratio",
                float(np.sum(yh)) / float(np.sum(yt)) if float(np.sum(yt)) > 1e-9 else float("nan"),
            ),
        ):
            rows.append(
                _row(
                    experiment=experiment,
                    engine=engine,
                    scenario=scenario,
                    seed=seed,
                    variable=col,
                    metric=m,
                    value=v,
                    scope="holdout_cells",
                    period="sample",
                    metric_family="breakpoint_distribution",
                    n=n_fill,
                )
            )
    # I30 proxy on held-out days if DUR+PRECIP present
    if {"PRECIP", "DUR"}.issubset(truth.columns) and "DT_1" in truth.columns:
        try:
            from Source.weppcliff_impute.ei_validate import i30_proxy

            # Compare I30 on days that had any amputed PRECIP
            if "YYYYMMDD" not in truth.columns:
                t = truth.copy()
                t["YYYYMMDD"] = pd.to_datetime(t["DT_1"], errors="coerce").dt.strftime("%Y%m%d")
                i = imputed.copy()
                i["YYYYMMDD"] = pd.to_datetime(i["DT_1"], errors="coerce").dt.strftime("%Y%m%d")
            else:
                t, i = truth, imputed
            miss_days = set(t.loc[missing_mask["PRECIP"].to_numpy(), "YYYYMMDD"].astype(str))
            ratios = []
            for ymd in list(miss_days)[:200]:
                tday = t.loc[t["YYYYMMDD"].astype(str) == ymd, ["DT_1", "DUR", "PRECIP"]]
                iday = i.loc[i["YYYYMMDD"].astype(str) == ymd, ["DT_1", "DUR", "PRECIP"]]
                if len(tday) < 2 or len(iday) < 2:
                    continue
                ti = i30_proxy(tday)
                ii = i30_proxy(iday)
                if ti > 1e-9:
                    ratios.append(abs(ii - ti) / ti)
            if ratios:
                rows.append(
                    _row(
                        experiment=experiment,
                        engine=engine,
                        scenario=scenario,
                        seed=seed,
                        variable="I30",
                        metric="mean_rel_bias",
                        value=float(np.mean(ratios)),
                        scope="holdout_cells",
                        period="daily",
                        metric_family="erosivity",
                        n=len(ratios),
                    )
                )
        except Exception as exc:  # noqa: BLE001 — scoring best-effort
            rows.append(
                _row(
                    experiment=experiment,
                    engine=engine,
                    scenario=scenario,
                    seed=seed,
                    variable="I30",
                    metric="mean_rel_bias",
                    value=float("nan"),
                    scope="holdout_cells",
                    period="daily",
                    metric_family="erosivity",
                    n=0,
                    detail=str(exc)[:120],
                )
            )
    return rows


def score_dual_panel(
    truth: pd.DataFrame,
    imputed: pd.DataFrame,
    missing_mask: pd.DataFrame,
    hole_day: np.ndarray,
    value_cols: list[str],
    *,
    experiment: str,
    engine: str,
    scenario: str,
    seed: int,
    panel: str = "daily",
) -> pd.DataFrame:
    """
    Full dual-scope scorecard for a daily (or similar) panel.

    Returns long-form metrics with columns:
    experiment, engine, scenario, seed, variable, metric, value, scope, period,
    metric_family, n, detail
    """
    rows: list[dict] = []
    rows.extend(
        score_pointwise_holdout(
            truth,
            imputed,
            missing_mask,
            value_cols,
            experiment=experiment,
            engine=engine,
            scenario=scenario,
            seed=seed,
        )
    )
    # Pointwise whole/hole climate means for temps/solar
    ymd = truth["YYYYMMDD"].astype(str).to_numpy()
    whole = np.ones(len(truth), dtype=bool)
    hole = np.asarray(hole_day, dtype=bool)
    if hole.shape[0] != len(truth):
        hole = missing_mask.to_numpy(dtype=bool).any(axis=1)
    for col in value_cols:
        if col not in POINTWISE_VARS or col not in truth.columns:
            continue
        recon = _reconstruct(truth, imputed, missing_mask, col)
        t_arr = truth[col].to_numpy(dtype=float)
        for scope, sel in (("whole", whole), ("hole", hole)):
            rows.extend(
                _period_mean_metrics(
                    experiment=experiment,
                    engine=engine,
                    scenario=scenario,
                    seed=seed,
                    variable=col,
                    ymd=ymd,
                    truth_arr=t_arr,
                    recon_arr=recon,
                    day_sel=sel,
                    scope=scope,
                    metric_family="pointwise_climate",
                )
            )

    rows.extend(
        score_precip_holdout_and_climate(
            truth,
            imputed,
            missing_mask,
            hole,
            experiment=experiment,
            engine=engine,
            scenario=scenario,
            seed=seed,
        )
    )
    rows.extend(
        score_storm_holdout(
            truth,
            imputed,
            missing_mask,
            value_cols,
            experiment=experiment,
            engine=engine,
            scenario=scenario,
            seed=seed,
        )
    )
    rows.extend(
        score_wind_dual(
            truth,
            imputed,
            missing_mask,
            hole,
            experiment=experiment,
            engine=engine,
            scenario=scenario,
            seed=seed,
        )
    )

    # Temp order violations on hole days
    if {"MIN_TEMP", "MAX_TEMP"}.issubset(imputed.columns):
        sub = imputed.loc[hole] if hole.any() else imputed
        bad = (sub["MIN_TEMP"] > sub["MAX_TEMP"]).sum()
        rows.append(
            _row(
                experiment=experiment,
                engine=engine,
                scenario=scenario,
                seed=seed,
                variable="TEMP_ORDER",
                metric="violations",
                value=float(bad),
                scope="hole",
                period="daily",
                metric_family="constraint",
                n=int(hole.sum()) if hole.any() else len(imputed),
            )
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["panel"] = panel
    return df


def aggregate_metric_frame(replicates: pd.DataFrame) -> pd.DataFrame:
    """Mean/std/median by experiment×engine×scenario×variable×metric×scope×period."""
    if replicates.empty:
        return replicates
    keys = [
        c
        for c in (
            "experiment",
            "engine",
            "scenario",
            "variable",
            "metric",
            "scope",
            "period",
            "metric_family",
            "panel",
        )
        if c in replicates.columns
    ]
    rows = []
    for grp_keys, g in replicates.groupby(keys, dropna=False):
        if not isinstance(grp_keys, tuple):
            grp_keys = (grp_keys,)
        vals = pd.to_numeric(g["value"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        rec = dict(zip(keys, grp_keys, strict=False))
        rec.update(
            {
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
        rows.append(rec)
    return pd.DataFrame(rows)
