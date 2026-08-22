"""L4 hurdle precipitation: EVENT occurrence, wet amounts, dry-day enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from scipy.optimize import minimize
from scipy.special import expit

from Source.missing_precip import detect_missing_precip_months
from Source.phigi.dates import as_yyyymmdd_strings
from Source.phigi.holdout_metrics import event_brier_score, monthly_precip_mass_bias
from Source.phigi.l0_astronomy import clear_sky_series_langleys
from Source.phigi.l3_tau import compute_tau, meteorological_season

if TYPE_CHECKING:
    from Source.config import WeppcliffConfig
    from Source.phigi.report import ImputationReport

_DEFAULT_WET_RATE = 0.15
_MIN_DONORS = 3


@dataclass
class PrecipConstraints:
    """Dry-month constraints from calendar-month gap detection."""

    force_dry: NDArray[np.bool_]
    dry_clusters: set[float]
    missing_months: list[int]
    dry_calendar_months: set[int] | None = None
    expected_dry_run: int = 0


def _seasons_for_dates(dates: NDArray[np.str_], lat_deg: float) -> NDArray[np.int_]:
    ymd = as_yyyymmdd_strings(dates)
    months = np.array([int(d[4:6]) if len(d) >= 6 else 1 for d in ymd], dtype=int)
    return np.array([meteorological_season(m, lat_deg) for m in months], dtype=int)


def compute_precip_constraints(
    yyyymmdd: NDArray[np.str_],
    cluster: NDArray[np.float64],
    event: NDArray[np.float64],
) -> PrecipConstraints:
    """Calendar-month gap vs dry detection; CLUSTER is not used as a cutoff."""
    del cluster
    dates = as_yyyymmdd_strings(yyyymmdd)
    evt = np.asarray(event, dtype=np.float64)
    scan = detect_missing_precip_months(dates, evt)
    yyyymm = np.zeros(len(dates), dtype=np.int64)
    for i, d in enumerate(dates):
        if isinstance(d, str) and len(d) >= 6 and d[:6].isdigit():
            yyyymm[i] = int(d[:6])
    force_dry = np.isin(yyyymm, scan.force_dry_months) if scan.force_dry_months else np.zeros(
        len(dates), dtype=bool
    )
    return PrecipConstraints(
        force_dry=force_dry,
        dry_clusters=set(),
        missing_months=list(scan.missing_months),
        dry_calendar_months=set(scan.dry_calendar_months),
        expected_dry_run=int(scan.expected_dry_run),
    )


def _month_wet_rate(
    event: NDArray[np.float64],
    months: NDArray[np.int_],
    *,
    month: int,
) -> float:
    loc = np.isfinite(event) & (months == month)
    if int(loc.sum()) < _MIN_DONORS:
        loc = np.isfinite(event)
    if not loc.any():
        return _DEFAULT_WET_RATE
    return float(np.mean(event[loc] == 1))


def _months_from_dates(dates: NDArray[np.str_]) -> NDArray[np.int_]:
    out = np.ones(len(dates), dtype=int)
    for i, d in enumerate(dates):
        if isinstance(d, str) and len(d) >= 6 and d[4:6].isdigit():
            moy = int(d[4:6])
            if 1 <= moy <= 12:
                out[i] = moy
    return out


def _wet_probability(
    *,
    base_rate: float,
    tau: float,
    prev_wet: bool,
) -> float:
    """P(wet): empirical rate scaled by cloudiness (low τ) and persistence."""
    tau_adj = float(np.clip(1.15 - tau, 0.55, 1.45)) if np.isfinite(tau) else 1.0
    p = base_rate * tau_adj
    if prev_wet:
        p *= 1.2
    return float(np.clip(p, 0.02, 0.95))


def _fit_logistic_wet_coeffs(
    event: NDArray[np.float64],
    tau: NDArray[np.float64],
    months: NDArray[np.int_],
    prev_wet: NDArray[np.bool_],
) -> NDArray[np.float64] | None:
    """Fit intercept + τ + persistence + month intercepts for EVENT occurrence."""
    y = np.asarray(event, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(tau)
    if int(m.sum()) < 30:
        return None

    month_dummies = np.column_stack([(months == mo).astype(float) for mo in range(2, 13)])
    x = np.column_stack(
        [
            np.ones(int(m.sum())),
            tau[m],
            prev_wet[m].astype(float),
            month_dummies[m],
        ]
    )
    target = y[m]

    def nll(beta: NDArray[np.float64]) -> float:
        p = expit(x @ beta)
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return float(-np.sum(target * np.log(p) + (1.0 - target) * np.log(1.0 - p)))

    beta0 = np.zeros(x.shape[1], dtype=np.float64)
    result = minimize(nll, beta0, method="L-BFGS-B")
    if not result.success:
        return None
    return np.asarray(result.x, dtype=np.float64)


def _logistic_wet_probability(
    *,
    coeffs: NDArray[np.float64],
    tau: float,
    month: int,
    prev_wet: bool,
) -> float:
    month_vec = np.array([float(month == mo) for mo in range(2, 13)], dtype=np.float64)
    x = np.array(
        [1.0, float(tau) if np.isfinite(tau) else 0.5, float(prev_wet), *month_vec]
    )
    return float(np.clip(expit(coeffs @ x), 0.02, 0.95))


def _lognormal_mean(values: NDArray[np.float64]) -> float:
    """Mean of a lognormal fit to positive donor amounts."""
    pos = values[np.isfinite(values) & (values > 0)]
    if len(pos) < 3:
        return float("nan")
    logv = np.log(pos)
    mu = float(np.mean(logv))
    sigma = float(max(np.std(logv), 1e-3))
    return float(np.exp(mu + 0.5 * sigma * sigma))


def _gamma_mean(values: NDArray[np.float64]) -> float:
    """Method-of-moments gamma mean for positive donor amounts."""
    pos = values[np.isfinite(values) & (values > 0)]
    if len(pos) < 3:
        return float("nan")
    mean = float(np.mean(pos))
    var = float(np.var(pos))
    if var <= 1e-9:
        return mean
    shape = (mean * mean) / var
    scale = var / mean
    return float(shape * scale)


def impute_event_occurrence(
    dly_var_df: pd.DataFrame,
    yyyymmdd: pd.Series | NDArray[np.str_],
    *,
    lat_deg: float,
    elev_m: float,
    constraints: PrecipConstraints | None = None,
    report: ImputationReport | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """L4.1/4.2: logistic EVENT imputation with month intercepts; dry months stay 0."""
    from Source.phigi.report import ImputationReport

    rep = report or ImputationReport()
    if "L4" not in rep.layers_used:
        rep.layers_used.append("L4")

    out = dly_var_df.copy()
    gen = rng or np.random.default_rng(42)
    dates = as_yyyymmdd_strings(yyyymmdd)
    clusters = pd.to_numeric(out["CLUSTER"], errors="coerce").to_numpy(dtype=np.float64)
    months = _months_from_dates(dates)
    event = pd.to_numeric(out["EVENT"], errors="coerce").to_numpy(dtype=np.float64, copy=True)

    r_cs = clear_sky_series_langleys(dates.tolist(), lat_deg=lat_deg, elev_m=elev_m)
    so_rad = pd.to_numeric(out.get("SO_RAD"), errors="coerce").to_numpy(dtype=np.float64)
    tau = compute_tau(so_rad, r_cs)
    prev_wet_arr = np.zeros(len(event), dtype=bool)
    prev_wet_arr[1:] = event[:-1] == 1

    cons = constraints or compute_precip_constraints(dates, clusters, event)
    need = ~np.isfinite(event)
    n_filled = 0
    logistic_coeffs = _fit_logistic_wet_coeffs(event, tau, months, prev_wet_arr)
    if logistic_coeffs is not None and "logistic_event" not in rep.strategies:
        rep.strategies.append("logistic_event")

    for idx in np.where(need)[0]:
        if idx < len(cons.force_dry) and cons.force_dry[idx]:
            event[idx] = 0.0
            n_filled += 1
            continue

        month = int(months[idx])
        prev_wet = bool(prev_wet_arr[idx])
        if logistic_coeffs is not None:
            p = _logistic_wet_probability(
                coeffs=logistic_coeffs,
                tau=float(tau[idx]),
                month=month,
                prev_wet=prev_wet,
            )
        else:
            base = _month_wet_rate(event, months, month=month)
            p = _wet_probability(base_rate=base, tau=float(tau[idx]), prev_wet=prev_wet)
        event[idx] = 1.0 if p >= 0.5 or gen.random() < p else 0.0
        n_filled += 1

    out["EVENT"] = event
    rep.n_event_imputed += n_filled
    if n_filled and "hurdle_event" not in rep.strategies:
        rep.strategies.append("hurdle_event")
    if cons.missing_months:
        rep.notes.append(f"L4 month-of-year gap months: {len(cons.missing_months)}")
    if cons.dry_calendar_months:
        rep.notes.append(
            f"L4 climatological dry months: {sorted(cons.dry_calendar_months)}"
        )
    return out


def enforce_dry_days(
    dly_var_df: pd.DataFrame,
    dly_pcp_df: pd.DataFrame,
    *,
    report: ImputationReport | None = None,
) -> pd.DataFrame:
    """L4.5: zero BPS/ELEN/PRECIP on dry days (EVENT=0)."""
    from Source.phigi.report import ImputationReport

    rep = report or ImputationReport()
    out = dly_pcp_df.copy()
    event = pd.to_numeric(dly_var_df["EVENT"], errors="coerce").to_numpy(dtype=np.float64)
    dry = event == 0
    if not dry.any():
        return out

    n_enforced = 0
    for col in ("BPS", "ELEN", "PRECIP"):
        if col not in out.columns:
            continue
        orig = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=np.float64, copy=True)
        observed_positive = orig > 0
        to_zero = dry & ~observed_positive
        n_enforced = max(n_enforced, int(to_zero.sum()))
        orig[to_zero] = 0.0
        out[col] = orig

    rep.n_dry_days_enforced += n_enforced
    if "dry_day_enforcement" not in rep.strategies:
        rep.strategies.append("dry_day_enforcement")
    return out


def _cluster_donor_stat(
    values: NDArray[np.float64],
    clusters: NDArray[np.float64],
    seasons: NDArray[np.int_],
    *,
    cid: float,
    season: int,
    positive_only: bool = False,
    use_median: bool = False,
) -> float:
    loc = np.isfinite(values) & (clusters == cid) & (seasons == season)
    if positive_only:
        loc &= values > 0
    if int(loc.sum()) < _MIN_DONORS:
        loc = np.isfinite(values) & (seasons == season)
        if positive_only:
            loc &= values > 0
    if not loc.any():
        return float("nan")
    subset = values[loc]
    return float(np.median(subset) if use_median else np.mean(subset))


def impute_wet_event_characteristics(
    dly_eve_df: pd.DataFrame,
    *,
    lat_deg: float,
    report: ImputationReport | None = None,
) -> pd.DataFrame:
    """
    L4.3/4.4: fill BPS, ELEN, PRECIP on wet-event rows from cluster donors.

    Uses log-space mean for PRECIP; median for BPS/ELEN (storm generator deferred to L5).
    """
    from Source.phigi.report import ImputationReport

    rep = report or ImputationReport()
    if "L4" not in rep.layers_used:
        rep.layers_used.append("L4")

    out = dly_eve_df.copy()
    if "YYYYMMDD" not in out.columns or "CLUSTER" not in out.columns:
        return out

    dates = as_yyyymmdd_strings(out["YYYYMMDD"])
    seasons = _seasons_for_dates(dates, lat_deg)
    clusters = pd.to_numeric(out["CLUSTER"], errors="coerce").to_numpy(dtype=np.float64)
    n_filled = 0

    for col, positive, median in (
        ("PRECIP", True, False),
        ("BPS", False, True),
        ("ELEN", False, True),
    ):
        if col not in out.columns:
            continue
        vals = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=np.float64, copy=True)
        need = ~np.isfinite(vals)
        if col == "BPS":
            need |= vals <= 0
        for idx in np.where(need)[0]:
            if col == "PRECIP":
                loc = (
                    np.isfinite(vals)
                    & (clusters == clusters[idx])
                    & (seasons == int(seasons[idx]))
                    & (vals > 0)
                )
                if int(loc.sum()) < _MIN_DONORS:
                    loc = np.isfinite(vals) & (seasons == int(seasons[idx])) & (vals > 0)
                fill = _lognormal_mean(vals[loc]) if loc.any() else float("nan")
                if not np.isfinite(fill):
                    fill = _gamma_mean(vals[loc]) if loc.any() else float("nan")
            else:
                fill = _cluster_donor_stat(
                    vals,
                    clusters,
                    seasons,
                    cid=clusters[idx],
                    season=int(seasons[idx]),
                    positive_only=positive,
                    use_median=median,
                )
            if col == "PRECIP" and np.isfinite(fill):
                fill = max(fill, 0.01)
            if not np.isfinite(fill):
                fill = 1.0 if col in ("BPS", "ELEN") else 0.1
                rep.mark_low_confidence(f"L4 default fallback for {col}")
            vals[idx] = fill
            n_filled += 1
        out[col] = vals

    rep.n_wet_event_imputed += n_filled
    if n_filled and "hurdle_wet_amount" not in rep.strategies:
        rep.strategies.append("hurdle_wet_amount")
    if n_filled and "lognormal_precip" not in rep.strategies:
        rep.strategies.append("lognormal_precip")
    return out


def score_precip_holdout(
    *,
    dates: NDArray[np.str_],
    actual_event: NDArray[np.float64],
    predicted_event: NDArray[np.float64],
    predicted_prob: NDArray[np.float64],
    actual_precip: NDArray[np.float64],
    predicted_precip: NDArray[np.float64],
    holdout_mask: NDArray[np.bool_],
) -> dict[str, float]:
    """L4 verification gate metrics on a masked hold-out set."""
    mask = np.asarray(holdout_mask, dtype=bool)
    return {
        "event_brier": event_brier_score(actual_event, predicted_prob, mask=mask),
        "monthly_precip_mass_bias": monthly_precip_mass_bias(
            dates, actual_precip, predicted_precip, mask=mask
        ),
        "event_accuracy": float(
            np.mean(
                (np.asarray(predicted_event)[mask] == np.asarray(actual_event)[mask])
            )
        )
        if mask.any()
        else float("nan"),
    }


def impute_precip_hurdle(
    dly_var_df: pd.DataFrame,
    dly_pcp_df: pd.DataFrame,
    yyyymmdd: pd.Series | NDArray[np.str_],
    *,
    lat_deg: float,
    elev_m: float,
    config: WeppcliffConfig,
    constraints: PrecipConstraints | None = None,
    report: ImputationReport | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    L4 orchestrator: EVENT hurdle → dry-day enforcement on daily precip columns.

    Wet-event BPS/ELEN/PRECIP are filled later via :func:`impute_wet_event_characteristics`.
    """
    from Source.phigi.report import ImputationReport

    rep = report or ImputationReport(gf=str(config.gf))
    dates = as_yyyymmdd_strings(yyyymmdd)
    clusters = pd.to_numeric(dly_var_df["CLUSTER"], errors="coerce").to_numpy(dtype=np.float64)
    event = pd.to_numeric(dly_var_df["EVENT"], errors="coerce").to_numpy(dtype=np.float64)
    cons = constraints or compute_precip_constraints(dates, clusters, event)

    var_out = impute_event_occurrence(
        dly_var_df,
        yyyymmdd,
        lat_deg=lat_deg,
        elev_m=elev_m,
        constraints=cons,
        report=rep,
        rng=np.random.default_rng(int(getattr(config, "seed", 42)) + 2),
    )
    pcp_out = enforce_dry_days(var_out, dly_pcp_df, report=rep)
    rep.notes.append("L4 precip hurdle: EVENT → dry-day enforcement")
    return var_out, pcp_out