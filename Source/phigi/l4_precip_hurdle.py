"""L4 hurdle precipitation: EVENT occurrence, wet amounts, dry-day enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from Source.phigi.l0_astronomy import clear_sky_series_langleys
from Source.phigi.l3_tau import compute_tau, meteorological_season

if TYPE_CHECKING:
    from Source.config import WeppcliffConfig
    from Source.phigi.report import ImputationReport

_DEFAULT_WET_RATE = 0.15
_MIN_DONORS = 3


@dataclass
class PrecipConstraints:
    """Dry-month and dry-cluster constraints from ``find_missing_precip`` logic."""

    force_dry: NDArray[np.bool_]
    dry_clusters: set[float]
    missing_months: list[int]


def _as_yyyymmdd_strings(values: pd.Series | NDArray | list) -> NDArray[np.str_]:
    """Normalize gap-fill date cells (may be float) to ``YYYYMMDD`` strings."""
    out: list[str] = []
    for v in np.asarray(values, dtype=object).ravel():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append("")
            continue
        if isinstance(v, (int, np.integer)):
            out.append(f"{int(v):08d}")
            continue
        if isinstance(v, float):
            out.append(f"{int(v):08d}")
            continue
        s = str(v).strip()
        if s.endswith(".0"):
            s = s[:-2]
        out.append(s)
    return np.asarray(out, dtype=str)


def _seasons_for_dates(dates: NDArray[np.str_], lat_deg: float) -> NDArray[np.int_]:
    ymd = _as_yyyymmdd_strings(dates)
    months = np.array([int(d[4:6]) if len(d) >= 6 else 1 for d in ymd], dtype=int)
    return np.array([meteorological_season(m, lat_deg) for m in months], dtype=int)


def compute_precip_constraints(
    yyyymmdd: NDArray[np.str_],
    cluster: NDArray[np.float64],
    event: NDArray[np.float64],
) -> PrecipConstraints:
    """
    Replicate ``find_missing_precip`` dry-month / dry-cluster detection.

    Returns a mask of days that must remain dry (EVENT=0).
    """
    dates = np.asarray(yyyymmdd, dtype=str)
    yyyymm = np.array([int(d[:6]) for d in dates], dtype=int)
    yrs_observed = len({d[:4] for d in dates})
    evt = np.asarray(event, dtype=np.float64)

    monthly = pd.DataFrame({"YYYYMM": yyyymm, "EVENT": evt}).groupby("YYYYMM", as_index=False)["EVENT"].sum(min_count=1)
    vals = monthly["EVENT"].fillna(0).to_numpy()
    changes = np.where(vals[:-1] != vals[1:])[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(vals)]])
    lengths = ends - starts
    run_vals = vals[starts]

    streak_max = 0
    zero_streaks = lengths[run_vals == 0]
    if len(zero_streaks) > 6:
        bins = pd.cut(zero_streaks, 3)
        zero_bins = bins.value_counts().sort_index().to_numpy()
        if len(zero_bins) == 3:
            if zero_bins[0] < zero_bins[1] and zero_bins[2] < zero_bins[1]:
                streak_max = int(np.quantile(zero_streaks, 0.75))

    missing_months: list[int] = []
    for zi in np.where(run_vals == 0)[0]:
        if lengths[zi] > streak_max:
            s, e = starts[zi], ends[zi]
            missing_months.extend(monthly["YYYYMM"].iloc[s:e].tolist())

    cluster_tbl = pd.DataFrame({"CLUSTER": cluster, "EVENT": evt}).groupby("CLUSTER", as_index=False)["EVENT"].sum(min_count=1)
    abs_limit = 7
    rel_limit = 0.025
    dry_clusters = set(
        cluster_tbl.loc[
            (cluster_tbl["EVENT"] <= abs_limit)
            | (cluster_tbl["EVENT"] / max(yrs_observed, 1) < rel_limit),
            "CLUSTER",
        ].astype(float).tolist()
    )

    force_dry = np.isin(cluster, list(dry_clusters)) if dry_clusters else np.zeros(len(dates), dtype=bool)

    return PrecipConstraints(
        force_dry=force_dry,
        dry_clusters=dry_clusters,
        missing_months=missing_months,
    )


def _cluster_season_wet_rate(
    event: NDArray[np.float64],
    clusters: NDArray[np.float64],
    seasons: NDArray[np.int_],
    *,
    idx: int,
    cid: float,
    season: int,
) -> float:
    loc = np.isfinite(event) & (clusters == cid) & (seasons == season)
    if int(loc.sum()) < _MIN_DONORS:
        loc = np.isfinite(event) & (seasons == season)
    if not loc.any():
        return _DEFAULT_WET_RATE
    return float(np.mean(event[loc] == 1))


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
    """L4.1/4.2: logistic-style EVENT imputation with dry-cluster constraints."""
    from Source.phigi.report import ImputationReport

    rep = report or ImputationReport()
    if "L4" not in rep.layers_used:
        rep.layers_used.append("L4")

    out = dly_var_df.copy()
    gen = rng or np.random.default_rng(42)
    dates = np.asarray(yyyymmdd.astype(str), dtype=str)
    clusters = pd.to_numeric(out["CLUSTER"], errors="coerce").to_numpy(dtype=np.float64)
    seasons = _seasons_for_dates(dates, lat_deg)
    event = pd.to_numeric(out["EVENT"], errors="coerce").to_numpy(dtype=np.float64, copy=True)

    r_cs = clear_sky_series_langleys(dates.tolist(), lat_deg=lat_deg, elev_m=elev_m)
    so_rad = pd.to_numeric(out.get("SO_RAD"), errors="coerce").to_numpy(dtype=np.float64)
    tau = compute_tau(so_rad, r_cs)

    cons = constraints or compute_precip_constraints(dates, clusters, event)
    need = ~np.isfinite(event)
    n_filled = 0

    for idx in np.where(need)[0]:
        if cons.force_dry[idx]:
            event[idx] = 0.0
            n_filled += 1
            continue

        cid = clusters[idx]
        season = int(seasons[idx])
        base = _cluster_season_wet_rate(event, clusters, seasons, idx=idx, cid=cid, season=season)
        prev_wet = idx > 0 and event[idx - 1] == 1
        p = _wet_probability(base_rate=base, tau=float(tau[idx]), prev_wet=prev_wet)
        event[idx] = 1.0 if p >= 0.5 or gen.random() < p else 0.0
        n_filled += 1

    out["EVENT"] = event
    rep.n_event_imputed += n_filled
    if n_filled and "hurdle_event" not in rep.strategies:
        rep.strategies.append("hurdle_event")
    if cons.dry_clusters:
        rep.notes.append(f"L4 dry-cluster constraint: {len(cons.dry_clusters)} strata")
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
    n_dry = int(dry.sum())
    if n_dry == 0:
        return out

    for col in ("BPS", "ELEN", "PRECIP"):
        if col in out.columns:
            vals = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=np.float64, copy=True)
            vals[dry] = 0.0
            out[col] = vals

    rep.n_dry_days_enforced += n_dry
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

    dates = _as_yyyymmdd_strings(out["YYYYMMDD"])
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
            vals[idx] = fill
            n_filled += 1
        out[col] = vals

    rep.n_wet_event_imputed += n_filled
    if n_filled and "hurdle_wet_amount" not in rep.strategies:
        rep.strategies.append("hurdle_wet_amount")
    return out


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
    dates = np.asarray(yyyymmdd.astype(str), dtype=str)
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