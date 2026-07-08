"""Cloud transmittance τ for SO_RAD = R^cs · τ."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from Source.phigi.dates import as_yyyymmdd_strings
from Source.phigi.l0_astronomy import clear_sky_series_langleys, day_of_year_from_yyyymmdd

_TAU_AR1_RHO = 0.45

if TYPE_CHECKING:
    from Source.config import WeppcliffConfig
    from Source.phigi.report import ImputationReport

_RCS_FLOOR_LANGLEY = 1.0


def meteorological_season(month: int, lat_deg: float = 45.0) -> int:
    """Return DJF=1, MAM=2, JJA=3, SON=4 with hemisphere flip."""
    m = int(month)
    if lat_deg < 0.0:
        m = ((m + 5) % 12) + 1
    if m in (12, 1, 2):
        return 1
    if m in (3, 4, 5):
        return 2
    if m in (6, 7, 8):
        return 3
    return 4


def resolve_tau_qc_bounds(tauqc: str) -> tuple[float, float]:
    """Parse ``-tauqc`` into (lo, hi) transmittance bounds."""
    if str(tauqc).strip().upper() == "AUTO":
        return 0.05, 1.15
    try:
        hi = float(tauqc)
    except ValueError:
        return 0.05, 1.15
    return 0.05, min(max(hi, 0.2), 1.5)


def compute_tau(
    so_rad: NDArray[np.float64],
    r_cs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """τ = SO_RAD / R^cs where both finite and R^cs > floor."""
    so = np.asarray(so_rad, dtype=np.float64)
    rc = np.asarray(r_cs, dtype=np.float64)
    tau = np.full(so.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(so) & np.isfinite(rc) & (rc > _RCS_FLOOR_LANGLEY)
    tau[valid] = so[valid] / rc[valid]
    return tau


def tau_qc_mask(
    tau: NDArray[np.float64],
    *,
    lo: float,
    hi: float,
) -> NDArray[np.bool_]:
    """True where τ is outside acceptable cloud-transmittance bounds."""
    t = np.asarray(tau, dtype=np.float64)
    return np.isfinite(t) & ((t < lo) | (t > hi))


def _koppen_tau_fallback(
    season: int,
    *,
    zone: str,
    store: "KoppenPriorStore",
) -> float:
    """L1 τ prior mean for a season when local donors are unavailable."""
    prior = store.tau_prior(zone, int(season))
    if prior is None:
        return 0.5
    return float(np.clip(prior.mean, 0.02, 1.15))


def _impute_tau(
    tau: NDArray[np.float64],
    yyyymmdd: NDArray[np.str_],
    cluster: NDArray[np.float64] | NDArray[np.int_],
    *,
    lat_deg: float,
    lon_deg: float,
    kg: str,
    store: "KoppenPriorStore | None" = None,
) -> tuple[NDArray[np.float64], int, str]:
    """Fill missing τ from cluster×season mean, then Köppen L1 prior."""
    from Source.phigi.l1_priors import KoppenPriorStore, load_default_priors

    prior_store = store or load_default_priors()
    zone = prior_store.resolve_zone(kg, lat_deg, lon_deg)
    out = np.array(tau, dtype=np.float64, copy=True)
    dates = as_yyyymmdd_strings(yyyymmdd)
    clusters = np.asarray(cluster)
    n_filled = 0
    used_ar1 = 0

    need = ~np.isfinite(out)
    if not need.any():
        return out, 0, zone

    months = np.array([int(d[4:6]) if len(d) >= 6 else 1 for d in dates], dtype=int)
    seasons = np.array([meteorological_season(m, lat_deg) for m in months], dtype=int)

    for idx in np.where(need)[0]:
        cid = clusters[idx]
        season = seasons[idx]
        base = float("nan")
        loc = np.isfinite(out) & (clusters == cid) & (seasons == season)
        if loc.any():
            base = float(np.nanmean(out[loc]))
        else:
            loc = np.isfinite(out) & (seasons == season)
            if loc.any():
                base = float(np.nanmean(out[loc]))
            else:
                base = _koppen_tau_fallback(season, zone=zone, store=prior_store)

        if idx > 0 and np.isfinite(out[idx - 1]):
            out[idx] = _TAU_AR1_RHO * float(out[idx - 1]) + (1.0 - _TAU_AR1_RHO) * base
            used_ar1 += 1
        else:
            out[idx] = base
        n_filled += 1

    if used_ar1:
        pass  # caller may note AR(1) in report via strategy tag
    return out, n_filled, zone


def impute_daily_solar_radiation(
    dly_var_df: pd.DataFrame,
    yyyymmdd: pd.Series | NDArray[np.str_],
    *,
    lat_deg: float,
    elev_m: float,
    config: "WeppcliffConfig",
    report: "ImputationReport | None" = None,
) -> pd.DataFrame:
    """
    PHIGI L0/L3 solar step: QC τ, impute missing τ, reconstruct SO_RAD.

    Expects ``dly_var_df`` columns ``CLUSTER`` and ``SO_RAD`` (WEPPCLIFF fill_data layout).
    """
    from Source.phigi.report import ImputationReport

    from Source.phigi.l1_priors import load_default_priors

    rep = report or ImputationReport()
    rep.layers_used.append("L0")
    rep.layers_used.append("L1")
    rep.layers_used.append("L3_tau")
    rep.lat = float(lat_deg)
    rep.lon = float(getattr(config, "lon", 0.0))
    rep.elev_m = float(elev_m)
    prior_store = load_default_priors()

    out = dly_var_df.copy()
    if "SO_RAD" not in out.columns:
        rep.notes.append("SO_RAD column missing; skipped PHIGI solar step")
        return out

    dates = as_yyyymmdd_strings(yyyymmdd)
    r_cs = clear_sky_series_langleys(dates, lat_deg=lat_deg, elev_m=elev_m)
    so_rad = pd.to_numeric(out["SO_RAD"], errors="coerce").to_numpy(dtype=np.float64, copy=True)
    cluster = pd.to_numeric(out["CLUSTER"], errors="coerce").to_numpy()

    tau_lo, tau_hi = resolve_tau_qc_bounds(str(config.tauqc))
    rep.tau_qc_lo = tau_lo
    rep.tau_qc_hi = tau_hi

    tau = compute_tau(so_rad, r_cs)
    bad = tau_qc_mask(tau, lo=tau_lo, hi=tau_hi)
    n_invalid = int(bad.sum())
    if n_invalid:
        so_rad[bad] = np.nan
        rep.n_sorad_invalidated = n_invalid
        rep.strategies.append("tau_qc_invalidate")
        rep.notes.append(f"invalidated {n_invalid} SO_RAD values via τ QC")

    tau = compute_tau(so_rad, r_cs)
    tau, n_tau_fill, zone = _impute_tau(
        tau,
        dates,
        cluster,
        lat_deg=lat_deg,
        lon_deg=float(getattr(config, "lon", 0.0)),
        kg=str(config.kg),
        store=prior_store,
    )
    rep.koppen_zone = zone
    rep.n_tau_imputed = n_tau_fill

    missing_so = ~np.isfinite(so_rad)
    n_imputed = int(missing_so.sum())
    if n_imputed:
        so_rad[missing_so] = tau[missing_so] * r_cs[missing_so]
        rep.n_sorad_imputed = n_imputed
        rep.strategies.append("koppen_tau_impute")
        if n_tau_fill > 0 and "tau_ar1_persistence" not in rep.strategies:
            rep.strategies.append("tau_ar1_persistence")
        if n_tau_fill < n_imputed:
            rep.confidence_tier = "medium"

    out["SO_RAD"] = so_rad
    return out