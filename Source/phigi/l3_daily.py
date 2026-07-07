"""L3 directed daily posterior: τ-conditioned temperature, then moisture and wind."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from Source.phigi.l0_astronomy import clear_sky_series_langleys
from Source.phigi.l2_weather_types import FEATURES, WeatherTypeModel
from Source.phigi.l3_tau import compute_tau, meteorological_season

if TYPE_CHECKING:
    from Source.config import WeppcliffConfig
    from Source.phigi.l1_priors import KoppenPriorStore
    from Source.phigi.report import ImputationReport

_TEMP_VARS = ("MIN_TEMP", "MAX_TEMP")
_MOISTURE_WIND_VARS = ("DP_TEMP", "W_VEL", "W_DIR")
_FEATURE_INDEX = {name: i for i, name in enumerate(FEATURES)}


def _seasons_for_dates(dates: NDArray[np.str_], lat_deg: float) -> NDArray[np.int_]:
    months = np.array([int(d[4:6]) for d in dates], dtype=int)
    return np.array([meteorological_season(m, lat_deg) for m in months], dtype=int)


def _cluster_component(cluster: float) -> tuple[int, int]:
    """Decode ``season×10 + component`` CLUSTER encoding."""
    cid = int(cluster)
    season = cid // 10
    component = cid % 10
    if season < 1 or season > 4:
        month_season = ((cid % 40) // 10) or 1
        return month_season, max(cid % 10, 0)
    return season, component


def _tau_weighted_mean(
    values: NDArray[np.float64],
    tau_donors: NDArray[np.float64],
    tau_target: float,
) -> float:
    valid = np.isfinite(values) & np.isfinite(tau_donors)
    if not valid.any():
        return float("nan")
    weights = 1.0 / (1.0 + np.abs(tau_donors[valid] - tau_target))
    return float(np.average(values[valid], weights=weights))


def _circular_mean_degrees(angles: NDArray[np.float64]) -> float:
    valid = np.isfinite(angles)
    if not valid.any():
        return float("nan")
    rad = np.deg2rad(angles[valid])
    mean_rad = np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))
    deg = float(np.rad2deg(mean_rad))
    return deg % 360.0 if deg >= 0 else deg + 360.0


def _mixture_component_mean(
    model: WeatherTypeModel | None,
    season: int,
    component: int,
    variable: str,
) -> float:
    if model is None:
        return float("nan")
    mixture = model.seasons.get(int(season))
    if mixture is None or component < 0 or component >= mixture.n_components:
        return float("nan")
    idx = _FEATURE_INDEX.get(variable)
    if idx is None:
        return float("nan")
    return float(mixture.means[component, idx])


def _l1_scalar_prior(
    store: KoppenPriorStore,
    zone: str,
    season: int,
    variable: str,
    *,
    rng: np.random.Generator,
) -> float:
    if variable in _TEMP_VARS:
        draw = store.sample_multivariate(
            zone,
            season,
            variables=["MIN_TEMP", "MAX_TEMP"],
            rng=rng,
        )
        return float(draw.get(variable, np.nan))
    prior = store.variable_prior(zone, season, variable)
    if prior is None:
        return float("nan")
    return float(rng.normal(prior.mean, prior.std * 0.25))


def _impute_scalar_directed(
    values: NDArray[np.float64],
    *,
    need: NDArray[np.bool_],
    clusters: NDArray[np.float64],
    seasons: NDArray[np.int_],
    tau: NDArray[np.float64],
    variable: str,
    model: WeatherTypeModel | None,
    store: KoppenPriorStore,
    zone: str,
    rng: np.random.Generator,
    circular: bool = False,
) -> tuple[NDArray[np.float64], int]:
    """Fill missing scalars: cluster×season (τ-weighted) → season → L2 → L1."""
    out = np.array(values, dtype=np.float64, copy=True)
    n_filled = 0

    for idx in np.where(need)[0]:
        cid = clusters[idx]
        season = int(seasons[idx])
        tau_target = float(tau[idx]) if np.isfinite(tau[idx]) else 0.5
        _, component = _cluster_component(cid)

        loc = np.isfinite(out) & (clusters == cid) & (seasons == season)
        if loc.any():
            if circular:
                fill = _circular_mean_degrees(out[loc])
            else:
                fill = _tau_weighted_mean(out[loc], tau[loc], tau_target)
            if np.isfinite(fill):
                out[idx] = fill
                n_filled += 1
                continue

        loc = np.isfinite(out) & (seasons == season)
        if loc.any():
            if circular:
                fill = _circular_mean_degrees(out[loc])
            else:
                fill = _tau_weighted_mean(out[loc], tau[loc], tau_target)
            if np.isfinite(fill):
                out[idx] = fill
                n_filled += 1
                continue

        fill = _mixture_component_mean(model, season, component, variable)
        if np.isfinite(fill):
            out[idx] = fill
            n_filled += 1
            continue

        fill = _l1_scalar_prior(store, zone, season, variable, rng=rng)
        if np.isfinite(fill):
            out[idx] = fill
            n_filled += 1
            continue

        if circular:
            out[idx] = 0.0
        else:
            out[idx] = float(np.nanmean(out[np.isfinite(out)])) if np.isfinite(out).any() else 0.0
        n_filled += 1

    return out, n_filled


def _enforce_temperature_order(
    min_temp: NDArray[np.float64],
    max_temp: NDArray[np.float64],
    *,
    max_passes: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Ensure MIN_TEMP ≤ MAX_TEMP; return adjusted arrays and violation count."""
    tmin = np.array(min_temp, dtype=np.float64, copy=True)
    tmax = np.array(max_temp, dtype=np.float64, copy=True)
    violations = int(np.sum(tmin > tmax))
    for _ in range(max_passes):
        bad = tmin > tmax
        if not bad.any():
            break
        tmin[bad] = tmax[bad]
    return tmin, tmax, violations


def impute_daily_temperature(
    dly_var_df: pd.DataFrame,
    yyyymmdd: pd.Series | NDArray[np.str_],
    *,
    lat_deg: float,
    elev_m: float,
    config: WeppcliffConfig,
    weather_model: WeatherTypeModel | None = None,
    report: ImputationReport | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """L3b: τ-conditioned MIN/MAX temperature imputation."""
    from Source.phigi.l1_priors import load_default_priors
    from Source.phigi.report import ImputationReport

    rep = report or ImputationReport()
    if "L3_daily" not in rep.layers_used:
        rep.layers_used.append("L3_daily")

    out = dly_var_df.copy()
    gen = rng or np.random.default_rng(int(getattr(config, "seed", 42)))
    store = load_default_priors()
    zone = store.resolve_zone(
        str(config.kg),
        lat_deg,
        float(getattr(config, "lon", 0.0)),
    )

    dates = np.asarray(yyyymmdd.astype(str), dtype=str)
    seasons = _seasons_for_dates(dates, lat_deg)
    clusters = pd.to_numeric(out["CLUSTER"], errors="coerce").to_numpy(dtype=np.float64)
    r_cs = clear_sky_series_langleys(dates.tolist(), lat_deg=lat_deg, elev_m=elev_m)
    so_rad = pd.to_numeric(out.get("SO_RAD"), errors="coerce").to_numpy(dtype=np.float64)
    tau = compute_tau(so_rad, r_cs)

    min_temp = pd.to_numeric(out.get("MIN_TEMP"), errors="coerce").to_numpy(dtype=np.float64, copy=True)
    max_temp = pd.to_numeric(out.get("MAX_TEMP"), errors="coerce").to_numpy(dtype=np.float64, copy=True)

    n_filled = 0
    for var_name, arr in (("MIN_TEMP", min_temp), ("MAX_TEMP", max_temp)):
        need = ~np.isfinite(arr)
        if not need.any():
            continue
        filled, n = _impute_scalar_directed(
            arr,
            need=need,
            clusters=clusters,
            seasons=seasons,
            tau=tau,
            variable=var_name,
            model=weather_model,
            store=store,
            zone=zone,
            rng=gen,
        )
        n_filled += n
        if var_name == "MIN_TEMP":
            min_temp = filled
        else:
            max_temp = filled

    min_temp, max_temp, n_violations = _enforce_temperature_order(min_temp, max_temp)
    if n_violations:
        rep.notes.append(f"L3 corrected {n_violations} Tmin>Tmax violations")

    out["MIN_TEMP"] = min_temp
    out["MAX_TEMP"] = max_temp
    rep.n_temp_imputed += n_filled
    if n_filled and "directed_temperature" not in rep.strategies:
        rep.strategies.append("directed_temperature")
    return out


def impute_daily_moisture_wind(
    dly_var_df: pd.DataFrame,
    yyyymmdd: pd.Series | NDArray[np.str_],
    *,
    lat_deg: float,
    elev_m: float,
    config: WeppcliffConfig,
    weather_model: WeatherTypeModel | None = None,
    report: ImputationReport | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """L3c/L3d: dew point and wind after temperature is filled."""
    from Source.phigi.l1_priors import load_default_priors
    from Source.phigi.report import ImputationReport

    rep = report or ImputationReport()
    if "L3_daily" not in rep.layers_used:
        rep.layers_used.append("L3_daily")

    out = dly_var_df.copy()
    gen = rng or np.random.default_rng(int(getattr(config, "seed", 42)) + 1)
    store = load_default_priors()
    zone = store.resolve_zone(
        str(config.kg),
        lat_deg,
        float(getattr(config, "lon", 0.0)),
    )

    dates = np.asarray(yyyymmdd.astype(str), dtype=str)
    seasons = _seasons_for_dates(dates, lat_deg)
    clusters = pd.to_numeric(out["CLUSTER"], errors="coerce").to_numpy(dtype=np.float64)
    r_cs = clear_sky_series_langleys(dates.tolist(), lat_deg=lat_deg, elev_m=elev_m)
    so_rad = pd.to_numeric(out.get("SO_RAD"), errors="coerce").to_numpy(dtype=np.float64)
    tau = compute_tau(so_rad, r_cs)

    n_filled = 0
    for var_name in _MOISTURE_WIND_VARS:
        if var_name not in out.columns:
            continue
        arr = pd.to_numeric(out[var_name], errors="coerce").to_numpy(dtype=np.float64, copy=True)
        need = ~np.isfinite(arr)
        if not need.any():
            continue
        filled, n = _impute_scalar_directed(
            arr,
            need=need,
            clusters=clusters,
            seasons=seasons,
            tau=tau,
            variable=var_name,
            model=weather_model,
            store=store,
            zone=zone,
            rng=gen,
            circular=(var_name == "W_DIR"),
        )
        out[var_name] = filled
        n_filled += n

    rep.n_moisture_wind_imputed += n_filled
    if n_filled and "directed_moisture_wind" not in rep.strategies:
        rep.strategies.append("directed_moisture_wind")
    return out


def impute_daily_posterior(
    dly_var_df: pd.DataFrame,
    yyyymmdd: pd.Series | NDArray[np.str_],
    *,
    lat_deg: float,
    elev_m: float,
    config: WeppcliffConfig,
    weather_model: WeatherTypeModel | None = None,
    report: ImputationReport | None = None,
) -> pd.DataFrame:
    """
    L3 orchestrator: directed temperature (3b), then moisture/wind (3c–3d).

    Expects SO_RAD already imputed (L0/L3 τ stage). EVENT/PRECIP handled by L4.
    """
    from Source.phigi.report import ImputationReport

    rep = report or ImputationReport(gf=str(config.gf))
    model = weather_model

    out = impute_daily_temperature(
        dly_var_df,
        yyyymmdd,
        lat_deg=lat_deg,
        elev_m=elev_m,
        config=config,
        weather_model=model,
        report=rep,
    )
    out = impute_daily_moisture_wind(
        out,
        yyyymmdd,
        lat_deg=lat_deg,
        elev_m=elev_m,
        config=config,
        weather_model=model,
        report=rep,
    )
    rep.notes.append("L3 daily posterior: directed temperature → moisture/wind")
    return out