"""PHIGI pipeline hooks into WEPPCLIFF ``fill_data``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from Source.phigi.l2_weather_types import (
    WeatherTypeModel,
    assign_weather_types,
    fit_local_weather_types,
    weather_type_cluster_labels,
)
from Source.phigi.l3_tau import impute_daily_solar_radiation
from Source.phigi.report import ImputationReport

if TYPE_CHECKING:
    from Source.config import WeppcliffConfig
    from Source.data_model import WeppcliffData


def _station_elev_m(data: "WeppcliffData") -> float:
    meta = data.metadata
    raw = meta.get("ELEV", 0)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def phigi_impute_solar_stage(
    dly_var_df: pd.DataFrame,
    yyyymmdd: pd.Series,
    *,
    config: "WeppcliffConfig",
    data: "WeppcliffData",
    report: ImputationReport | None = None,
) -> pd.DataFrame:
    """Run PHIGI τ-based solar imputation for the WEPPCLIFF solar stage."""
    rep = (
        report
        or getattr(config, "phigi_report", None)
        or ImputationReport(gf=str(config.gf))
    )
    result = impute_daily_solar_radiation(
        dly_var_df,
        yyyymmdd,
        lat_deg=float(config.lat),
        elev_m=_station_elev_m(data),
        config=config,
        report=rep,
    )
    config.phigi_report = rep
    return result


def phigi_fit_weather_types(
    dly_var_df: pd.DataFrame,
    yyyymmdd: pd.Series,
    *,
    config: "WeppcliffConfig",
    data: "WeppcliffData",
    report: ImputationReport | None = None,
) -> WeatherTypeModel:
    """
    L2: fit local weather-type mixtures on observed daily rows (no Köppen required).

    Stores the model on ``config.phigi_weather_types``; call
    :func:`phigi_wire_weather_type_clusters` to apply labels as CLUSTER for MICE.
    """
    rep = report or getattr(config, "phigi_report", None) or ImputationReport(gf=str(config.gf))
    if "L2" not in rep.layers_used:
        rep.layers_used.append("L2")
    if "local_weather_mixture" not in rep.strategies:
        rep.strategies.append("local_weather_mixture")

    panel = dly_var_df.copy()
    need_cols = ["SO_RAD", "MIN_TEMP", "MAX_TEMP", "W_VEL", "DP_TEMP"]
    for col in need_cols:
        if col not in panel.columns:
            rep.notes.append(f"L2 skipped: missing column {col}")
            config.phigi_report = rep
            config.phigi_weather_types = WeatherTypeModel()
            return config.phigi_weather_types

    model = fit_local_weather_types(
        panel,
        yyyymmdd,
        lat_deg=float(config.lat),
        elev_m=_station_elev_m(data),
        rng=None,
    )
    components = assign_weather_types(
        panel,
        yyyymmdd,
        model,
        lat_deg=float(config.lat),
        elev_m=_station_elev_m(data),
    )
    rep.weather_type_seasons_fitted = model.n_seasons_fitted
    rep.weather_type_components = {
        str(s): m.n_components for s, m in sorted(model.seasons.items())
    }
    rep.notes.append(
        f"L2 fitted {model.n_seasons_fitted} seasonal mixtures; "
        f"assigned {int((components >= 0).sum())} day labels"
    )
    config.phigi_weather_types = model
    config.phigi_weather_type_labels = components
    config.phigi_report = rep
    return model


def phigi_cluster_lut(
    yyyymmdd: pd.Series,
    *,
    config: "WeppcliffConfig",
) -> pd.DataFrame:
    """Build ``YYYYMMDD`` → L2 CLUSTER lookup (season×10 + component)."""
    components = getattr(config, "phigi_weather_type_labels", None)
    if components is None:
        return pd.DataFrame(columns=["YYYYMMDD", "CLUSTER"])
    dates = yyyymmdd.astype(str)
    if len(components) != len(dates):
        return pd.DataFrame(columns=["YYYYMMDD", "CLUSTER"])
    clusters = weather_type_cluster_labels(
        dates,
        components,
        lat_deg=float(config.lat),
    )
    return pd.DataFrame({"YYYYMMDD": dates.values, "CLUSTER": clusters})


def phigi_wire_weather_type_clusters(
    dly_var_df: pd.DataFrame,
    yyyymmdd: pd.Series,
    *,
    config: "WeppcliffConfig",
    data: "WeppcliffData | None" = None,
    dly_ts_out: int | None = None,
    pcp_ts_out: int | None = None,
    context: str = "mice",
) -> pd.DataFrame:
    """
    Replace calendar CLUSTER with L2 weather-type labels.

    ``context='solar'`` wires donors for the τ step; ``'mice'`` wires downstream MICE.
    Updates ``dly_var_df``, optional daily/precip slots, and ``config.phigi_cluster_lut``.
    """
    lut = phigi_cluster_lut(yyyymmdd, config=config)
    if lut.empty:
        return dly_var_df

    config.phigi_cluster_lut = lut
    config.impute_granularity = "weather_type"

    rep = getattr(config, "phigi_report", None)
    if rep is not None:
        n_unique = int(lut["CLUSTER"].nunique())
        if context == "solar":
            if "weather_type_tau_donors" not in rep.strategies:
                rep.strategies.append("weather_type_tau_donors")
            rep.notes.append(
                f"L2 CLUSTER wired for solar τ donors ({n_unique} weather-type strata)"
            )
        else:
            if "weather_type_cluster" not in rep.strategies:
                rep.strategies.append("weather_type_cluster")
            suffix = " (refreshed after solar)" if context == "mice_refresh" else ""
            rep.notes.append(
                f"L2 CLUSTER wired for MICE ({n_unique} weather-type strata){suffix}"
            )
        config.phigi_report = rep

    out = dly_var_df.copy()
    out["CLUSTER"] = lut["CLUSTER"].to_numpy()

    if data is not None and dly_ts_out is not None:
        slot = data.slot_df(dly_ts_out).copy()
        if "YYYYMMDD" in slot.columns:
            slot["YYYYMMDD"] = slot["YYYYMMDD"].astype(str)
            if "CLUSTER" in slot.columns:
                slot = slot.drop(columns=["CLUSTER"])
            slot = slot.merge(lut, on="YYYYMMDD", how="left")
            data[dly_ts_out] = slot

    if data is not None and pcp_ts_out is not None:
        pcp = data.slot_df(pcp_ts_out).copy()
        ymd = pd.to_datetime(pcp["DT_1"], errors="coerce").dt.strftime("%Y%m%d")
        pcp["YYYYMMDD"] = ymd
        if "CLUSTER" in pcp.columns:
            pcp = pcp.drop(columns=["CLUSTER"])
        pcp = pcp.merge(lut, on="YYYYMMDD", how="left")
        data[pcp_ts_out] = pcp

    return out


def phigi_refresh_weather_type_labels(
    dly_var_df: pd.DataFrame,
    yyyymmdd: pd.Series,
    *,
    config: "WeppcliffConfig",
    data: "WeppcliffData | None" = None,
    dly_ts_out: int | None = None,
    pcp_ts_out: int | None = None,
) -> pd.DataFrame:
    """Re-assign weather-type components after SO_RAD/τ imputation; re-wire CLUSTER."""
    model = getattr(config, "phigi_weather_types", None)
    if model is None or not model.seasons:
        return dly_var_df

    panel = dly_var_df.copy()
    need_cols = ["SO_RAD", "MIN_TEMP", "MAX_TEMP", "W_VEL", "DP_TEMP"]
    if not all(col in panel.columns for col in need_cols):
        return dly_var_df

    components = assign_weather_types(
        panel,
        yyyymmdd,
        model,
        lat_deg=float(config.lat),
        elev_m=_station_elev_m(data) if data is not None else 0.0,
    )
    config.phigi_weather_type_labels = components
    return phigi_wire_weather_type_clusters(
        dly_var_df,
        yyyymmdd,
        config=config,
        data=data,
        dly_ts_out=dly_ts_out,
        pcp_ts_out=pcp_ts_out,
        context="mice_refresh",
    )


def merge_phigi_clusters(
    df: pd.DataFrame,
    *,
    config: "WeppcliffConfig",
    date_col: str = "YYYYMMDD",
) -> pd.DataFrame:
    """Attach PHIGI CLUSTER from ``config.phigi_cluster_lut`` when available."""
    lut = getattr(config, "phigi_cluster_lut", None)
    if lut is None or lut.empty or date_col not in df.columns:
        return df
    out = df.copy()
    out[date_col] = out[date_col].astype(str)
    if "CLUSTER" in out.columns:
        out = out.drop(columns=["CLUSTER"])
    return out.merge(lut, on=date_col, how="left")


def write_imputation_report(config: "WeppcliffConfig") -> None:
    """Write JSON imputation report when ``-ir t``."""
    import json
    from pathlib import Path

    from Source.paths import create_directory

    rep = getattr(config, "phigi_report", None)
    if rep is None:
        return
    export_dir = Path(config.ex_dir or config.e or ".")
    create_directory(export_dir)
    out_path = export_dir / "Imputation-Report.json"
    out_path.write_text(json.dumps(rep.to_dict(), indent=2), encoding="utf-8")