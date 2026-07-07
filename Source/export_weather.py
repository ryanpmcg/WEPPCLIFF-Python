"""Weather timeseries export with kinetic energy and erosion index summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from Source.slots import PipelineSlots

if TYPE_CHECKING:
    from Source.config import WeppcliffConfig
    from Source.data_model import WeppcliffData

ExportKind = Literal[
    "daily_ts",
    "monthly_ts",
    "annual_ts",
    "monthly_mean",
    "annual_mean",
    "biweekly_mean",
]

_INCHES_TO_MM = 25.4
OMITTED_STORMS_COL = "Omitted-Storms"


def _erosion_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.endswith("_KE") or c.endswith("_EI")]


def _storm_depth_threshold(config: "WeppcliffConfig") -> float:
    """Return the minimum storm depth for KE/EI inclusion, in pipeline depth units."""
    sod_in = float(config.sod)
    if str(config.u).upper().startswith("E"):
        return sod_in
    return sod_in * _INCHES_TO_MM


def _storm_omission_flags(storm_df: pd.DataFrame, config: "WeppcliffConfig") -> pd.Series:
    """Per-storm omission flag: 0 kept, 1 omitted."""
    if not config.is_true("os") or "DEPTH" not in storm_df.columns:
        return pd.Series(0, index=storm_df.index, dtype=int)
    threshold = _storm_depth_threshold(config)
    return (storm_df["DEPTH"] < threshold).astype(int)


def _filter_storms_for_erosion(
    storm_df: pd.DataFrame,
    config: "WeppcliffConfig",
) -> pd.DataFrame:
    """Drop storms below ``-sod`` when ``-os`` is enabled."""
    if not config.is_true("os"):
        return storm_df
    if "DEPTH" not in storm_df.columns:
        return storm_df
    threshold = _storm_depth_threshold(config)
    return storm_df.loc[storm_df["DEPTH"] >= threshold].copy()


def annotate_storm_export_frame(
    frame: pd.DataFrame,
    config: "WeppcliffConfig",
) -> pd.DataFrame:
    """Add per-event ``Omitted-Storms`` status (0 keep, 1 omit) to storm export."""
    out = frame.copy()
    out[OMITTED_STORMS_COL] = _storm_omission_flags(out, config).to_numpy()
    return out


def _storm_dataframe(data: "WeppcliffData", config: "WeppcliffConfig") -> pd.DataFrame | None:
    slots = PipelineSlots.for_config(config)
    loc = slots.storm_df
    if loc not in data.store:
        return None
    frame = data.slot_df(loc)
    if frame is None or frame.empty or "DT" not in frame.columns:
        return None
    return frame


def aggregate_omitted_storms(
    storm_df: pd.DataFrame,
    config: "WeppcliffConfig",
    kind: ExportKind,
) -> pd.DataFrame:
    """Count omitted storms per export period (sum; means for climatological kinds)."""
    dt = pd.to_datetime(storm_df["DT"], errors="coerce")
    valid = dt.notna()
    if not valid.any():
        return pd.DataFrame()

    skipna = config.na_rm()
    omitted = _storm_omission_flags(storm_df.loc[valid], config).reset_index(drop=True)
    dt = dt.loc[valid].reset_index(drop=True)
    keyed = pd.DataFrame({OMITTED_STORMS_COL: omitted})

    if kind == "daily_ts":
        keyed.insert(0, "YYYYMMDD", dt.dt.strftime("%Y%m%d").values)
        return (
            keyed.groupby("YYYYMMDD", as_index=False)[OMITTED_STORMS_COL]
            .agg(lambda s: s.sum(skipna=skipna))
            .sort_values("YYYYMMDD")
            .reset_index(drop=True)
        )

    if kind == "monthly_ts":
        keyed.insert(0, "YYYYMM", dt.dt.strftime("%Y%m").values)
        return (
            keyed.groupby("YYYYMM", as_index=False)[OMITTED_STORMS_COL]
            .agg(lambda s: s.sum(skipna=skipna))
            .sort_values("YYYYMM")
            .reset_index(drop=True)
        )

    if kind == "annual_ts":
        keyed.insert(0, "YYYY", dt.dt.strftime("%Y").values)
        return (
            keyed.groupby("YYYY", as_index=False)[OMITTED_STORMS_COL]
            .agg(lambda s: s.sum(skipna=skipna))
            .sort_values("YYYY")
            .reset_index(drop=True)
        )

    if kind == "monthly_mean":
        keyed["YYYYMM"] = dt.dt.strftime("%Y%m").values
        monthly_totals = keyed.groupby("YYYYMM")[OMITTED_STORMS_COL].agg(
            lambda s: s.sum(skipna=skipna),
        )
        monthly_totals = monthly_totals.reset_index()
        monthly_totals["MM"] = pd.to_datetime(
            monthly_totals["YYYYMM"] + "01",
            format="%Y%m%d",
        ).dt.month
        return (
            monthly_totals.groupby("MM", as_index=False)[OMITTED_STORMS_COL]
            .agg(lambda s: s.mean(skipna=skipna))
            .sort_values("MM")
            .reset_index(drop=True)
        )

    if kind == "biweekly_mean":
        keyed["YYYYMM"] = dt.dt.strftime("%Y%m").values
        keyed["HALF"] = np.where(dt.dt.day <= 15, 1, 2)
        biweekly_totals = keyed.groupby(["YYYYMM", "HALF"])[OMITTED_STORMS_COL].agg(
            lambda s: s.sum(skipna=skipna),
        )
        biweekly_totals = biweekly_totals.reset_index()
        biweekly_totals["MM"] = pd.to_datetime(
            biweekly_totals["YYYYMM"] + "01",
            format="%Y%m%d",
        ).dt.month
        return (
            biweekly_totals.groupby(["MM", "HALF"], as_index=False)[OMITTED_STORMS_COL]
            .agg(lambda s: s.mean(skipna=skipna))
            .sort_values(["MM", "HALF"])
            .reset_index(drop=True)
        )

    keyed["YYYY"] = dt.dt.strftime("%Y").values
    annual_totals = keyed.groupby("YYYY")[OMITTED_STORMS_COL].agg(
        lambda s: s.sum(skipna=skipna),
    )
    mean_row = annual_totals.agg(lambda s: s.mean(skipna=skipna))
    out = pd.DataFrame([{OMITTED_STORMS_COL: mean_row}])
    out.insert(0, "STATISTIC", "MEAN")
    return out


def aggregate_erosion_indices(
    storm_df: pd.DataFrame,
    config: "WeppcliffConfig",
    kind: ExportKind,
) -> pd.DataFrame:
    """Aggregate storm KE/EI columns to the export period, honoring ``-pmd``."""
    storm_df = _filter_storms_for_erosion(storm_df, config)
    cols = _erosion_columns(storm_df)
    if not cols:
        return pd.DataFrame()

    skipna = config.na_rm()
    dt = pd.to_datetime(storm_df["DT"], errors="coerce")
    valid = dt.notna()
    if not valid.any():
        return pd.DataFrame()

    keyed = pd.DataFrame({c: storm_df.loc[valid, c].to_numpy() for c in cols})
    dt = dt.loc[valid].reset_index(drop=True)
    keyed = keyed.reset_index(drop=True)

    if keyed.empty:
        return pd.DataFrame()

    if kind == "daily_ts":
        keyed.insert(0, "YYYYMMDD", dt.dt.strftime("%Y%m%d").values)
        return (
            keyed.groupby("YYYYMMDD", as_index=False)[cols]
            .agg(lambda s: s.sum(skipna=skipna))
            .sort_values("YYYYMMDD")
            .reset_index(drop=True)
        )

    if kind == "monthly_ts":
        keyed.insert(0, "YYYYMM", dt.dt.strftime("%Y%m").values)
        return (
            keyed.groupby("YYYYMM", as_index=False)[cols]
            .agg(lambda s: s.sum(skipna=skipna))
            .sort_values("YYYYMM")
            .reset_index(drop=True)
        )

    if kind == "annual_ts":
        keyed.insert(0, "YYYY", dt.dt.strftime("%Y").values)
        return (
            keyed.groupby("YYYY", as_index=False)[cols]
            .agg(lambda s: s.sum(skipna=skipna))
            .sort_values("YYYY")
            .reset_index(drop=True)
        )

    if kind == "monthly_mean":
        keyed["YYYYMM"] = dt.dt.strftime("%Y%m").values
        monthly_totals = keyed.groupby("YYYYMM")[cols].agg(lambda s: s.sum(skipna=skipna))
        monthly_totals = monthly_totals.reset_index()
        monthly_totals["MM"] = pd.to_datetime(
            monthly_totals["YYYYMM"] + "01",
            format="%Y%m%d",
        ).dt.month
        return (
            monthly_totals.groupby("MM", as_index=False)[cols]
            .agg(lambda s: s.mean(skipna=skipna))
            .sort_values("MM")
            .reset_index(drop=True)
        )

    if kind == "biweekly_mean":
        keyed["YYYYMM"] = dt.dt.strftime("%Y%m").values
        keyed["HALF"] = np.where(dt.dt.day <= 15, 1, 2)
        biweekly_totals = keyed.groupby(["YYYYMM", "HALF"])[cols].agg(
            lambda s: s.sum(skipna=skipna),
        )
        biweekly_totals = biweekly_totals.reset_index()
        biweekly_totals["MM"] = pd.to_datetime(
            biweekly_totals["YYYYMM"] + "01",
            format="%Y%m%d",
        ).dt.month
        return (
            biweekly_totals.groupby(["MM", "HALF"], as_index=False)[cols]
            .agg(lambda s: s.mean(skipna=skipna))
            .sort_values(["MM", "HALF"])
            .reset_index(drop=True)
        )

    # annual_mean — mean of annual totals across years
    keyed["YYYY"] = dt.dt.strftime("%Y").values
    annual_totals = keyed.groupby("YYYY")[cols].agg(lambda s: s.sum(skipna=skipna))
    mean_row = annual_totals[cols].agg(lambda s: s.mean(skipna=skipna))
    out = pd.DataFrame([mean_row.to_dict()])
    out.insert(0, "STATISTIC", "MEAN")
    return out


def _build_biweekly_weather_frame(
    data: "WeppcliffData",
    config: "WeppcliffConfig",
) -> pd.DataFrame:
    """Climatological biweekly means from daily weather (days 1-15 and 16-end)."""
    slots = PipelineSlots.for_config(config)
    daily = data.slot_df(slots.daily_ts)
    if daily is None or daily.empty or "YYYYMMDD" not in daily.columns:
        return pd.DataFrame(columns=["MM", "HALF"])

    skipna = config.na_rm()
    ymd_dt = pd.to_datetime(daily["YYYYMMDD"], format="%Y%m%d", errors="coerce")
    valid = ymd_dt.notna()
    if not valid.any():
        return pd.DataFrame(columns=["MM", "HALF"])

    value_cols = [c for c in daily.columns if c != "YYYYMMDD"]
    work = daily.loc[valid, value_cols].copy()
    work["MM"] = ymd_dt.loc[valid].dt.month.to_numpy()
    work["HALF"] = np.where(ymd_dt.loc[valid].dt.day.to_numpy() <= 15, 1, 2)
    out = (
        work.groupby(["MM", "HALF"], as_index=False)[value_cols]
        .agg(lambda s: s.mean(skipna=skipna))
    )
    out["MM"] = pd.to_numeric(out["MM"], errors="coerce").astype(int)
    out["HALF"] = pd.to_numeric(out["HALF"], errors="coerce").astype(int)

    expected = pd.DataFrame(
        [(m, h) for m in range(1, 13) for h in (1, 2)],
        columns=["MM", "HALF"],
    )
    out = expected.merge(out, on=["MM", "HALF"], how="left")
    return out.sort_values(["MM", "HALF"]).reset_index(drop=True)


def _mean_across_periods(frame: pd.DataFrame, config: "WeppcliffConfig") -> pd.DataFrame:
    """Climatological mean across annual periods (one summary row)."""
    skipna = config.na_rm()
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.DataFrame({"STATISTIC": ["MEAN"]})
    mean_row = numeric.agg(lambda s: s.mean(skipna=skipna))
    out = pd.DataFrame([mean_row.to_dict()])
    out.insert(0, "STATISTIC", "MEAN")
    return out


def _merge_export_supplements(
    base: pd.DataFrame,
    config: "WeppcliffConfig",
    *,
    erosion: pd.DataFrame,
    omitted: pd.DataFrame,
    merge_keys: list[str] | None,
) -> pd.DataFrame:
    """Merge KE/EI totals and omitted-storm counts onto a weather export frame."""
    out = base.copy()

    if merge_keys is None:
        if not omitted.empty and OMITTED_STORMS_COL in omitted.columns:
            out[OMITTED_STORMS_COL] = omitted[OMITTED_STORMS_COL].iloc[0]
        elif OMITTED_STORMS_COL not in out.columns:
            out[OMITTED_STORMS_COL] = 0
        if not erosion.empty:
            for col in erosion.columns:
                if col == "STATISTIC":
                    continue
                if col not in out.columns:
                    out[col] = erosion[col].iloc[0]
        return out

    for key in merge_keys:
        out[key] = out[key].astype(str)

    if not omitted.empty:
        omitted = omitted.copy()
        for key in merge_keys:
            omitted[key] = omitted[key].astype(str)
        out = out.merge(omitted, on=merge_keys, how="left")
        out[OMITTED_STORMS_COL] = out[OMITTED_STORMS_COL].fillna(0)
    elif OMITTED_STORMS_COL not in out.columns:
        out[OMITTED_STORMS_COL] = 0

    if erosion.empty:
        return out

    erosion = erosion.copy()
    for key in merge_keys:
        erosion[key] = erosion[key].astype(str)
    out = out.merge(erosion, on=merge_keys, how="left", suffixes=("", "_EIEXP"))
    drop_cols = [c for c in out.columns if c.endswith("_EIEXP")]
    out = out.drop(columns=drop_cols, errors="ignore")

    ke_cols = _erosion_columns(out)
    omitted_mask = out[OMITTED_STORMS_COL] > 0
    if omitted_mask.any() and ke_cols:
        out.loc[omitted_mask, ke_cols] = out.loc[omitted_mask, ke_cols].fillna(0)

    return out


def build_weather_export_frame(
    data: "WeppcliffData",
    config: "WeppcliffConfig",
    kind: ExportKind,
) -> pd.DataFrame:
    """Build a weather export table with KE/EI and omitted-storm summaries."""
    slots = PipelineSlots.for_config(config)

    if kind == "daily_ts":
        base = data.slot_df(slots.daily_ts).copy()
        merge_keys = ["YYYYMMDD"]
    elif kind == "monthly_ts":
        base = data.slot_df(slots.monthly_ts).copy()
        merge_keys = ["YYYYMM"]
    elif kind == "annual_ts":
        base = data.slot_df(slots.annual_ts).copy()
        merge_keys = ["YYYY"]
    elif kind == "monthly_mean":
        base = data.slot_df(slots.monthly_mean).copy()
        merge_keys = ["MM"]
    elif kind == "biweekly_mean":
        base = _build_biweekly_weather_frame(data, config)
        merge_keys = ["MM", "HALF"]
    else:
        annual = data.slot_df(slots.annual_ts).copy()
        base = _mean_across_periods(annual, config)
        merge_keys = None

    storm_df = _storm_dataframe(data, config)
    if storm_df is None:
        base[OMITTED_STORMS_COL] = 0
        return base

    omitted = aggregate_omitted_storms(storm_df, config, kind)
    erosion = (
        aggregate_erosion_indices(storm_df, config, kind)
        if config.is_true("ei")
        else pd.DataFrame()
    )
    return _merge_export_supplements(
        base,
        config,
        erosion=erosion,
        omitted=omitted,
        merge_keys=merge_keys,
    )