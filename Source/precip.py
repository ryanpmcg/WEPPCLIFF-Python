"""Precipitation processing — port of R precip/aggregate functions."""

from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np
import pandas as pd

from .config import WeppcliffConfig
from .data_model import WeppcliffData
from .utils import convert_df_type, df_find_replace, replace_sentinels_columns


def _finalize_numeric_ts(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """Apply standard type conversion and sentinel replacement to a time-series frame."""
    out = convert_df_type(df, 1, "c")
    out = convert_df_type(out, slice(2, None), "n")
    return replace_sentinels_columns(
        out,
        slice(2, None),
        vals=-99999,
        replace=np.nan,
    )


def _sid_unique_stats(sids: pd.Series) -> pd.Series:
    """ELEN/EMAX/EMIN from unique storm IDs per day (matches R aggregate(unique))."""
    unique = sids.unique()
    return pd.Series(
        {"ELEN": float(len(unique)), "EMAX": float(unique.max()), "EMIN": float(unique.min())}
    )


def aggregate_precip_data(
    data: WeppcliffData,
    config: WeppcliffConfig,
) -> WeppcliffData:
    """Aggregate precipitation breakpoints to a specific interval (R port)."""
    df = data[1].copy()
    ai = float(config.ai)
    ai_int = int(ai)

    dt = pd.to_datetime(df["DT_1"])
    df["Year"] = dt.dt.year
    df["Month"] = dt.dt.month
    df["Day"] = dt.dt.day
    df["Hour"] = dt.dt.hour
    df["Minute"] = dt.dt.minute
    df["Cluster"] = (df["Hour"] * 60 + df["Minute"]) // ai_int

    agg_df = (
        df.groupby(["Year", "Month", "Day", "Cluster"], as_index=False)["PRECIP"]
        .sum(min_count=1)
    )
    agg_df["Cluster"] = agg_df["Cluster"] * ai
    agg_df["Hour"] = agg_df["Cluster"] // 60
    agg_df["Minute"] = agg_df["Cluster"] - agg_df["Hour"] * 60
    agg_df["DT_1"] = pd.to_datetime({
        "year": agg_df["Year"],
        "month": agg_df["Month"],
        "day": agg_df["Day"],
        "hour": agg_df["Hour"],
        "minute": agg_df["Minute"],
        "second": 0,
    })
    agg_df["DT_2"] = agg_df["DT_1"] - pd.to_timedelta(ai, unit="m")

    all_dts = agg_df["DT_1"].drop_duplicates().union(agg_df["DT_2"].drop_duplicates())
    unique_dts = pd.DataFrame({"DT_1": all_dts.sort_values()})
    out_df = unique_dts.merge(
        agg_df[["DT_1", "PRECIP"]],
        on="DT_1",
        how="left",
    )
    out_df["DUR"] = ai
    out_df["PRECIP"] = out_df["PRECIP"].fillna(0)
    out_df["DT_1"] = pd.to_datetime(out_df["DT_1"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    data[1] = out_df
    return data


def fill_agg_df(
    agg_df: pd.DataFrame,
    col_name: str,
    obs: Sequence[Union[int, float]],
    val: Union[int, float, None],
    config: WeppcliffConfig,
) -> pd.DataFrame:
    """Fill missing rows in an aggregated dataframe (R ``fill_agg_df``)."""
    present = set(pd.to_numeric(agg_df[col_name], errors="coerce").dropna().astype(int).tolist())
    missing = [o for o in obs if o not in present]

    if not missing:
        return agg_df

    na_matrix = np.full((len(missing), agg_df.shape[1] - 1), val, dtype=float)
    add_df = pd.DataFrame(na_matrix, columns=agg_df.columns[1:])
    add_df.insert(0, col_name, missing)
    out_df = pd.concat([add_df, agg_df], ignore_index=True)
    out_df = out_df.sort_values(by=col_name, key=lambda s: pd.to_numeric(s)).reset_index(drop=True)
    return out_df


def process_precip_data(
    data: WeppcliffData,
    pcp_in: int,
    dly_ts: int,
    mly_ts: int,
    yly_ts: int,
    mly_mn: int,
    config: WeppcliffConfig,
) -> WeppcliffData:
    """Process precipitation into daily/monthly/annual aggregates (R port)."""
    config.verbose("Processing precipitation data...")
    skipna = config.na_rm()

    pcp_df = data[pcp_in].copy()
    dt = pd.to_datetime(pcp_df["DT_1"])
    dates = dt.dt.strftime("%Y%m%d")

    daily_stats = pcp_df.groupby(dates).agg(
        BPS=("PRECIP", "size"),
        PRECIP=("PRECIP", lambda x: x.sum(skipna=skipna)),
    )
    sid_stats = pcp_df.groupby(dates)["SID"].apply(_sid_unique_stats).unstack()

    d_precip = pd.DataFrame({
        "YYYYMMDD": daily_stats.index.astype(str),
        "BPS": daily_stats["BPS"].astype(float).values,
        "ELEN": sid_stats["ELEN"].reindex(daily_stats.index).values,
        "EMAX": sid_stats["EMAX"].reindex(daily_stats.index).values,
        "EMIN": sid_stats["EMIN"].reindex(daily_stats.index).values,
        "PRECIP": daily_stats["PRECIP"].values,
    })
    d_precip = _finalize_numeric_ts(d_precip, "YYYYMMDD")

    if not config.cv_off:
        left = data[dly_ts].copy()
        left["YYYYMMDD"] = left["YYYYMMDD"].astype(str)
        d_precip["YYYYMMDD"] = d_precip["YYYYMMDD"].astype(str)
        d_precip = pd.merge(left, d_precip, on="YYYYMMDD", how="left")
    data[dly_ts] = d_precip

    ymd_dt = pd.to_datetime(d_precip["YYYYMMDD"], format="%Y%m%d")
    yyyymm = ymd_dt.dt.strftime("%Y%m")
    yyyy = ymd_dt.dt.strftime("%Y")

    m_agg = d_precip.groupby(yyyymm).agg(
        PRECIP=("PRECIP", "sum"),
        BPS=("BPS", "sum"),
        ELEN=("ELEN", "sum"),
    )
    m_ts = pd.DataFrame({
        "YYYYMM": m_agg.index.astype(str),
        "PRECIP": m_agg["PRECIP"].values,
        "BPS": m_agg["BPS"].values,
        "ELEN": m_agg["ELEN"].values,
    })
    m_ts = _finalize_numeric_ts(m_ts, "YYYYMM")
    if not config.cv_off:
        left = data[mly_ts].copy()
        left["YYYYMM"] = left["YYYYMM"].astype(str)
        m_ts["YYYYMM"] = m_ts["YYYYMM"].astype(str)
        m_ts = pd.merge(left, m_ts, on="YYYYMM", how="left")
    data[mly_ts] = m_ts

    an_agg = d_precip.groupby(yyyy).agg(
        PRECIP=("PRECIP", "sum"),
        BPS=("BPS", "sum"),
        ELEN=("ELEN", "sum"),
    )
    an_ts = pd.DataFrame({
        "YYYY": an_agg.index.astype(str),
        "PRECIP": an_agg["PRECIP"].values,
        "BPS": an_agg["BPS"].values,
        "ELEN": an_agg["ELEN"].values,
    })
    an_ts = _finalize_numeric_ts(an_ts, "YYYY")
    if not config.cv_off:
        left = data[yly_ts].copy()
        left["YYYY"] = left["YYYY"].astype(str)
        an_ts["YYYY"] = an_ts["YYYY"].astype(str)
        an_ts = pd.merge(left, an_ts, on="YYYY", how="left")
    data[yly_ts] = an_ts

    month_key = pd.to_datetime(m_ts["YYYYMM"] + "01", format="%Y%m%d").dt.month
    m_mean_agg = m_ts.groupby(month_key).agg(
        PRECIP=("PRECIP", lambda s: s.mean(skipna=skipna)),
        BPS=("BPS", lambda s: s.mean(skipna=skipna)),
        ELEN=("ELEN", lambda s: s.mean(skipna=skipna)),
    )
    m_mn = pd.DataFrame({
        "MM": m_mean_agg.index.astype(float),
        "PRECIP": m_mean_agg["PRECIP"].values,
        "BPS": m_mean_agg["BPS"].values,
        "ELEN": m_mean_agg["ELEN"].values,
    })
    m_mn = convert_df_type(m_mn, slice(1, None), "n")
    m_mn = replace_sentinels_columns(m_mn, slice(2, None), vals=-99999, replace=np.nan)
    m_mn = fill_agg_df(m_mn, "MM", list(range(1, 13)), np.nan, config)
    m_mn = convert_df_type(m_mn, slice(1, None), "n")
    if not config.cv_off:
        left = data[mly_mn].copy()
        left["MM"] = pd.to_numeric(left["MM"], errors="coerce")
        m_mn["MM"] = pd.to_numeric(m_mn["MM"], errors="coerce")
        m_mn = pd.merge(left, m_mn, on="MM", how="left")
    data[mly_mn] = m_mn

    return data