"""Daily and alternative data processing — port of R daily functions."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import WeppcliffConfig
from .data_model import WeppcliffData
from .precip import fill_agg_df
from .utils import convert_df_type, groupby_mean, replace_sentinels_columns


def _finalize_daily_ts(df: pd.DataFrame) -> pd.DataFrame:
    out = convert_df_type(df, 1, "c")
    out = convert_df_type(out, slice(2, None), "n")
    return replace_sentinels_columns(out, slice(2, None), vals=-99999, replace=np.nan)


def process_daily_data(
    data: WeppcliffData,
    dly_in: int,
    alt_in: Optional[int],
    dly_ts: int,
    mly_ts: int,
    yly_ts: int,
    mly_mn: int,
    config: WeppcliffConfig,
) -> WeppcliffData:
    """Process daily max/min and aggregate to monthly/annual (R port)."""
    config.verbose("Processing daily data...")
    skipna = config.na_rm()

    dly_df = data[dly_in].dropna(subset=[data[dly_in].columns[0]]).copy()
    dt_col = dly_df.columns[0]
    dates = pd.to_datetime(dly_df[dt_col]).dt.strftime("%Y%m%d")

    daily_agg = dly_df.groupby(dates).agg(
        MIN_TEMP=("MIN_TEMP", lambda s: s.min(skipna=skipna)),
        MAX_TEMP=("MAX_TEMP", lambda s: s.max(skipna=skipna)),
        DP_TEMP=("DP_TEMP", lambda s: s.mean(skipna=skipna)),
        W_VEL=("W_VEL", lambda s: s.mean(skipna=skipna)),
        W_DIR=("W_DIR", lambda s: s.mean(skipna=skipna)),
        SO_RAD=("SO_RAD", lambda s: s.sum(skipna=skipna)),
    )

    d_dly = pd.DataFrame({
        "YYYYMMDD": daily_agg.index.astype(str),
        "MIN_TEMP": daily_agg["MIN_TEMP"].values,
        "MAX_TEMP": daily_agg["MAX_TEMP"].values,
        "SO_RAD": daily_agg["SO_RAD"].values,
        "W_VEL": daily_agg["W_VEL"].values,
        "W_DIR": daily_agg["W_DIR"].values,
        "DP_TEMP": daily_agg["DP_TEMP"].values,
    })
    d_dly = _finalize_daily_ts(d_dly)
    agg_names = ["MIN_TEMP", "MAX_TEMP", "SO_RAD", "W_VEL", "W_DIR", "DP_TEMP"]

    if config.do_alt and alt_in is not None:
        d_alt = process_alt_data(data, alt_in, config)
        d_dly = use_alt_data(d_alt, d_dly, config)
        agg_names = agg_names + ["AIR_TEMP", "REL_HUM"]

    data[dly_ts] = d_dly

    ymd_dt = pd.to_datetime(d_dly["YYYYMMDD"], format="%Y%m%d")
    yyyymm = ymd_dt.dt.strftime("%Y%m")
    yyyy = ymd_dt.dt.strftime("%Y")

    mn_ts = groupby_mean(d_dly.iloc[:, 1:], yyyymm, skipna=skipna)
    mn_ts.columns = ["YYYYMM"] + agg_names
    mn_ts = convert_df_type(mn_ts, slice(1, None), "n")
    mn_ts = replace_sentinels_columns(mn_ts, slice(2, None), vals=-99999, replace=np.nan)
    data[mly_ts] = mn_ts

    an_ts = groupby_mean(d_dly.iloc[:, 1:], yyyy, skipna=skipna)
    an_ts.columns = ["YYYY"] + agg_names
    an_ts = convert_df_type(an_ts, slice(1, None), "n")
    an_ts = replace_sentinels_columns(an_ts, slice(2, None), vals=-99999, replace=np.nan)
    data[yly_ts] = an_ts

    mm = ymd_dt.dt.strftime("%m").astype(int)
    m_mean = groupby_mean(d_dly.iloc[:, 1:], mm, skipna=skipna)
    m_mean.columns = ["MM"] + agg_names
    m_mean = convert_df_type(m_mean, slice(1, None), "n")
    m_mean = replace_sentinels_columns(m_mean, slice(2, None), vals=-99999, replace=np.nan)
    m_mean = fill_agg_df(m_mean, "MM", list(range(1, 13)), np.nan, config)
    m_mean = convert_df_type(m_mean, slice(1, None), "n")
    data[mly_mn] = m_mean

    return data


def process_alt_data(
    data: WeppcliffData,
    alt_in: int,
    config: WeppcliffConfig,
) -> pd.DataFrame:
    """Process sub-daily air temperature and humidity (R ``process_alt_data``)."""
    config.verbose("Processing alternative data...")
    alt_df = data[alt_in].copy()
    alt_df["DT_2"] = pd.to_datetime(alt_df["DT_2"])
    dates = alt_df["DT_2"].dt.strftime("%Y%m%d")

    dp_temp = dew_point_temperature(
        alt_df["AIR_TEMP"].to_numpy(dtype=float),
        alt_df["REL_HUM"].to_numpy(dtype=float),
    )
    dp_series = pd.Series(dp_temp, index=alt_df.index)
    dp_series = dp_series.mask(dp_series == -99999)

    alt_agg = alt_df.assign(_dp=dp_series).groupby(dates).agg(
        ALT_MIN_TEMP=("AIR_TEMP", "min"),
        ALT_MAX_TEMP=("AIR_TEMP", "max"),
        ALT_DP_TEMP=("_dp", "mean"),
        AIR_TEMP=("AIR_TEMP", "mean"),
        REL_HUM=("REL_HUM", "mean"),
    )

    d_alt = pd.DataFrame({
        "YYYYMMDD": alt_agg.index.astype(str),
        "ALT_MIN_TEMP": alt_agg["ALT_MIN_TEMP"].values,
        "ALT_MAX_TEMP": alt_agg["ALT_MAX_TEMP"].values,
        "ALT_DP_TEMP": alt_agg["ALT_DP_TEMP"].values,
        "AIR_TEMP": alt_agg["AIR_TEMP"].values,
        "REL_HUM": alt_agg["REL_HUM"].values,
    })
    return _finalize_daily_ts(d_alt)


def dew_point_temperature(
    air_temp: np.ndarray,
    rel_hum: np.ndarray,
) -> np.ndarray:
    """Calculate dew point temperature (R ``dew_point_temperature``)."""
    rh = np.where(rel_hum == 0, 1, rel_hum)
    a1 = 7.625
    b1 = 243.04
    top = b1 * (np.log(rh / 100) + (a1 * air_temp) / (b1 + air_temp))
    bottom = a1 - np.log(rh / 100) - (a1 * air_temp) / (b1 + air_temp)
    return top / bottom


def use_alt_data(
    d_alt: pd.DataFrame,
    d_dly: pd.DataFrame,
    config: WeppcliffConfig,
) -> pd.DataFrame:
    """Replace daily temperature with alternative data (R ``use_alt_data``)."""
    merged = pd.merge(d_dly, d_alt, on="YYYYMMDD", how="left")
    out_df = merged[
        [
            "YYYYMMDD", "ALT_MIN_TEMP", "ALT_MAX_TEMP", "SO_RAD", "W_VEL", "W_DIR",
            "DP_TEMP", "AIR_TEMP", "REL_HUM",
        ]
    ].rename(columns={"ALT_MIN_TEMP": "MIN_TEMP", "ALT_MAX_TEMP": "MAX_TEMP"})
    out_df = convert_df_type(out_df, 1, "c")
    out_df = convert_df_type(out_df, slice(2, None), "n")
    out_df = replace_sentinels_columns(out_df, slice(2, None), vals=-99999, replace=np.nan)
    return out_df


def annual_summary(
    data: WeppcliffData,
    yly_ts: int,
    an_sum_loc: int,
    config: WeppcliffConfig,
) -> WeppcliffData:
    """Store annual time series summary (R ``annual_summary``)."""
    data[an_sum_loc] = data[yly_ts].describe()
    return data