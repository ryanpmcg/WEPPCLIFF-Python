"""Storm separation — port of R trim/break/identify storm functions."""

from __future__ import annotations

from typing import List, Union

import numpy as np
import pandas as pd

from .config import WeppcliffConfig
from .data_model import WeppcliffData
from .time_bounds import time_bounds

DT_LIST = ["DT_1", "DT_2", "DT_3"]


def trim_data(
    data: WeppcliffData,
    precip_ts_df: int,
    alt_ts_df: int,
    d_ts_df: int,
    event_list: int,
    config: WeppcliffConfig,
) -> WeppcliffData:
    """Trim data to acceptable period and update metadata (R ``trim_data``)."""
    config.verbose("Trimming to specified time period...")
    data = break_storms(data, precip_ts_df, event_list, config)

    maxs: List[pd.Timestamp] = []
    mins: List[pd.Timestamp] = []

    if config.is_true("alt") and not config.is_true("ipb") and config.cv.upper() != "F":
        maxs = [
            pd.to_datetime(data[precip_ts_df]["DT_1"]).max(),
            pd.to_datetime(data[alt_ts_df]["DT_2"]).max(),
            pd.to_datetime(data[d_ts_df]["DT_3"]).max(),
        ]
        mins = [
            pd.to_datetime(data[precip_ts_df]["DT_1"]).min(),
            pd.to_datetime(data[alt_ts_df]["DT_2"]).min(),
            pd.to_datetime(data[d_ts_df]["DT_3"]).min(),
        ]
    elif not config.is_true("alt") and not config.is_true("ipb") and config.cv.upper() != "F":
        maxs = [
            pd.to_datetime(data[precip_ts_df]["DT_1"]).max(),
            pd.to_datetime(data[d_ts_df]["DT_3"]).max(),
        ]
        mins = [
            pd.to_datetime(data[precip_ts_df]["DT_1"]).min(),
            pd.to_datetime(data[d_ts_df]["DT_3"]).min(),
        ]
    elif config.is_true("alt") and config.is_true("ipb") and config.cv.upper() != "F":
        maxs = [
            pd.to_datetime(data[alt_ts_df]["DT_2"]).max(),
            pd.to_datetime(data[d_ts_df]["DT_3"]).max(),
        ]
        mins = [
            pd.to_datetime(data[alt_ts_df]["DT_2"]).min(),
            pd.to_datetime(data[d_ts_df]["DT_3"]).min(),
        ]
    elif not config.is_true("alt") and config.is_true("ipb") and config.cv.upper() != "F":
        maxs = [pd.to_datetime(data[d_ts_df]["DT_3"]).max()]
        mins = [pd.to_datetime(data[d_ts_df]["DT_3"]).min()]
    elif config.cv.upper() == "F":
        maxs = [pd.to_datetime(data[precip_ts_df]["DT_1"]).max()]
        mins = [pd.to_datetime(data[precip_ts_df]["DT_1"]).min()]

    start = max(mins)
    end = min(maxs)
    bounds = time_bounds(start, end, config)

    if config.is_true("alt"):
        groups = [5, 6, 7]
    else:
        groups = [5, 7]
    if config.cv.upper() == "F":
        groups = [5]

    for j in groups:
        df = data[j]
        dt_col = DT_LIST[j - 5]
        dt_series = pd.to_datetime(df[dt_col])
        mask = (dt_series >= bounds[0]) & (dt_series <= bounds[1])
        data[j] = df.loc[mask].reset_index(drop=True)

    config.days_count = (bounds[1] - bounds[0]).total_seconds() / 86400.0
    meta = data.metadata
    meta["B_YR"] = bounds[0].strftime("%Y")
    meta["OBS_YRS"] = (bounds[3] - bounds[2]).total_seconds() / (365.2425 * 86400)
    meta["YRS_SIM"] = (bounds[1] - bounds[0]).total_seconds() / (365.2425 * 86400)
    data[4] = meta

    return data


def break_storms(
    data: WeppcliffData,
    in_index: int,
    out_index: int,
    config: WeppcliffConfig,
) -> WeppcliffData:
    """Separate storms and store list at out_index (R ``break_storms``)."""
    config.verbose("Separating storms...")
    df = data[in_index].dropna(subset=["PRECIP"]).reset_index(drop=True)
    df = cum_to_inc(df, config)
    df = identify_storms(df, config)
    if config.is_true("ei"):
        df = int_30m(df, config)
    storm_list = [group.reset_index(drop=True) for _, group in df.groupby("SID", sort=True)]
    data[in_index] = df
    data[out_index] = storm_list
    return data


def cum_to_inc(
    df: pd.DataFrame,
    config: WeppcliffConfig,
) -> pd.DataFrame:
    """Convert cumulative precipitation to incremental (R ``cum_to_inc``)."""
    out = df.copy()
    if config.is_true("cp"):
        config.verbose("Converting to incremental precipitation...")
        pcp = out["PRECIP"].to_numpy(dtype=float)
        difs = np.concatenate([[0.0], np.diff(pcp)])
        neg_idx = np.where(difs < 0)[0]
        if len(neg_idx) > 0:
            difs[neg_idx] = pcp[neg_idx]
        out["PRECIP"] = difs.astype(float)

    zeros = out.index[out["PRECIP"] == 0].tolist()
    if zeros:
        out = out.drop(index=zeros).reset_index(drop=True)
    out["INT"] = out["PRECIP"] / (out["DUR"] / 60.0)
    return out


def identify_storms(
    df: pd.DataFrame,
    config: WeppcliffConfig,
) -> pd.DataFrame:
    """Identify storm events via connectivity rules (R ``identify_storms``)."""
    out = df.copy()
    if out.empty:
        for col in ("ANT", "BTW", "CUM", "D_CON", "T_CON", "SID"):
            out[col] = pd.Series(dtype=float)
        return out
    dt = pd.to_datetime(out["DT_1"])
    dif = np.concatenate([
        [out["DUR"].iloc[0]],
        (dt.diff().dt.total_seconds() / 60.0).iloc[1:].to_numpy(),
    ])
    out["ANT"] = dif
    out["BTW"] = out["ANT"] - out["DUR"]
    out["CUM"] = out["ANT"].cumsum()
    out = evaluate_connectivity(out, config)

    d_con = out["D_CON"].astype(int)
    t_con = out["T_CON"].astype(int)
    d_logic = np.concatenate([
        [1],
        (~(d_con.iloc[1:].to_numpy() & d_con.iloc[:-1].to_numpy())).astype(int),
    ])
    e_logic = (d_logic == 1) | (t_con == 1)
    out["SID"] = np.cumsum(e_logic.astype(int))
    return out


def evaluate_connectivity(
    df: pd.DataFrame,
    config: WeppcliffConfig,
) -> pd.DataFrame:
    """Evaluate storm connectivity using depth and time thresholds."""
    out = df.copy()
    minutes = float(config.tth) * 60.0
    positive_dur = out.loc[out["DUR"] > 0, "DUR"]
    min_dur = positive_dur.min() if len(positive_dur) > 0 else 1.0
    rows = int(np.ceil(minutes / min_dur))

    out["SUM_BEF_TTH"] = smart_window_sum(
        out["CUM"].to_numpy(),
        out["PRECIP"].to_numpy(),
        out["DUR"].to_numpy(),
        rows,
        "reverse",
        minutes,
    )
    out["SUM_AFT_TTH"] = smart_window_sum(
        out["CUM"].to_numpy(),
        out["PRECIP"].to_numpy(),
        out["DUR"].to_numpy(),
        rows,
        "forward",
        minutes,
    )
    out["D_CON"] = (out["SUM_BEF_TTH"] >= float(config.dth)) | (
        out["SUM_AFT_TTH"] >= float(config.dth)
    )
    out["T_CON"] = out["BTW"] >= minutes
    return out


def int_30m(
    df: pd.DataFrame,
    config: WeppcliffConfig,
) -> pd.DataFrame:
    """Calculate 30-minute intensities (R ``int_30m``)."""
    config.verbose("Calculating 30-minute intensities...")
    out = df.copy()
    minutes = 30.0
    positive_dur = out.loc[out["DUR"] > 0, "DUR"]
    min_dur = positive_dur.min() if len(positive_dur) > 0 else 1.0
    rows = int(np.ceil(minutes / min_dur))

    out["INT_BEF_30"] = (
        smart_window_sum(
            out["CUM"].to_numpy(),
            out["PRECIP"].to_numpy(),
            out["DUR"].to_numpy(),
            rows,
            "reverse",
            minutes,
        )
        * 2
    )
    out["INT_AFT_30"] = (
        smart_window_sum(
            out["CUM"].to_numpy(),
            out["PRECIP"].to_numpy(),
            out["DUR"].to_numpy(),
            rows,
            "forward",
            minutes,
        )
        * 2
    )
    return out


def _smart_window_sum_reference(
    cum_vec: np.ndarray,
    sum_vec: np.ndarray,
    dur_vec: np.ndarray,
    rows: int,
    direction: str,
    threshold: float,
) -> np.ndarray:
    """Reference implementation — slice-based, no per-iteration arange."""
    n = len(cum_vec)
    sum_out = np.full(n, np.nan, dtype=np.float64)
    is_reverse = direction == "reverse"

    for j in range(n):
        w0 = max(j - rows, 0)
        w1 = min(j + rows, n - 1) + 1
        sum_var = 0.0
        matched = False

        if is_reverse:
            now = cum_vec[j]
            lim = max(-1.0, now - threshold)
            edge_cross = 0.0
            edge_sum = 0.0
            edge_dur = 0.0
            first = True
            for k in range(w0, w1):
                c = cum_vec[k]
                if c <= now and c > lim:
                    matched = True
                    s = sum_vec[k]
                    d = dur_vec[k]
                    if not np.isnan(s):
                        sum_var += s
                    if first:
                        edge_cross = lim - (c - d)
                        edge_sum = s
                        edge_dur = d
                        first = False
        else:
            now = cum_vec[j] - dur_vec[j]
            lim = now + threshold
            edge_cross = 0.0
            edge_sum = 0.0
            edge_dur = 0.0
            for k in range(w0, w1):
                c = cum_vec[k]
                d = dur_vec[k]
                if c > now and (c - d) < lim:
                    matched = True
                    s = sum_vec[k]
                    if not np.isnan(s):
                        sum_var += s
                    edge_cross = c - lim
                    edge_sum = s
                    edge_dur = d

        if not matched:
            sum_out[j] = np.nan
            continue

        if is_reverse:
            if edge_cross > 0 and edge_dur > 0:
                sum_var -= edge_sum * edge_cross / edge_dur
        else:
            if edge_cross > 0 and edge_dur > 0:
                sum_var -= edge_sum * edge_cross / edge_dur

        sum_out[j] = sum_var

    return sum_out


try:
    from numba import njit

    @njit(cache=True)
    def _smart_window_sum_numba(
        cum_vec: np.ndarray,
        sum_vec: np.ndarray,
        dur_vec: np.ndarray,
        rows: int,
        is_reverse: bool,
        threshold: float,
    ) -> np.ndarray:
        n = len(cum_vec)
        sum_out = np.empty(n, dtype=np.float64)
        for j in range(n):
            w0 = max(j - rows, 0)
            w1 = min(j + rows, n - 1) + 1
            sum_var = 0.0
            matched = False
            edge_cross = 0.0
            edge_sum = 0.0
            edge_dur = 0.0
            last_k = -1

            if is_reverse:
                now = cum_vec[j]
                lim = max(-1.0, now - threshold)
                first = True
                for k in range(w0, w1):
                    c = cum_vec[k]
                    if c <= now and c > lim:
                        matched = True
                        s = sum_vec[k]
                        d = dur_vec[k]
                        if not np.isnan(s):
                            sum_var += s
                        if first:
                            edge_cross = lim - (c - d)
                            edge_sum = s
                            edge_dur = d
                            first = False
            else:
                now = cum_vec[j] - dur_vec[j]
                lim = now + threshold
                for k in range(w0, w1):
                    c = cum_vec[k]
                    d = dur_vec[k]
                    if c > now and (c - d) < lim:
                        matched = True
                        s = sum_vec[k]
                        if not np.isnan(s):
                            sum_var += s
                        edge_cross = c - lim
                        edge_sum = s
                        edge_dur = d

            if not matched:
                sum_out[j] = np.nan
            else:
                if edge_cross > 0.0 and edge_dur > 0.0:
                    sum_var -= edge_sum * edge_cross / edge_dur
                sum_out[j] = sum_var
        return sum_out

    _USE_NUMBA = True
except ImportError:
    _USE_NUMBA = False


def smart_window_sum(
    cum_vec: np.ndarray,
    sum_vec: np.ndarray,
    dur_vec: np.ndarray,
    rows: int,
    direction: str,
    threshold: float,
) -> np.ndarray:
    """Sum precipitation over a fixed time window (R ``smart_window_sum``)."""
    cum = np.asarray(cum_vec, dtype=np.float64)
    summ = np.asarray(sum_vec, dtype=np.float64)
    dur = np.asarray(dur_vec, dtype=np.float64)
    is_reverse = direction == "reverse"

    if _USE_NUMBA:
        return _smart_window_sum_numba(cum, summ, dur, rows, is_reverse, float(threshold))
    return _smart_window_sum_reference(cum, summ, dur, rows, direction, threshold)