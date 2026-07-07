"""Quality control pipeline — port of R QC functions."""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from .config import WeppcliffConfig
from .data_model import WeppcliffData
from .utils import rle

NM = [
    "PRECIP", "AIR_TEMP", "REL_HUM", "MIN_TEMP", "MAX_TEMP",
    "SO_RAD", "W_VEL", "W_DIR", "DP_TEMP",
]
R_LOC = [5, 6, 6, 7, 7, 7, 7, 7, 7]
I_LOC = [5, 6, 6, 7, 7, 7, 7, 7, 7]
I_UV_LIMS = [2000, 70, 1000, 45, 60, 2865, 120, 360, 40]
I_LV_LIMS = [0, -100, 0, -90, -20, 0, 0, 0, -25]
I_R_LIMS = [35, 1, 1, 1, 1, 30, 1, 360, 1]
D_U_LIMS = [2000, 60, 500, 45, 60, 1432, 60, 360, 40]
D_L_LIMS = [0, -90, 0, -90, -20, 1, 0, 0, -25]
DAILY_LIMITS = dict(zip(NM, zip(D_U_LIMS, D_L_LIMS)))


def initialize_qc_data(
    data: WeppcliffData,
    config: WeppcliffConfig,
) -> WeppcliffData:
    """Copy groups 1–3 into QC slots 5–7 (R ``initialize_qc_data``)."""
    for j in (1, 2, 3):
        data[j + 4] = data.slot_df(j).copy()
    return data


def quality_check_inputs(
    data: WeppcliffData,
    config: WeppcliffConfig,
) -> WeppcliffData:
    """Run full input QC pipeline."""
    config.verbose("Checking input data quality...")
    data = create_rate_qc_data(data, config)
    data = run_nonprecip_checks(data, config)
    data = run_precip_checks(data, config)
    data = remove_qc_fails(data, config)
    return data


def create_rate_qc_data(
    data: WeppcliffData,
    config: WeppcliffConfig,
) -> WeppcliffData:
    """Add delta and QC columns for breakpoint data (R ``create_rate_qc_data``)."""
    for j, name in enumerate(NM):
        loc = R_LOC[j]
        df = data.slot_df(loc)
        if name not in df.columns:
            continue
        var = df[name].to_numpy(dtype=float)
        if len(var) == 0:
            continue
        del_vals = np.concatenate(([0.0], np.diff(var)))
        if len(del_vals) == len(var):
            df[f"{name}_DEL"] = del_vals
        df[f"{name}_QC"] = "PASS"
        data[loc] = df
    return data


def _out_of_limits_mask(
    arr: np.ndarray,
    upper_limit: float,
    lower_limit: float,
) -> np.ndarray:
    num_loc = ~np.isnan(arr)
    return num_loc & ((arr > upper_limit) | (arr < lower_limit))


def extreme_limit_test(
    data_vec: np.ndarray,
    qcr_vec: np.ndarray,
    upper_limit: float,
    lower_limit: float,
    flag: str,
) -> np.ndarray:
    """Flag values outside physical limits (R ``extreme_limit_test``)."""
    out = qcr_vec.copy()
    mask = _out_of_limits_mask(np.asarray(data_vec, dtype=float), upper_limit, lower_limit)
    if mask.any():
        out[mask] = flag
    return out


def extreme_limit_replace(
    data_vec: np.ndarray,
    upper_limit: float,
    lower_limit: float,
    flag: float,
) -> np.ndarray:
    """Replace out-of-limit values with flag (R ``extreme_limit_replace``)."""
    out = data_vec.copy()
    mask = _out_of_limits_mask(out, upper_limit, lower_limit)
    if mask.any():
        out[mask] = flag
    return out


def rle_streak_test(
    data_vec: np.ndarray,
    qcr_vec: np.ndarray,
    str_th: int,
    flag: str,
) -> np.ndarray:
    """Flag consecutive identical value streaks (R ``rle_streak_test``)."""
    out = qcr_vec.copy()
    lengths, _ = rle(data_vec)
    loc_data = np.cumsum(lengths)
    streak_intersect = np.where(lengths >= str_th)[0]
    for idx in streak_intersect:
        streak_end = loc_data[idx]
        streak_len = lengths[idx]
        streak_begin = streak_end - streak_len
        out[streak_begin:streak_end] = flag
    return out


def snr_spike_test(
    dur_vec: np.ndarray,
    del_vec: np.ndarray,
    rat_vec: np.ndarray,
    qcr_vec: np.ndarray,
    chk_th: float,
    spk_th: float,
    flag: str,
) -> np.ndarray:
    """Flag spikes using SNR heuristic (R ``snr_spike_test``)."""
    out = qcr_vec.copy()
    noise = np.nanvar(rat_vec)
    if noise == 0 or np.isnan(noise):
        return out

    snr_vec = np.abs(rat_vec) / noise
    lcrit = len(snr_vec)
    crit_diff = 1 - spk_th

    short_dur = dur_vec <= 360
    high_dels = np.abs(del_vec) > np.nanquantile(np.abs(del_vec), chk_th)
    high_rats = np.abs(rat_vec) > np.nanquantile(np.abs(rat_vec), chk_th)
    high_snrs = snr_vec > np.nanquantile(snr_vec, chk_th)
    highs = np.where(short_dur & high_dels & high_rats & high_snrs)[0]
    highs = highs[highs < lcrit - 1]

    if len(highs) == 0:
        return out

    heads = rat_vec[highs]
    tails = rat_vec[highs + 1]
    denom = np.abs(heads) + np.abs(tails)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_diff = np.abs(heads + tails) / denom
    out[highs[norm_diff < crit_diff]] = flag
    return out


def rle_stick_test(
    qcr_vec: np.ndarray,
    stk_th: int,
    flag: str,
) -> np.ndarray:
    """Flag short PASS segments between identical QC flags (R ``rle_stick_test``)."""
    out = qcr_vec.copy()
    lengths, values = rle(qcr_vec)
    loc_data = np.cumsum(lengths)
    short_lengths = np.where(lengths <= stk_th)[0]
    qcr_passes = np.where(values == "PASS")[0]
    qcr_checks = np.intersect1d(short_lengths, qcr_passes)

    if len(qcr_checks) == 0:
        return out

    for check in qcr_checks:
        start_idx = check - 1
        end_idx = check + 1
        if start_idx >= 0 and end_idx < len(values):
            if values[start_idx] == values[end_idx]:
                out[loc_data[check] - 1] = flag
    return out


def run_nonprecip_checks(
    data: WeppcliffData,
    config: WeppcliffConfig,
) -> WeppcliffData:
    """Impose non-precipitation QC checks (R ``run_nonprecip_checks``)."""
    qcop = config.qcop.upper()
    chkth = float(config.chkth)
    spkth = float(config.spkth)
    strth = int(config.strth)
    stkth = int(config.stkth)

    slot_indices = {6, 7}
    for loc in slot_indices:
        df = data.slot_df(loc)
        if df.empty:
            continue
        dur_vec = df["DUR"].to_numpy(dtype=float)
        updated = False

        for j in range(1, len(NM)):
            if I_LOC[j] != loc:
                continue
            name = NM[j]
            del_col = f"{name}_DEL"
            qc_col = f"{name}_QC"
            if del_col not in df.columns or qc_col not in df.columns:
                continue

            var_vec = df[name].to_numpy(dtype=float)
            del_vec = df[del_col].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                rat_vec = del_vec / dur_vec
            qcr_vec = df[qc_col].to_numpy(dtype=object)

            if qcop in ("B", "P"):
                qcr_vec = extreme_limit_test(var_vec, qcr_vec, I_UV_LIMS[j], I_LV_LIMS[j], "VALUE")
                qcr_vec = extreme_limit_test(rat_vec, qcr_vec, I_R_LIMS[j], -I_R_LIMS[j], "RATE")

            if qcop in ("B", "E"):
                qcr_vec = rle_streak_test(var_vec, qcr_vec, strth, "STREAK")
                qcr_vec = snr_spike_test(dur_vec, del_vec, rat_vec, qcr_vec, chkth, spkth, "SPIKE")
                qcr_vec = rle_stick_test(qcr_vec, stkth, "STICK")

            df[qc_col] = qcr_vec
            updated = True

        if updated:
            data[loc] = df
    return data


def run_precip_checks(
    data: WeppcliffData,
    config: WeppcliffConfig,
) -> WeppcliffData:
    """Impose precipitation QC checks (R ``run_precip_checks``)."""
    df = data.slot_df(5)
    var_vec = df["PRECIP"].to_numpy(dtype=float)
    if config.qcop.upper() in ("B", "E"):
        qc_col = "PRECIP_QC" if "PRECIP_QC" in df.columns else None
        qcr = df[qc_col].to_numpy(dtype=object) if qc_col else np.array(["PASS"] * len(df), dtype=object)
        df["PRECIP_QC"] = return_period_test(var_vec, qcr, config.rp, "RARE")
        data[5] = df
    return data


def return_period_test(
    data_vec: np.ndarray,
    qcr_vec: np.ndarray,
    max_return_period: Union[int, float],
    flag: str,
) -> np.ndarray:
    """Return-period test placeholder (R returns qcr unchanged)."""
    return qcr_vec


def remove_qc_fails(
    data: WeppcliffData,
    config: WeppcliffConfig,
) -> WeppcliffData:
    """Replace QC failures with NA (R ``remove_qc_fails``)."""
    for j, name in enumerate(NM):
        loc = I_LOC[j]
        df = data.slot_df(loc)
        if df.empty:
            continue
        qc_col = f"{name}_QC"
        if name not in df.columns or qc_col not in df.columns:
            continue
        fail = df[qc_col] != "PASS"
        if fail.any():
            df.loc[fail, name] = np.nan
        data[loc] = df
    return data


def quality_check_outputs(
    data: WeppcliffData,
    dly_df: int,
    config: WeppcliffConfig,
) -> WeppcliffData:
    """Check output daily data quality (R ``quality_check_outputs``)."""
    config.verbose("Checking output data quality...")
    return impose_daily_value_limits(data, dly_df, config)


def impose_daily_value_limits(
    data: WeppcliffData,
    dly_df: int,
    config: WeppcliffConfig,
) -> WeppcliffData:
    """Impose physical limitations on daily output (R port)."""
    df = data.slot_df(dly_df).copy()
    for name in NM:
        if name in df.columns:
            upper, lower = DAILY_LIMITS[name]
            df[name] = extreme_limit_replace(
                df[name].to_numpy(dtype=float), upper, lower, np.nan
            )
    data[dly_df] = df
    return data