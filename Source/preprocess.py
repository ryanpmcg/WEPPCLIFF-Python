"""Input preprocessing — port of R preprocess/store/convert functions."""

from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np
import pandas as pd

from .config import WeppcliffConfig
from .data_model import WeppcliffData
from .utils import convert_df_type, replace_sentinels_columns

REQUIRED_VARS: List[str] = [
    "STATION", "LAT", "LON", "ELEV", "DT_1", "PRECIP", "DT_2", "AIR_TEMP",
    "REL_HUM", "DT_3", "MAX_TEMP", "MIN_TEMP", "SO_RAD", "W_VEL", "W_DIR",
    "DP_TEMP",
]

LONG_VARS: List[str] = [
    "DT_1", "PRECIP", "DT_2", "AIR_TEMP", "REL_HUM", "DT_3", "MAX_TEMP",
    "MIN_TEMP", "SO_RAD", "W_VEL", "W_DIR", "DP_TEMP",
]

SHORT_VARS: List[str] = ["STATION", "LAT", "LON", "ELEV", "OBS_YRS", "B_YR", "YRS_SIM"]


def preprocess_file(
    file: pd.DataFrame,
    config: WeppcliffConfig,
) -> pd.DataFrame:
    """Add missing variable columns as NA (R ``preprocess_file``)."""
    config.verbose("Preprocessing data... ")
    out = file.copy()
    for var in REQUIRED_VARS:
        if var not in out.columns:
            out[var] = np.nan
    return out


def store_data(
    file: Union[Sequence[pd.DataFrame], List[pd.DataFrame], pd.DataFrame],
    config: WeppcliffConfig,
) -> WeppcliffData:
    """Store input file columns in the primary data structure (R ``store_data``)."""
    config.verbose("Storing in memory...")
    if isinstance(file, pd.DataFrame):
        frame = file
    else:
        frame = pd.DataFrame(file[0])

    metadata: dict = {}
    for var in SHORT_VARS:
        if var not in frame.columns or frame[var].isna().all():
            metadata[var] = np.nan
        else:
            metadata[var] = frame[var].iloc[0]

    def _series(name: str) -> pd.Series:
        if name not in frame.columns:
            return pd.Series(dtype=float)
        return frame[name]

    data = WeppcliffData()
    data[1] = pd.DataFrame({"DT_1": _series("DT_1"), "PRECIP": _series("PRECIP")})
    data[2] = pd.DataFrame({
        "DT_2": _series("DT_2"),
        "AIR_TEMP": _series("AIR_TEMP"),
        "REL_HUM": _series("REL_HUM"),
    })
    data[3] = pd.DataFrame({
        "DT_3": _series("DT_3"),
        "MIN_TEMP": _series("MIN_TEMP"),
        "MAX_TEMP": _series("MAX_TEMP"),
        "SO_RAD": _series("SO_RAD"),
        "W_VEL": _series("W_VEL"),
        "W_DIR": _series("W_DIR"),
        "DP_TEMP": _series("DP_TEMP"),
    })
    data[4] = metadata
    return data


def convert_data(
    data: WeppcliffData,
    config: WeppcliffConfig,
) -> WeppcliffData:
    """Convert groups 1–3 to numeric dataframes (R ``convert_data``)."""
    config.verbose("Converting data format...")
    for idx in (1, 2, 3):
        df = data.slot_df(idx)
        if df.shape[1] > 1:
            df = convert_df_type(df, list(range(2, df.shape[1] + 1)), "ftn")
        data[idx] = df
    return data


def remove_missing_datetime_rows(
    data: WeppcliffData,
    config: WeppcliffConfig,
) -> WeppcliffData:
    """Remove rows with missing datetime in groups 1–3."""
    for idx in (1, 2, 3):
        df = data.slot_df(idx)
        if len(df.columns) > 0:
            dt_col = df.columns[0]
            data[idx] = df.loc[df[dt_col].notna()].reset_index(drop=True)
    return data


def preserve_missing_data(
    data: WeppcliffData,
    group: Sequence[int],
    config: WeppcliffConfig,
) -> WeppcliffData:
    """Replace sentinel / missing values with NA in selected groups."""
    for idx in group:
        df = data.slot_df(idx)
        if df.shape[1] > 1:
            data[idx] = replace_sentinels_columns(
                df.copy(),
                cols=list(range(2, df.shape[1] + 1)),
                vals=-99999,
                replace=np.nan,
            )
    return data