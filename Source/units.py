"""Unit conversion — port of R ``convert_units``."""

from __future__ import annotations

from typing import List

import numpy as np

from .config import WeppcliffConfig
from .data_model import WeppcliffData

VAR_LIST = [
    "PRECIP", "AIR_TEMP", "MAX_TEMP", "MIN_TEMP", "SO_RAD",
    "W_VEL", "DP_TEMP", "ELEV",
]


def _f_to_c(series):
    return (series - 32) * 5 / 9


def _vars_to_convert(config: WeppcliffConfig, exceptions: List[str]) -> List[str]:
    exceptions_upper = [v.upper() for v in exceptions]
    if "SO_RAD" in exceptions_upper:
        base = [v for v in exceptions_upper if v != "SO_RAD"]
    else:
        base = list(exceptions_upper) + ["SO_RAD"]

    if config.u.upper() == "E":
        pattern_excl = set(exceptions_upper)
        return [v for v in VAR_LIST if v.upper() not in pattern_excl]
    if config.u.upper() == "M":
        return base
    return []


def convert_units(
    data: WeppcliffData,
    config: WeppcliffConfig,
) -> WeppcliffData:
    """Convert English units to metric (R ``convert_units``)."""
    config.verbose("Converting units...")
    exceptions: List[str] = config.parse_unit_vars() or []
    vars_to_convert = _vars_to_convert(config, exceptions)

    meta = data.metadata
    if meta.get("ELEV") is None or (isinstance(meta["ELEV"], float) and np.isnan(meta["ELEV"])):
        meta["ELEV"] = "0"
    if str(meta.get("ELEV")) == "-99999":
        meta["ELEV"] = "0"

    pcp = data.slot_df(1)
    alt = data.slot_df(2)
    daily = data.slot_df(3)

    for var in vars_to_convert:
        if var == "PRECIP" and "PRECIP" in pcp.columns:
            pcp["PRECIP"] = pcp["PRECIP"] * 25.4
        elif var == "MAX_TEMP" and "MAX_TEMP" in daily.columns:
            daily["MAX_TEMP"] = _f_to_c(daily["MAX_TEMP"])
        elif var == "MIN_TEMP" and "MIN_TEMP" in daily.columns:
            daily["MIN_TEMP"] = _f_to_c(daily["MIN_TEMP"])
        elif var == "AIR_TEMP" and "AIR_TEMP" in alt.columns:
            alt["AIR_TEMP"] = _f_to_c(alt["AIR_TEMP"])
        elif var == "DP_TEMP" and "DP_TEMP" in daily.columns:
            daily["DP_TEMP"] = _f_to_c(daily["DP_TEMP"])
        elif var == "SO_RAD" and "SO_RAD" in daily.columns and "DUR" in daily.columns:
            daily["SO_RAD"] = daily["SO_RAD"] * daily["DUR"] * 60 / 41840
        elif var == "W_VEL" and "W_VEL" in daily.columns:
            daily["W_VEL"] = daily["W_VEL"] * 0.44704
        elif var == "ELEV" and meta.get("ELEV") is not None and str(meta["ELEV"]) != "":
            meta["ELEV"] = float(meta["ELEV"]) * 0.3048

    data[1] = pcp
    data[3] = daily
    data[2] = alt

    if meta.get("STATION") is None or (isinstance(meta["STATION"], float) and np.isnan(meta["STATION"])):
        meta["STATION"] = "STATION"
    if meta.get("LAT") is None or (isinstance(meta["LAT"], float) and np.isnan(meta["LAT"])):
        meta["LAT"] = "0"
    if meta.get("LON") is None or (isinstance(meta["LON"], float) and np.isnan(meta["LON"])):
        meta["LON"] = "0"
    if meta.get("ELEV") is None or (isinstance(meta["ELEV"], float) and np.isnan(meta["ELEV"])):
        meta["ELEV"] = "0"

    data[4] = meta
    return data