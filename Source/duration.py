"""Duration analysis — port of R create_duration_data and midnight breakpoints."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from .config import WeppcliffConfig
from .data_model import WeppcliffData
from .missing_precip import snapshot_observed_precip_days


def create_duration_data(
    data: WeppcliffData,
    x: Sequence[int],
    config: WeppcliffConfig,
) -> WeppcliffData:
    """Calculate duration columns and filter precipitation breakpoints (R port)."""
    config.verbose("Analyzing data intervals...")
    config.ensure_datetime_formats()
    if 1 in data:
        data.observed_precip_ymd = snapshot_observed_precip_days(data.slot_df(1))

    dt_list = ["DT_1", "DT_2", "DT_3"]
    dtf_list = [config.dtf1, config.dtf2, config.dtf3]

    for j in x:
        df = data.slot_df(j).copy()
        dt_col = dt_list[j - 1]
        dtf = dtf_list[j - 1]

        if df[dt_col].dropna().empty:
            raise ValueError(
                "Datetime data may not be handled correctly. "
                "Please check the input file(s) and arguments."
            )

        dt_series = (
            df[dt_col]
            .astype(str)
            .str.replace("T", " ", regex=False)
            .str.replace("Z", "", regex=False)
            .str.strip()
        )
        parsed = pd.to_datetime(dt_series, format=dtf, errors="coerce")
        if parsed.isna().all():
            parsed = pd.to_datetime(dt_series, errors="coerce")
        df[dt_col] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S")
        df = df.dropna(subset=[dt_col]).reset_index(drop=True)

        if j == 1 and str(config.pi).upper() != "F":
            precip_interval = float(config.pi)
            dt_parsed = pd.to_datetime(df[dt_col])
            new_dates = dt_parsed - pd.to_timedelta(precip_interval, unit="m")
            existing = set(dt_parsed)
            add_dates = [d for d in new_dates if d not in existing]
            if add_dates:
                new_rows = pd.DataFrame({
                    "DT_1": [d.strftime("%Y-%m-%d %H:%M:%S") for d in add_dates],
                    "PRECIP": [0.0] * len(add_dates),
                })
                df = pd.concat([df, new_rows], ignore_index=True)

        df = df.sort_values(dt_col).reset_index(drop=True)
        dt_parsed = pd.to_datetime(df[dt_col])
        dif = dt_parsed.diff().dt.total_seconds() / 60.0
        if len(dif) == 0:
            df["DUR"] = pd.Series(dtype=float)
        elif len(dif) == 1:
            df["DUR"] = 0.0
        else:
            dif = dif.copy()
            dif.iloc[0] = dif.iloc[1]
            df["DUR"] = dif.fillna(0).values
        data[j] = df

    pcp = data.slot_df(1)
    precip = pd.to_numeric(pcp["PRECIP"], errors="coerce")
    p_rows = pcp.index[precip.notna() & (precip != 0)]
    z_rows = pcp.index[precip == 0]
    s_rows = p_rows - 1
    b_rows = z_rows.intersection(s_rows)
    keep = p_rows.union(b_rows)
    data[1] = pcp.loc[keep].sort_values("DT_1").reset_index(drop=True)
    data[1] = modify_midnight_breakpoints(data.slot_df(1), config.dtf1, config)
    return data


def modify_midnight_breakpoints(
    df: pd.DataFrame,
    dtf: str,
    config: WeppcliffConfig,
) -> pd.DataFrame:
    """Split precipitation breakpoints that cross midnight (R port)."""
    out = df.copy()
    dt_parsed = pd.to_datetime(out["DT_1"], format=dtf, errors="coerce")

    tfm = (
        dt_parsed.dt.hour * 60
        + dt_parsed.dt.minute
        + dt_parsed.dt.second / 60.0
    )
    dif_vec = tfm - out["DUR"]
    crosses = dif_vec[(dif_vec < 0) & (out["PRECIP"] > 0)].index

    if len(crosses) == 0:
        return out.sort_values("DT_1").reset_index(drop=True)

    pcp_frac = -dif_vec.loc[crosses] / out.loc[crosses, "DUR"]
    pcp_new = pcp_frac * out.loc[crosses, "PRECIP"]
    dt_new = dt_parsed.loc[crosses] - pd.to_timedelta(tfm.loc[crosses], unit="m")
    # Pre-midnight duration (R dur.new). Post-midnight row keeps TFM minutes.
    # R: df$DUR[crosses] = df$DUR[crosses] - dur.new  → equals TFM.
    dur_before_midnight = out.loc[crosses, "DUR"] - tfm.loc[crosses]

    out.loc[crosses, "PRECIP"] = out.loc[crosses, "PRECIP"] - pcp_new
    out.loc[crosses, "DUR"] = tfm.loc[crosses]

    df_new = pd.DataFrame({
        "DT_1": dt_new.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "PRECIP": pcp_new.values,
        "DUR": dur_before_midnight.values,
    })
    out = pd.concat([out, df_new], ignore_index=True)
    return out.sort_values("DT_1").reset_index(drop=True)