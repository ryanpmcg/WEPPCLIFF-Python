"""CLIGEN .cli file generation — matches R fixed-width formatting."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from Source.cli_breakpoint_time import first_breakpoint_time
from Source.cli_emission import build_pcp_emit_context, emit_day_breakpoints
from Source.cli_validate import (
    CLIValidationError,
    raise_on_errors,
    validate_day_emission,
    validate_export_bps,
    validate_loop_bounds,
    ymd_to_date_key,
)
from Source.config import LR, get_config
from Source.data_model import WeppcliffData
from Source.paths import create_directory


def _r_fround(x: float, digits: int) -> float:
    """
    Match R's ``round(x, digits)`` / ``fround`` for CLI formatting.

    R rounds via floor/ceil of ``x * 10^digits`` with ties-to-even on the
    integer scaled value (see R's ``src/nmath/fround.c``). Python's built-in
    ``round`` differs on some binary edge cases (e.g. 5.715 → 5.71 vs R 5.72).
    """
    x = float(x)
    if not math.isfinite(x) or x == 0.0:
        return x
    dig = int(math.floor(digits + 0.5))
    sgn = 1.0
    if x < 0.0:
        sgn = -1.0
        x = -x
    # R: if log10(x) + dig > DBL_DIG, no rounding needed
    mant, exp = math.frexp(x)
    logb = exp - 1
    l10x = math.log10(2.0) * (0.5 + logb)
    if l10x + dig > 15:
        return sgn * x
    max10e = 308
    if dig <= max10e:
        pow10 = 10.0 ** dig
        x10 = x * pow10
        i10 = math.floor(x10)
        xd = i10 / pow10
        xu = math.ceil(x10) / pow10
    else:
        e10 = dig - max10e
        p10 = 10.0 ** e10
        pow10 = 10.0 ** max10e
        x10 = (x * pow10) * p10
        i10 = math.floor(x10)
        xd = i10 / pow10 / p10
        xu = math.ceil(x10) / pow10 / p10
    du = xu - x
    dd = x - xd
    # Tie-break: when equidistant, pick xu if floor(x10) is odd (→ even last digit)
    if du < dd or (math.fmod(i10, 2.0) == 1.0 and du == dd):
        return sgn * xu
    return sgn * xd


def _fmt_num(val, width: int, decimals: int, justify: str = "right", na_token: str | None = "NA") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
        if na_token is None:
            val = 0
        else:
            token = na_token
            return token.rjust(width) if justify == "right" else token.ljust(width)
    rounded = _r_fround(float(val), decimals)
    # Avoid Python signed-zero formatting ("-0.0") which R does not emit.
    if rounded == 0.0:
        rounded = 0.0
    if decimals == 0:
        s = str(int(rounded))
    else:
        s = f"{rounded:.{decimals}f}"
    if justify == "right":
        return s.rjust(width)
    return s.ljust(width)


def _fmt_series(
    values: pd.Series,
    width: int,
    decimals: int,
    na_token: str | None = "NA",
) -> pd.Series:
    """Vectorized wrapper around ``_fmt_num`` for export columns."""
    arr = values.to_numpy()
    out = [_fmt_num(v, width, decimals, na_token=na_token) for v in arr]
    return pd.Series(out, index=values.index)


def generate_header_data(data: WeppcliffData) -> str:
    config = get_config()
    if config.do_verb:
        print(LR, "Generating header data...", end="")
    mm_loc = 21 if config.do_id else 13
    meta = data.metadata
    station = meta.get("STATION", "STATION")
    lat = _fmt_num(meta.get("LAT", 0), 8, 4)
    lon = _fmt_num(meta.get("LON", 0), 9, 4)
    elev = _fmt_num(meta.get("ELEV", 0), 6, 2)
    oyr = _fmt_num(meta.get("OBS_YRS"), 3, 0)
    byr = _fmt_num(meta.get("B_YR", 0), 4, 0)
    syr = _fmt_num(meta.get("YRS_SIM", 0), 3, 0)

    if config.cligen_monthly:
        m_max_t = config.cligen_monthly["MAX_TEMP"]
        m_min_t = config.cligen_monthly["MIN_TEMP"]
        m_sorad = config.cligen_monthly["SO_RAD"]
        m_precip = config.cligen_monthly["PRECIP"]
    else:
        mm = data.slot_df(mm_loc)
        m_min_t = mm["MIN_TEMP"].values
        m_max_t = mm["MAX_TEMP"].values
        m_sorad = mm["SO_RAD"].values
        m_precip = mm["PRECIP"].values

    h1 = (
        f" {config.cv}{LR}"
        f"   {config.sm}   {config.bf}   {config.wi}{LR}"
        f"   Station: {station}{LR}"
        f" Latitude Longitude Elevation (m) Obs. Years   Beginning Year  Years Simulated {LR}"
    )
    h2 = f" {lat} {lon}     {elev}         {oyr}           {byr}             {syr}"
    h3 = (
        f"{LR} Observed monthly ave max temperature (C){LR}"
        f" {' '.join(_fmt_num(v, 5, 1) for v in m_max_t)}{LR}"
        f" Observed monthly ave min temperature (C){LR}"
        f" {' '.join(_fmt_num(v, 5, 1) for v in m_min_t)}{LR}"
        f" Observed monthly ave solar radiation (Langleys/day){LR}"
        f" {' '.join(_fmt_num(v, 5, 1) for v in m_sorad)}{LR}"
        f" Observed monthly ave precipitation (mm){LR}"
        f" {' '.join(_fmt_num(v, 5, 1) for v in m_precip)}"
    )
    if int(config.bf) == 0:
        h4 = f"{LR} Day Month Year  Prcp   Dur   TP     IP  T-Max  T-Min   Rad   W-Vel  W-Dir  T-Dew"
        h5 = f"{LR}                 (mm)   (h) (0-1)  (>=1)  (C)    (C)   (L/d)  (m/s)  (Deg)   (C)"
    else:
        # R h.5 has no trailing space; cat(sep=" ") adds one before the first body line.
        h4 = f"{LR}  Day Month Year  Breaks  T-Max  T-Min   Rad   W-Vel  W-Dir  T-Dew"
        h5 = f"{LR}                    (#)    (C)    (C)   (L/d)  (m/s)  (Deg)   (C)"
    return h1 + h2 + h3 + h4 + h5


def generate_export_data(data: WeppcliffData, dly_loc: int, pcp_loc: int):
    """Generate body rows for .cli — mirrors R breakpoint/midnight logic."""
    config = get_config()
    if config.do_verb:
        print(LR, "Generating body data...", end="")

    daily_df = data.slot_df(dly_loc).copy()
    if int(config.sm) == 2:
        daily_df = daily_df.iloc[:1].copy()

    dates = daily_df.iloc[:, 0].astype(str)
    dy = _fmt_series(dates.str[-2:].astype(int), 2, 0)
    mn = _fmt_series(dates.str[4:6].astype(int), 2, 0)
    yr = _fmt_series(dates.str[0:4].astype(int), 4, 0)

    daily_df["BPS"] = daily_df["BPS"].fillna(0)
    daily_ymd = daily_df["YYYYMMDD"].astype(str).to_numpy()

    pcp_df = data.slot_df(pcp_loc).copy()
    pcp_dt = pd.to_datetime(pcp_df["DT_1"])
    pcp_df["DT_1"] = pcp_dt
    pcp_times = (
        pcp_dt.dt.hour * 3600 + pcp_dt.dt.minute * 60 + pcp_dt.dt.second
    ).replace(0, 24 * 3600)
    clock_resets = pcp_times.diff()
    ant = pcp_df["ANT"].to_numpy(dtype=float) if "ANT" in pcp_df.columns else np.zeros(len(pcp_df))
    dur_col = pcp_df["DUR"].to_numpy(dtype=float)
    btw_col = pcp_df["BTW"].to_numpy(dtype=float) if "BTW" in pcp_df.columns else np.zeros(len(pcp_df))

    # R intends to reassign exact-midnight precip rows to the previous day for
    # BPS, but R compares integer YYYYMMDD to character decrease_days with
    # %in%, so the adjustment never applies. Skip it for R .cli parity.
    updated_bps = daily_df["BPS"].to_numpy(dtype=float).copy()
    pos = updated_bps > 0
    updated_bps[pos] = updated_bps[pos] + 1

    pcp_new_days = pcp_df.index[(clock_resets < 0) | (ant >= 1440) | (dur_col >= 1440)]
    pcp_rain_gap = pcp_df.index[btw_col > 0]
    pcp_gap_increase = pcp_rain_gap.difference(pcp_new_days)
    gap_days: list[str] = []
    for idx in pcp_gap_increase:
        dt = pcp_df.loc[idx, "DT_1"]
        if dt.strftime("%H%M%S") == "000000":
            dt = dt - pd.Timedelta(hours=24)
        gap_days.append(dt.strftime("%Y%m%d"))

    increase_rows: list[int] = []
    for day in gap_days:
        increase_rows.extend(np.where(daily_ymd == day)[0].tolist())
    nbrkpt_vals = updated_bps.copy()
    if increase_rows:
        from collections import Counter
        for row_idx, count in Counter(increase_rows).items():
            updated_bps[row_idx] += count

    daily_df["BPS"] = nbrkpt_vals
    daily_df["HEADER_BPS"] = updated_bps

    by_date = pcp_df.groupby(pcp_df["DT_1"].dt.date, sort=True)
    pcpDF = by_date["PRECIP"].sum().reset_index()
    durDF = by_date["DUR"].sum().reset_index()
    intDF = by_date["INT"].max().reset_index()
    day_counts = by_date.size().to_numpy(dtype=int)
    dur_sums = durDF["DUR"].to_numpy(dtype=float)
    iav_vals = pcpDF["PRECIP"].to_numpy(dtype=float) / (dur_sums / 60.0)
    ipk_vals = intDF["INT"].to_numpy(dtype=float) / iav_vals

    ends = np.cumsum(day_counts)
    begins = np.concatenate([[0], ends[:-1]])
    dur_arr = pcp_df["DUR"].to_numpy(dtype=float)
    int_arr = pcp_df["INT"].to_numpy(dtype=float)
    tpk_vals = []
    for begin, end in zip(begins, ends):
        durations = dur_arr[begin:end]
        intensities = int_arr[begin:end]
        if len(intensities):
            max_loc = int(np.argmax(intensities))
            peak_duration = durations[max_loc]
            previous = durations[:max_loc]
            peak_time = (previous.sum() + peak_duration / 2) if len(previous) else peak_duration / 2
            tpk_vals.append(peak_time / durations.sum() if durations.sum() else 0)
        else:
            tpk_vals.append(0)

    cligen = pd.DataFrame({
        "YYYYMMDD": pd.to_datetime(pcpDF.iloc[:, 0]).dt.strftime("%Y%m%d"),
        "AMOUNT": pcpDF["PRECIP"],
        "DURATION": durDF["DUR"] / 60,
        "TP": tpk_vals,
        "IP": ipk_vals,
    })
    daily_merge = pd.merge(daily_df, cligen, on="YYYYMMDD", how="left")
    zeros = daily_merge["AMOUNT"].isna()
    for col in ("AMOUNT", "DURATION", "TP", "IP"):
        daily_merge.loc[zeros, col] = 0
    daily_merge.loc[daily_merge["DURATION"] > 24, "DURATION"] = 24
    daily_merge.loc[daily_merge["TP"] > 1, "TP"] = 1
    daily_merge.loc[daily_merge["TP"] < 0, "TP"] = 0
    daily_merge.loc[daily_merge["IP"] < 1, "IP"] = 1

    nbrkpt = _fmt_series(pd.Series(nbrkpt_vals).astype(int), 3, 0)
    adjbrkpt = _fmt_series(pd.Series(updated_bps).astype(int), 3, 0)
    tmin = _fmt_series(daily_df["MIN_TEMP"], 5, 1)
    tmax = _fmt_series(daily_df["MAX_TEMP"], 5, 1)
    rad = _fmt_series(daily_df["SO_RAD"], 5, 1)
    wvel = _fmt_series(daily_df["W_VEL"], 5, 1)
    wdir = _fmt_series(daily_df["W_DIR"], 5, 1)
    dew = _fmt_series(daily_df["DP_TEMP"], 5, 1)
    amount = _fmt_series(daily_merge["AMOUNT"], 5, 2)
    duration = _fmt_series(daily_merge["DURATION"], 5, 2)
    tp = _fmt_series(daily_merge["TP"], 5, 2)
    ip = _fmt_series(daily_merge["IP"], 5, 2)

    if int(config.bf) == 1:
        df_out = pd.DataFrame({
            "dy": dy, "mn": mn, "yr": yr, "nbrkpt": nbrkpt,
            "tmax": tmax, "tmin": tmin, "rad": rad, "wvel": wvel, "wdir": wdir, "dew": dew,
            "adjbrkpt": adjbrkpt,
        })
    else:
        df_out = pd.DataFrame({
            "dy": dy, "mn": mn, "yr": yr, "amount": amount, "duration": duration,
            "tp": tp, "ip": ip, "tmax": tmax, "tmin": tmin, "rad": rad,
            "wvel": wvel, "wdir": wdir, "dew": dew,
        })
    return [df_out, daily_df]


def create_cli_file(data: WeppcliffData, header: str, export) -> None:
    """Write .cli file — match R line formatting and breakpoint loop indexing."""
    config = get_config()
    if config.do_verb:
        print(LR, "Generating .cli file...", end="")

    pcp_loc = 15 if config.do_id else 5
    df_out = export[0]
    daily_df = export[1]
    pcp_df = data.slot_df(pcp_loc).copy()
    pcp_df["DT_1"] = pd.to_datetime(pcp_df["DT_1"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    current_pcp = 1
    last_pcp = 0
    daily_rows = list(range(len(daily_df)))

    if int(config.sid) > 1 or int(config.sm) == 2:
        storm_locs = pcp_df.index[pcp_df["SID"] == int(config.sid)] if "SID" in pcp_df.columns else pcp_df.index
        if len(storm_locs):
            storm_loc = storm_locs.min()
            end_loc = storm_locs.max()
            start = pd.Timestamp(pcp_df.loc[storm_loc, "DT_1"]).normalize()
            end = pd.Timestamp(pcp_df.loc[end_loc, "DT_1"]).normalize()
            daily_start = int(start.strftime("%Y%m%d"))
            daily_end = int(end.strftime("%Y%m%d"))
            gt_rows = pcp_df.index[pcp_df["DT_1"] >= start.strftime("%Y-%m-%d %H:%M:%S")]
            current_pcp = int(gt_rows.min()) + 1 if len(gt_rows) else 1
            daily_dates = daily_df["YYYYMMDD"].astype(int)
            gt_d = daily_df.index[daily_dates >= daily_start]
            lt_d = daily_df.index[daily_dates <= daily_end]
            daily_rows = list(gt_d.intersection(lt_d))
            config.days_count = (end - start).days

    dt_str = pcp_df["DT_1"].to_numpy()
    clock_h = np.array([s[11:13] for s in dt_str], dtype=object)
    clock_m = np.array([s[14:16] for s in dt_str], dtype=object)
    cligen_h = clock_h.copy()
    cligen_m = clock_m.copy()
    midnight = (clock_h == "00") & (clock_m == "00")
    cligen_h[midnight] = "24"
    pcp_df["CLOCK_H"] = clock_h
    pcp_df["CLOCK_M"] = clock_m
    pcp_df["HOUR"] = cligen_h
    pcp_df["MIN"] = cligen_m

    btw_arr = pcp_df["BTW"].to_numpy(dtype=float) if "BTW" in pcp_df.columns else np.zeros(len(pcp_df))
    dur_arr = pcp_df["DUR"].to_numpy(dtype=float)
    precip_arr = pcp_df["PRECIP"].to_numpy(dtype=float)
    emit_ctx = build_pcp_emit_context(clock_h, clock_m, cligen_h, cligen_m, btw_arr, dur_arr, precip_arr)

    if len(daily_df) < math.ceil(config.days_count):
        raise ValueError(
            "There are missing data. Rerun and perform quality checking [-qc] and fill missing data [-id]."
        )

    create_directory(config.o)
    out_dir = Path(config.o)
    fn = config.fn
    if str(fn).upper() == "STATION":
        fn = data.metadata.get("STATION", "station")
    cli_path = out_dir / f"{fn}.cli"

    lines = [header]
    zero_out = _fmt_num(0, 6, 2)
    violations: list = []
    fmt_time = lambda t: _fmt_num(t, 6, 3)
    fmt_depth = lambda d: _fmt_num(d, 6, 2)

    if int(config.bf) == 1:
        violations.extend(validate_export_bps(export))

    from Source.cli_validate import simulate_day_lines

    for day_pos, i in enumerate(daily_rows):
        row = df_out.iloc[i]
        if int(config.bf) == 0:
            temp_new = (
                f"{LR}  {row['dy']}   {row['mn']}  {row['yr']} {row['amount']} {row['duration']} "
                f"{row['tp']} {row['ip']} {row['tmax']}  {row['tmin']}  {row['rad']}  "
                f"{row['wvel']}  {row['wdir']}  {row['dew']}"
            )
        else:
            # R paste ends at dew; trailing spaces come from cat(sep=" ") on write.
            temp_new = (
                f"{LR}  {row['dy']}   {row['mn']}  {row['yr']}    {row['adjbrkpt']}   "
                f"{row['tmax']}  {row['tmin']}  {row['rad']}  {row['wvel']}  {row['wdir']}  {row['dew']}"
            )
        lines.append(temp_new)

        header_bps = int(daily_df.iloc[i]["HEADER_BPS"]) if int(config.bf) == 1 else 0
        loop_bps = int(daily_df.iloc[i]["BPS"]) if daily_df.iloc[i]["BPS"] == daily_df.iloc[i]["BPS"] else 0

        if int(config.bf) == 1:
            sim, _next = simulate_day_lines(
                day_pos, daily_df, current_pcp, daily_rows, emit_ctx, header_bps
            )
            violations.extend(validate_loop_bounds(sim, emit_ctx.n))

        if int(config.bf) == 1 and loop_bps > 0:
            next_day_opens_continuation = False
            if day_pos + 1 < len(daily_rows):
                next_i = daily_rows[day_pos + 1]
                if int(daily_df.iloc[next_i]["BPS"]) > 0:
                    preview = current_pcp + loop_bps - 1
                    if preview <= emit_ctx.n:
                        pidx = preview - 1
                        next_day_opens_continuation = first_breakpoint_time(
                            clock_h[pidx],
                            clock_m[pidx],
                            cligen_h[pidx],
                            cligen_m[pidx],
                            dur_arr[pidx],
                        ) == 0.0

            day_breakpoints, bp_lines, last_pcp = emit_day_breakpoints(
                loop_bps,
                current_pcp,
                emit_ctx,
                fmt_time=fmt_time,
                fmt_depth=fmt_depth,
                zero_depth_str=zero_out,
                next_day_opens_continuation=next_day_opens_continuation,
            )
            for bp_line in bp_lines:
                lines.append(f"{LR}{bp_line}")

            violations.extend(
                validate_day_emission(
                    day_breakpoints,
                    date_key=ymd_to_date_key(str(daily_df.iloc[i]["YYYYMMDD"])),
                    header_bps=header_bps,
                )
            )
        current_pcp = last_pcp + 1

    # R writes with cat(temp.file) default sep=" ". Each body element starts with
    # lr (newline), so the separator becomes a trailing space on the prior line.
    body = " ".join(lines)

    try:
        raise_on_errors(violations)
    except CLIValidationError:
        if config.do_verb:
            invalid_path = cli_path.with_suffix(".cli.invalid")
            invalid_path.write_text(body, encoding="utf-8", newline="\n")
        raise

    cli_path.write_text(body, encoding="utf-8", newline="\n")