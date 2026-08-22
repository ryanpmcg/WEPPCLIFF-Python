"""Gap filling and cluster imputation (PyMICE FCS or mean-fill fallback)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from Source.config import LR, get_config
from Source.data_model import WeppcliffData
from Source.missing_precip import detect_missing_precip_months
from Source.phigi.dates import as_yyyymmdd_strings, coerce_yyyymmdd_series
from Source.time_bounds import parse_user_datetime

try:
    from Source.weppcliff_impute import impute_by_cluster as _pymice_impute_by_cluster
    from Source.weppcliff_impute import impute_by_df as _pymice_impute_by_df
    from Source.weppcliff_impute import impute_subdaily_timeseries as _pymice_impute_subdaily
    from Source.weppcliff_impute.cluster import select_cluster_granularity
    from Source.weppcliff_impute.diagnostics import format_report as _format_impute_report
    from Source.weppcliff_impute.ei_validate import EIValidationResult
    from Source.weppcliff_impute.overlap import trim_overlap_durations
    from Source.weppcliff_impute.setup import dataframe_to_matrix

    _PYMICE_AVAILABLE = True
except ImportError:
    _PYMICE_AVAILABLE = False
    EIValidationResult = None  # type: ignore

    def trim_overlap_durations(df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[misc]
        return df


def _max_iter(fil_df: pd.DataFrame, config) -> int:
    complete = fil_df.dropna(how="all").shape[0]
    total = len(fil_df)
    max_iter = max(20, int(np.ceil(100 - 100 * complete / max(total, 1))))
    return min(int(config.io), max_iter)


_SKIP_IMPUTE_COLS = {"CLUSTER", "YYYYMMDD", "YYYYMM", "DT_1", "COUNT"}
_WX_EVENT_COLS = [
    "CLUSTER",
    "SO_RAD",
    "MAX_TEMP",
    "MIN_TEMP",
    "DP_TEMP",
    "W_VEL",
    "W_DIR",
    "EVENT",
]
_PCP_DAILY_COLS = ["CLUSTER", "BPS", "ELEN", "EMAX", "EMIN", "PRECIP"]
_PRECIP_TS_NUMERIC = (
    "DUR",
    "PRECIP",
    "CLUSTER",
    "HOUR",
    "MINUTE",
    "SECOND",
    "ANT",
    "CUM",
    "BPS",
    "ELEN",
    "EMAX",
    "EMIN",
)


def coerce_precip_ts_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Cast breakpoint amount/time columns to float so mice will impute them."""
    out = df.copy()
    for col in _PRECIP_TS_NUMERIC:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _coerce_impute_frame(fil_df: pd.DataFrame) -> pd.DataFrame:
    """Recover object-NA numeric columns (empty-event frames are all-object)."""
    out = fil_df.copy()
    for col in out.columns:
        if col in _SKIP_IMPUTE_COLS:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            continue
        if str(out[col].dtype) == "category":
            continue
        coerced = pd.to_numeric(out[col], errors="coerce")
        if coerced.notna().any() or out[col].isna().all():
            out[col] = coerced
    return out


def aligned_empty_storm_bps(
    imputed_bps: pd.Series,
    empty_days: list[int],
) -> pd.Series:
    """Integer BPS for new EVENT days from post-impute values (R).

    Pre-impute BPS is NA, which forced 1 placeholder row while the daily
    table stored the rounded imputed BPS. ``i_pcp`` then sliced the wrong
    rows and stole observed mass. Empty days get at least one row; the
    returned series is what both the daily table and placeholder frames use.
    """
    out = pd.to_numeric(imputed_bps, errors="coerce").clip(lower=0).round()
    for ed in empty_days:
        if ed >= len(out):
            continue
        val = out.iloc[ed]
        n = int(val) if pd.notna(val) else 1
        out.iloc[ed] = float(max(1, n))
    return out


def empty_breakpoint_frame(
    template: pd.DataFrame,
    n_rows: int,
    *,
    yyyymmdd: str,
    cluster,
) -> pd.DataFrame:
    """NA breakpoint rows for a new EVENT day, keeping observed column dtypes.

    ``DataFrame(index=..., columns=...)`` is all-object, so concat promotes
    PRECIP/DUR to object and mice skips them as non-numeric.
    """
    nrows = max(1, int(n_rows))
    data: dict[str, pd.Series] = {}
    idx = range(nrows)
    for col in template.columns:
        dt = template[col].dtype
        if pd.api.types.is_datetime64_any_dtype(dt):
            data[col] = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns]")
        elif pd.api.types.is_numeric_dtype(dt):
            data[col] = pd.Series(np.nan, index=idx, dtype="float64")
        else:
            data[col] = pd.Series([pd.NA] * nrows, dtype="object")
    frame = pd.DataFrame(data)
    frame["DT_1"] = pd.to_datetime(str(yyyymmdd), format="%Y%m%d", errors="coerce")
    if "YYYYMMDD" in frame.columns:
        frame["YYYYMMDD"] = str(yyyymmdd)
    if "CLUSTER" in frame.columns:
        frame["CLUSTER"] = cluster
    return frame


def _impute_value_columns(fil_df: pd.DataFrame) -> list[str]:
    """Numeric columns mice/mean-fill may touch; never dates or cluster ids."""
    cols: list[str] = []
    for col in fil_df.columns:
        if col in _SKIP_IMPUTE_COLS:
            continue
        if fil_df[col].dtype == object or str(fil_df[col].dtype) == "category":
            continue
        if not pd.api.types.is_numeric_dtype(fil_df[col]):
            continue
        cols.append(col)
    return cols


def _mean_impute_by_cluster(fil_df: pd.DataFrame, imp_method: str) -> pd.DataFrame:
    """Legacy mean-fill within CLUSTER (used when ``-pm f`` or PyMICE unavailable)."""
    del imp_method
    out = fil_df.copy()
    value_cols = _impute_value_columns(out)

    for cluster in out["CLUSTER"].dropna().unique():
        locs = out.index[out["CLUSTER"] == cluster]
        subset = out.loc[locs, value_cols].copy()
        for col in value_cols:
            col_mean = subset[col].mean()
            if pd.isna(col_mean):
                col_mean = out[col].mean()
            if pd.isna(col_mean):
                col_mean = 0
            out.loc[locs, col] = subset[col].fillna(col_mean)

    for col in value_cols:
        global_mean = out[col].mean()
        if pd.isna(global_mean):
            global_mean = 0
        out[col] = out[col].fillna(global_mean)
    return out


def _mean_impute_by_df(fil_df: pd.DataFrame, imp_method: str) -> pd.DataFrame:
    del imp_method
    out = fil_df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            continue
        mean = out[col].mean()
        if pd.isna(mean):
            mean = 0
        out[col] = out[col].fillna(mean)
    return out


def _station_coords(config, data: WeppcliffData | None = None) -> None:
    """Populate lat/lon on config from station metadata when available."""
    if data is None:
        return
    meta = data.metadata
    for key, attr in (("LAT", "lat"), ("LON", "lon")):
        raw = meta.get(key)
        if raw is None or raw == "":
            continue
        try:
            setattr(config, attr, float(raw))
        except (TypeError, ValueError):
            continue


def impute_by_cluster(
    fil_df: pd.DataFrame,
    imp_method: str,
    *,
    data: WeppcliffData | None = None,
) -> pd.DataFrame:
    """Fill NA via PyMICE FCS within each CLUSTER, or cluster means when disabled."""
    config = get_config()
    _station_coords(config, data)
    fil_df = _coerce_impute_frame(fil_df)
    use_pymice = str(config.pm).upper() == "T" and _PYMICE_AVAILABLE
    if use_pymice:
        return _pymice_impute_by_cluster(
            fil_df,
            imp_method,
            config,
            date_index=fil_df.index,
            dates_yyyymmdd=config.fill_dates,
        )
    return _mean_impute_by_cluster(fil_df, imp_method)


def impute_by_df(fil_df: pd.DataFrame, imp_method: str) -> pd.DataFrame:
    """Fill NA via PyMICE FCS across the dataframe, or column means when disabled."""
    config = get_config()
    use_pymice = str(config.pm).upper() == "T" and _PYMICE_AVAILABLE
    if use_pymice:
        return _pymice_impute_by_df(fil_df, imp_method, config)
    return _mean_impute_by_df(fil_df, imp_method)


def create_cluster_col(in_df: pd.DataFrame, dt_col_name: str, dt_format: str) -> pd.DataFrame:
    config = get_config()
    df = in_df.copy()
    dt = pd.to_datetime(df[dt_col_name], format=dt_format, errors="coerce")
    if dt.notna().sum() < 50:
        raise ValueError(
            "Insufficient Data: at least 50 complete observations of all variables must be present for gap filling."
        )
    start = dt.min().date()
    end = dt.max().date()
    if (end - start).days < 366:
        raise ValueError(
            "Insufficient Data: at least one year of mostly complete data must be present for gap filling."
        )

    if str(config.sdt).upper() != "START":
        start = parse_user_datetime(config.sdt, config.dtf1).date()
    if str(config.edt).upper() != "END":
        end = parse_user_datetime(config.edt, config.dtf1).date()
    if (end - start).days < 366:
        raise ValueError(
            "Insufficient Data: at least one year of mostly complete data must be present for gap filling."
        )

    fill_seq = pd.date_range(start, end, freq="D").strftime("%Y%m%d").tolist()

    if str(config.pm).upper() == "T" and _PYMICE_AVAILABLE:
        numeric_cols = [
            c
            for c in df.columns
            if c != dt_col_name and pd.api.types.is_numeric_dtype(df[c])
        ]
        value_matrix = None
        if numeric_cols:
            if dt_format != "%Y%m%d":
                df_dates = pd.to_datetime(df[dt_col_name], format=dt_format, errors="coerce").dt.strftime(
                    "%Y%m%d"
                )
            else:
                df_dates = coerce_yyyymmdd_series(df[dt_col_name])
            daily_src = df.assign(_ymd=df_dates)
            for col in numeric_cols:
                daily_src[col] = pd.to_numeric(daily_src[col], errors="coerce")
            daily = (
                daily_src.groupby("_ymd", as_index=False)[numeric_cols]
                .mean()
                .rename(columns={"_ymd": "YYYYMMDD"})
            )
            merged_tmp = pd.DataFrame({"YYYYMMDD": fill_seq}).merge(
                daily,
                on="YYYYMMDD",
                how="left",
            )
            sub = merged_tmp[numeric_cols].apply(pd.to_numeric, errors="coerce")
            value_matrix, _ = dataframe_to_matrix(sub, list(sub.columns))
        gran, cluster_labels = select_cluster_granularity(
            fill_seq,
            value_matrix,
            lat=float(config.lat),
        )
        config.impute_granularity = gran
        out_df = pd.DataFrame({"YYYYMMDD": fill_seq, "CLUSTER": cluster_labels})
    else:
        def _cluster_vectors(dys: list[str]):
            mns = [int(d[4:6]) for d in dys]
            qtr = [int(d[6:8]) for d in dys]
            qtr = [1 if q <= 7 else 4 if q >= 24 else 3 if q >= 16 else 2 if q >= 8 else q for q in qtr]
            hmn = [2 * m for m in mns]
            hmn = [h - 1 if q <= 2 else h for h, q in zip(hmn, qtr)]
            qmn = [2 * h for h in hmn]
            qmn = [q - 1 if qt == 1 or qt == 3 else q for q, qt in zip(qmn, qtr)]
            seasons = [((m + 2) % 12) // 3 + 1 for m in mns]
            return seasons, mns, hmn, qmn

        data_seq = fill_seq
        seasons, mns, hmn, qmn = _cluster_vectors(data_seq)
        min_len_sns = min(len([x for x in seasons if x == u]) for u in set(seasons))
        min_len_mns = min(len([x for x in mns if x == u]) for u in set(mns))
        min_len_hmn = min(len([x for x in hmn if x == u]) for u in set(hmn))
        min_len_qmn = min(len([x for x in qmn if x == u]) for u in set(qmn))

        seasons_f, mns_f, hmn_f, qmn_f = _cluster_vectors(fill_seq)
        out_df = pd.DataFrame({"YYYYMMDD": fill_seq, "CLUSTER": seasons_f})
        if min_len_mns >= 30:
            out_df = pd.DataFrame({"YYYYMMDD": fill_seq, "CLUSTER": mns_f})
        if min_len_hmn >= 30:
            out_df = pd.DataFrame({"YYYYMMDD": fill_seq, "CLUSTER": hmn_f})
        if min_len_qmn >= 30:
            out_df = pd.DataFrame({"YYYYMMDD": fill_seq, "CLUSTER": qmn_f})
        if min_len_sns < 30:
            out_df = pd.DataFrame({"YYYYMMDD": fill_seq, "CLUSTER": seasons_f})

    ymd_col = "YYYYMMDD"
    if dt_format != "%Y%m%d":
        df[ymd_col] = pd.to_datetime(df[dt_col_name], format=dt_format, errors="coerce").dt.strftime("%Y%m%d")
    else:
        df[ymd_col] = coerce_yyyymmdd_series(df[dt_col_name])

    cluster_lut = out_df[["YYYYMMDD", "CLUSTER"]].astype({"YYYYMMDD": str})
    if df[ymd_col].duplicated(keep=False).any():
        merge_df = df.merge(cluster_lut, on=ymd_col, how="left")
    else:
        merge_df = cluster_lut.merge(df, on=ymd_col, how="left")
    return merge_df


def find_missing_precip(in_df: pd.DataFrame) -> pd.DataFrame:
    """Set EVENT to NA on gauge-gap months; leave climatological dry months as 0.

    Gap vs dry uses month-of-year Poisson-Gamma tests, not CLUSTER counts.
    """
    out = in_df.copy()
    ymd = coerce_yyyymmdd_series(out["YYYYMMDD"])
    out["YYYYMMDD"] = ymd
    out["YYYYMM"] = ymd.str[:6].astype(int)
    out["EVENT"] = pd.to_numeric(out["EVENT"], errors="coerce")
    scan = detect_missing_precip_months(ymd.to_numpy(), out["EVENT"].to_numpy())
    if scan.missing_months:
        out.loc[out["YYYYMM"].isin(scan.missing_months), "EVENT"] = np.nan
    return out


def split_precip_by_daily_bps(
    pcp_source: pd.DataFrame,
    bps: pd.Series,
) -> list[pd.DataFrame]:
    """Split the leading ``sum(bps)`` precip rows into per-wet-day frames.

    Matches R ``split(pcp.ts.df[1:sum(BPS),], split.fact)`` where ``split.fact``
    repeats each day's COUNT ``BPS`` times. Days with BPS<=0 are skipped.
    """
    chunks: list[pd.DataFrame] = []
    start = 0
    bps_int = pd.to_numeric(bps, errors="coerce").fillna(0).astype(int)
    n_src = len(pcp_source)
    for n in bps_int.tolist():
        if n <= 0:
            continue
        end = min(start + n, n_src)
        chunks.append(pcp_source.iloc[start:end].copy().reset_index(drop=True))
        start = end
        if start >= n_src:
            break
    return chunks


def binarize_filled_event(filled: pd.Series, observed: pd.Series) -> pd.Series:
    """Keep observed 0/1 EVENT; map imputed values to {0,1} at 0.5."""
    obs = pd.to_numeric(observed, errors="coerce")
    fill = pd.to_numeric(filled, errors="coerce")
    keep = obs.isin([0.0, 1.0])
    imputed = (fill >= 0.5).astype(float)
    out = imputed.where(~keep, obs)
    return out.astype(float)


def assemble_event_table(
    dly_var_df: pd.DataFrame,
    dly_pcp_df: pd.DataFrame,
    daily_out: pd.DataFrame,
    event_mask: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """EVENT==1 subset with reset indices so dates align with wet-day rows."""
    idx = dly_var_df.index[event_mask]
    ymd_col = "YYYYMMDD" if "YYYYMMDD" in daily_out.columns else daily_out.columns[0]
    eve_var = dly_var_df.loc[idx].reset_index(drop=True)
    pcp_cols = [c for c in ("BPS", "ELEN", "PRECIP") if c in dly_pcp_df.columns]
    eve_pcp = dly_pcp_df.loc[idx, pcp_cols].copy().reset_index(drop=True)
    con_cols = [c for c in ("ELEN", "EMAX", "EMIN") if c in dly_pcp_df.columns]
    eve_con = dly_pcp_df.loc[idx, con_cols].copy().reset_index(drop=True)
    dates = coerce_yyyymmdd_series(daily_out.loc[idx, ymd_col]).reset_index(drop=True)
    clusters = daily_out.loc[idx, "CLUSTER"].reset_index(drop=True)
    if "BPS" in eve_pcp.columns:
        eve_pcp.loc[pd.to_numeric(eve_pcp["BPS"], errors="coerce") == 0, "BPS"] = np.nan
    eve = pd.concat([eve_var, eve_pcp], axis=1)
    eve["YYYYMMDD"] = dates.to_numpy()
    if "CLUSTER" not in eve.columns:
        eve.insert(0, "CLUSTER", pd.to_numeric(clusters, errors="coerce").to_numpy())
    return eve, dates, clusters, eve_con


def event_predictor_complete_mask(eve: pd.DataFrame) -> pd.Series:
    """Rows complete in weather/EVENT; BPS/ELEN/PRECIP may be NA (R ``[,1:9]``)."""
    skip = {"BPS", "ELEN", "PRECIP", "YYYYMMDD", "COUNT", "EMAX", "EMIN"}
    cols = [c for c in eve.columns if c not in skip]
    if not cols:
        return pd.Series(True, index=eve.index)
    return eve[cols].notna().all(axis=1)


def attach_event_dates(imp_df: pd.DataFrame, event_dates: pd.Series) -> pd.DataFrame:
    """Write aligned YYYYMMDD onto the imputed event table (no duplicate insert)."""
    out = imp_df.copy().reset_index(drop=True)
    dates = coerce_yyyymmdd_series(event_dates).reset_index(drop=True)
    if len(out) != len(dates):
        raise ValueError(
            f"imputed event table has {len(out)} rows but {len(dates)} event dates"
        )
    out["YYYYMMDD"] = dates.to_numpy()
    return out


def _require_columns(df: pd.DataFrame, cols: list[str], label: str) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")
    return df[cols].copy()


def assign_existing_event_timeseries(
    event_chunks: list[pd.DataFrame],
    n_events: int,
    empty_days: list[int],
) -> dict[int, pd.DataFrame]:
    """Map i-th observed wet-day series onto i-th non-empty event row (R)."""
    empty = set(int(x) for x in empty_days)
    pcp_list = [i for i in range(n_events) if i not in empty]
    fil: dict[int, pd.DataFrame] = {}
    for i, pl in enumerate(pcp_list):
        if i < len(event_chunks):
            fil[pl] = event_chunks[i]
    return fil


def combine_duplicate_precip_times(imp_ts_df: pd.DataFrame) -> pd.DataFrame:
    """Merge equal DT_1 rows: sum PRECIP, keep the first row's other fields."""
    out = imp_ts_df.copy()
    if out.empty or "DT_1" not in out.columns:
        return out
    out["DT_1"] = pd.to_datetime(out["DT_1"])
    out = out.sort_values("DT_1").reset_index(drop=True)
    rows: list[pd.Series] = []
    for _, grp in out.groupby("DT_1", sort=True):
        row = grp.iloc[0].copy()
        row["PRECIP"] = float(pd.to_numeric(grp["PRECIP"], errors="coerce").sum())
        if "DUR" in grp.columns:
            row["DUR"] = float(pd.to_numeric(grp["DUR"], errors="coerce").max())
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def fill_data(
    data: WeppcliffData,
    pcp_ts_in: int,
    dly_ts_in: int,
    dly_ts_out: int,
    pcp_ts_out: int,
) -> WeppcliffData:
    """Dispatch gap fill to PHIGI (L6) or legacy staged MICE."""
    config = get_config()
    if getattr(config, "do_phigi", str(config.gf).lower() == "phigi"):
        return fill_data_phigi(data, pcp_ts_in, dly_ts_in, dly_ts_out, pcp_ts_out)
    return fill_data_legacy(data, pcp_ts_in, dly_ts_in, dly_ts_out, pcp_ts_out)


def fill_data_phigi(
    data: WeppcliffData,
    pcp_ts_in: int,
    dly_ts_in: int,
    dly_ts_out: int,
    pcp_ts_out: int,
) -> WeppcliffData:
    """PHIGI hierarchical gap fill (L0–L5)."""
    return _fill_data_impl(
        data, pcp_ts_in, dly_ts_in, dly_ts_out, pcp_ts_out, use_phigi=True
    )


def fill_data_legacy(
    data: WeppcliffData,
    pcp_ts_in: int,
    dly_ts_in: int,
    dly_ts_out: int,
    pcp_ts_out: int,
) -> WeppcliffData:
    """Legacy staged PyMICE / mean-fill gap fill (-gf legacy)."""
    return _fill_data_impl(
        data, pcp_ts_in, dly_ts_in, dly_ts_out, pcp_ts_out, use_phigi=False
    )


def _fill_data_impl(
    data: WeppcliffData,
    pcp_ts_in: int,
    dly_ts_in: int,
    dly_ts_out: int,
    pcp_ts_out: int,
    *,
    use_phigi: bool,
) -> WeppcliffData:
    config = get_config()
    _station_coords(config, data)
    if str(config.verb).upper() == "T":
        print(LR, "Preparing to impute data...", end="")

    pcp_in = data.slot_df(pcp_ts_in)
    dly_in = data.slot_df(dly_ts_in)
    pcp_in = pcp_in[pcp_in["DT_1"].notna()].reset_index(drop=True)
    dly_in = dly_in[dly_in["YYYYMMDD"].notna()].reset_index(drop=True)

    data[dly_ts_out] = create_cluster_col(dly_in, "YYYYMMDD", "%Y%m%d")
    day_clusters = (
        data.slot_df(dly_ts_out)[["YYYYMMDD", "CLUSTER"]]
        .drop_duplicates("YYYYMMDD")
        .assign(YYYYMMDD=lambda d: coerce_yyyymmdd_series(d["YYYYMMDD"]))
    )
    pcp_ymd = pd.to_datetime(pcp_in["DT_1"], errors="coerce").dt.strftime("%Y%m%d")
    pcp_cluster_body = pcp_in.copy()
    pcp_cluster_body["YYYYMMDD"] = pcp_ymd
    pcp_cluster_body = pcp_cluster_body.merge(day_clusters, on="YYYYMMDD", how="left")
    pcp_cluster = pcp_cluster_body.dropna(subset=["DT_1", "CLUSTER"]).reset_index(drop=True)
    data[pcp_ts_out] = pcp_cluster

    daily_clustered = data.slot_df(dly_ts_out)
    dly_var_df = _require_columns(
        daily_clustered,
        ["CLUSTER", "SO_RAD", "MAX_TEMP", "MIN_TEMP", "DP_TEMP", "W_VEL", "W_DIR"],
        "clustered daily weather table",
    )
    dly_pcp_df = _require_columns(daily_clustered, _PCP_DAILY_COLS, "clustered daily precip table")

    dly_var_df["EVENT"] = 0.0
    precip_num = pd.to_numeric(dly_pcp_df["PRECIP"], errors="coerce")
    dly_var_df.loc[precip_num > 0, "EVENT"] = 1.0
    dly_pcp_df["BPS"] = pd.to_numeric(dly_pcp_df["BPS"], errors="coerce").fillna(0)

    dly_pcp_df["COUNT"] = range(1, len(dly_pcp_df) + 1)
    total_bps = int(dly_pcp_df["BPS"].sum())
    pcp_source = data.slot_df(pcp_ts_out).iloc[:total_bps].copy()
    event_chunks = split_precip_by_daily_bps(pcp_source, dly_pcp_df["BPS"])

    dly_mis_df = dly_var_df.copy()
    dly_mis_df.insert(0, "PRECIP", precip_num.to_numpy())
    dly_mis_df.insert(0, "YYYYMMDD", coerce_yyyymmdd_series(daily_clustered["YYYYMMDD"]).to_numpy())
    dly_mis_df = find_missing_precip(dly_mis_df)
    dly_mis_df["YYYYMMDD"] = coerce_yyyymmdd_series(dly_mis_df["YYYYMMDD"])
    config.fill_dates = dly_mis_df["YYYYMMDD"].tolist()

    dly_var_df = _require_columns(dly_mis_df, _WX_EVENT_COLS, "daily EVENT frame after missing-precip scan")
    observed_event = pd.to_numeric(dly_var_df["EVENT"], errors="coerce")

    if str(config.verb).upper() == "T":
        print(LR, "Imputing daily solar radiation data...", end="")
    if use_phigi:
        from Source.phigi.pipeline import (
            phigi_fit_weather_types,
            phigi_impute_daily_stage,
            phigi_impute_solar_stage,
            phigi_refresh_weather_type_labels,
            phigi_wire_weather_type_clusters,
        )

        phigi_fit_weather_types(
            dly_var_df,
            dly_mis_df["YYYYMMDD"],
            config=config,
            data=data,
        )
        dly_var_df = phigi_wire_weather_type_clusters(
            dly_var_df,
            dly_mis_df["YYYYMMDD"],
            config=config,
            context="solar",
        )

        solar_df = dly_var_df.iloc[:, :2].copy()
        solar_df.columns = ["CLUSTER", "SO_RAD"]
        solar_out = phigi_impute_solar_stage(
            solar_df,
            dly_mis_df["YYYYMMDD"],
            config=config,
            data=data,
        )
        dly_var_df.iloc[:, 1] = solar_out["SO_RAD"].to_numpy()

        dly_var_df = phigi_refresh_weather_type_labels(
            dly_var_df,
            dly_mis_df["YYYYMMDD"],
            config=config,
            data=data,
            dly_ts_out=dly_ts_out,
            pcp_ts_out=pcp_ts_out,
        )

        if str(config.verb).upper() == "T":
            print(LR, "Imputing daily temperature and moisture/wind data...", end="")
        dly_var_df = phigi_impute_daily_stage(
            dly_var_df,
            dly_mis_df["YYYYMMDD"],
            config=config,
            data=data,
        )
    else:
        dly_var_df.iloc[:, :2] = impute_by_cluster(dly_var_df.iloc[:, :2], "midastouch", data=data)

        if str(config.verb).upper() == "T":
            print(LR, "Imputing daily max/min temperature data...", end="")
        dly_var_df.iloc[:, :4] = impute_by_cluster(dly_var_df.iloc[:, :4], "midastouch", data=data)
        implausible = dly_var_df.index[dly_var_df["MIN_TEMP"] >= dly_var_df["MAX_TEMP"]]
        counter = 0
        while len(implausible) > 0 and counter < 10:
            counter += 1
            if str(config.verb).upper() == "T":
                print(LR, "Reimputing implausible temperature data...", end="")
            dly_var_df.loc[implausible, "MIN_TEMP"] = np.nan
            dly_var_df.iloc[:, :4] = impute_by_cluster(dly_var_df.iloc[:, :4], "midastouch", data=data)
            implausible = dly_var_df.index[dly_var_df["MIN_TEMP"] >= dly_var_df["MAX_TEMP"]]
        if len(implausible) > 0:
            dly_var_df.loc[implausible, "MIN_TEMP"] = dly_var_df.loc[implausible, "MAX_TEMP"]

        if str(config.verb).upper() == "T":
            print(LR, "Imputing daily dew point temperature and wind data...", end="")
        # R imputes [,1:7] (weather only); EVENT is filled later with cart.
        dly_var_df.iloc[:, :7] = impute_by_cluster(
            dly_var_df.iloc[:, :7], "midastouch", data=data
        )

    if str(config.verb).upper() == "T":
        print(LR, "Imputing daily precipitation event data...", end="")
    if use_phigi:
        from Source.phigi.pipeline import phigi_impute_precip_stage

        dly_var_df, dly_pcp_df = phigi_impute_precip_stage(
            dly_var_df,
            dly_pcp_df,
            dly_mis_df["YYYYMMDD"],
            config=config,
            data=data,
        )
    else:
        event_frame = dly_var_df.copy()
        event_frame["MONTH"] = (
            coerce_yyyymmdd_series(dly_mis_df["YYYYMMDD"]).str[4:6].astype(int)
        )
        event_frame = impute_by_cluster(event_frame, "cart", data=data)
        dly_var_df["EVENT"] = event_frame["EVENT"]

    dly_var_df["EVENT"] = binarize_filled_event(dly_var_df["EVENT"], observed_event)
    event_mask = pd.to_numeric(dly_var_df["EVENT"], errors="coerce") == 1
    daily_clustered = data.slot_df(dly_ts_out)
    dly_eve_df, event_dates, cluster_events, dly_eve_con_df = assemble_event_table(
        dly_var_df, dly_pcp_df, daily_clustered, event_mask
    )
    if use_phigi and getattr(config, "phigi_cluster_lut", None) is not None:
        from Source.phigi.pipeline import merge_phigi_clusters

        dly_eve_df = merge_phigi_clusters(dly_eve_df, config=config, date_col="YYYYMMDD")
    elif "CLUSTER" not in dly_eve_df.columns:
        dly_eve_df.insert(0, "CLUSTER", pd.to_numeric(cluster_events, errors="coerce").to_numpy())
    keep_events = event_predictor_complete_mask(dly_eve_df)
    dly_eve_df = dly_eve_df.loc[keep_events].reset_index(drop=True)
    event_dates = coerce_yyyymmdd_series(dly_eve_df["YYYYMMDD"]).reset_index(drop=True)
    dly_eve_con_df = dly_eve_con_df.loc[keep_events.to_numpy()].reset_index(drop=True)
    cluster_events = pd.to_numeric(dly_eve_df["CLUSTER"], errors="coerce")

    if str(config.verb).upper() == "T":
        print(LR, "Imputing daily event characteristics data...", end="")
    if dly_eve_df.empty:
        dly_imp_df = pd.DataFrame(columns=["YYYYMMDD", "BPS", "ELEN", "PRECIP"])
    elif use_phigi:
        from Source.phigi.pipeline import phigi_impute_wet_events

        dly_imp_df = attach_event_dates(
            phigi_impute_wet_events(dly_eve_df, config=config),
            event_dates,
        )
    else:
        eve_fil = dly_eve_df.drop(columns=["YYYYMMDD"], errors="ignore")
        dly_imp_df = attach_event_dates(
            impute_by_cluster(eve_fil, "midastouch", data=data),
            event_dates,
        )

    missing_pcp = [c for c in ("BPS", "ELEN", "PRECIP") if c not in dly_imp_df.columns]
    if missing_pcp:
        raise ValueError(f"imputed event table missing columns: {missing_pcp}")
    dly_pcp_out = dly_imp_df[["YYYYMMDD", "BPS", "ELEN", "PRECIP"]].copy()
    dly_pcp_out["YYYYMMDD"] = coerce_yyyymmdd_series(dly_pcp_out["YYYYMMDD"])
    for col in ("BPS", "ELEN"):
        dly_pcp_out[col] = pd.to_numeric(dly_pcp_out[col], errors="coerce").clip(lower=0).round()
    dly_pcp_out["PRECIP"] = pd.to_numeric(dly_pcp_out["PRECIP"], errors="coerce")

    empty_days: list[int] = []
    if "EMAX" in dly_eve_con_df.columns:
        empty_days = dly_eve_con_df.index[dly_eve_con_df["EMAX"].isna()].tolist()
    if empty_days:
        # R sizes empty storms from post-impute BPS, not the pre-impute NA.
        dly_pcp_out["BPS"] = aligned_empty_storm_bps(dly_pcp_out["BPS"], empty_days)

    # Match R column reorder: dly.var.df[,c(4:2, 6, 7, 5)]
    dly_var_out = pd.DataFrame({
        "YYYYMMDD": coerce_yyyymmdd_series(dly_mis_df["YYYYMMDD"]),
        "MIN_TEMP": dly_var_df["MIN_TEMP"].to_numpy(),
        "MAX_TEMP": dly_var_df["MAX_TEMP"].to_numpy(),
        "SO_RAD": dly_var_df["SO_RAD"].to_numpy(),
        "W_VEL": dly_var_df["W_VEL"].to_numpy(),
        "W_DIR": dly_var_df["W_DIR"].to_numpy(),
        "DP_TEMP": dly_var_df["DP_TEMP"].to_numpy(),
    })
    dly_out_df = pd.merge(dly_var_out, dly_pcp_out, on="YYYYMMDD", how="left")
    data[dly_ts_out] = dly_out_df

    fil_ts_list = assign_existing_event_timeseries(
        event_chunks, n_events=len(dly_eve_df), empty_days=empty_days
    )
    if empty_days:
        template = data.slot_df(pcp_ts_out)
        for ed in empty_days:
            bps_val = (
                pd.to_numeric(dly_pcp_out.iloc[ed]["BPS"], errors="coerce")
                if ed < len(dly_pcp_out)
                else np.nan
            )
            nrows = max(1, int(bps_val) if pd.notna(bps_val) else 1)
            ymd = str(event_dates.iloc[ed])
            cluster = cluster_events.iloc[ed] if ed < len(cluster_events) else np.nan
            # PRECIP/DUR stay NA (numeric) so later imputation fills amounts.
            fil_ts_list[ed] = empty_breakpoint_frame(
                template, nrows, yyyymmdd=ymd, cluster=cluster
            )

    ordered = [fil_ts_list[k] for k in sorted(fil_ts_list.keys())]
    pcp_ts_df = pd.concat(ordered, ignore_index=True) if ordered else data.slot_df(pcp_ts_out)
    pcp_ts_df = coerce_precip_ts_numeric(pcp_ts_df)

    pcp_ts_df["YYYYMMDD"] = coerce_yyyymmdd_series(pcp_ts_df["YYYYMMDD"])
    if use_phigi and getattr(config, "phigi_cluster_lut", None) is not None:
        from Source.phigi.pipeline import merge_phigi_clusters

        pcp_ts_df["YYYYMMDD"] = coerce_yyyymmdd_series(pcp_ts_df["YYYYMMDD"])
        pcp_ts_df = merge_phigi_clusters(pcp_ts_df, config=config, date_col="YYYYMMDD")
    else:
        drop_cols = [c for c in ["CLUSTER"] if c in pcp_ts_df.columns]
        pcp_ts_df = create_cluster_col(
            pcp_ts_df.drop(columns=drop_cols, errors="ignore"), "YYYYMMDD", "%Y%m%d"
        )
    pcp_ts_df = pcp_ts_df.dropna(subset=["DT_1"]).reset_index(drop=True)

    observed_ts = pcp_ts_df.dropna(subset=["PRECIP", "DUR", "DT_1"]).copy()
    datetimes = pd.to_datetime(pcp_ts_df["DT_1"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    imp_ts_df = pcp_ts_df[["CLUSTER", "DUR", "PRECIP", "YYYYMMDD"]].copy()
    imp_ts_df["HOUR"] = pd.to_numeric(datetimes.str[11:13], errors="coerce")
    imp_ts_df["MINUTE"] = pd.to_numeric(datetimes.str[14:16], errors="coerce")
    imp_ts_df["SECOND"] = pd.to_numeric(datetimes.str[17:19], errors="coerce")
    imp_ts_df["DT_1"] = pd.to_datetime(pcp_ts_df["DT_1"])
    imp_ts_df = coerce_precip_ts_numeric(imp_ts_df)
    imp_ts_df.loc[pcp_ts_df["PRECIP"].isna(), ["DUR", "PRECIP", "HOUR", "MINUTE", "SECOND"]] = np.nan

    use_storm_template = (
        str(config.pm).upper() == "T"
        and str(config.st).upper() == "T"
        and _PYMICE_AVAILABLE
    )
    if str(config.verb).upper() == "T":
        print(LR, "Imputing precipitation time series...", end="")

    daily_events = dly_pcp_out.copy()
    daily_events["YYYYMMDD"] = coerce_yyyymmdd_series(daily_events["YYYYMMDD"])
    daily_events["CLUSTER"] = cluster_events.reset_index(drop=True).values[: len(daily_events)]
    daily_events["BPS"] = pd.to_numeric(daily_events["BPS"], errors="coerce").fillna(1).astype(int)
    daily_events["ELEN"] = pd.to_numeric(daily_events["ELEN"], errors="coerce").fillna(1).astype(int)
    daily_events["PRECIP"] = pd.to_numeric(daily_events["PRECIP"], errors="coerce").fillna(0.0)

    if use_phigi:
        from Source.phigi.pipeline import phigi_impute_subdaily_stage

        imp_ts_df, _, ei_results = phigi_impute_subdaily_stage(
            imp_ts_df,
            observed_ts=observed_ts,
            daily_events=daily_events,
            config=config,
            imp_method="midastouch",
        )
        if str(config.verb).upper() == "T" and ei_results:
            for ei in ei_results:
                if not ei.passed:
                    for line in ei.summary_lines():
                        print(LR, line, end="")
    elif use_storm_template:
        imp_ts_df, sub_report, ei_results = _pymice_impute_subdaily(
            imp_ts_df,
            observed_ts=observed_ts,
            daily_events=daily_events,
            config=config,
            imp_method="midastouch",
        )
        if str(config.verb).upper() == "T":
            print(LR, _format_impute_report(sub_report).replace("\n", LR), end="")
            for ei in ei_results:
                if not ei.passed:
                    for line in ei.summary_lines():
                        print(LR, line, end="")
    else:
        observed_mask = pcp_ts_df["PRECIP"].notna().to_numpy()
        observed_dt = pd.to_datetime(pcp_ts_df["DT_1"], errors="coerce")
        row_df = imp_ts_df.drop(columns=["DT_1", "YYYYMMDD"], errors="ignore")
        row_df = impute_by_cluster(row_df, "midastouch", data=data)
        for col in row_df.columns:
            imp_ts_df[col] = row_df[col].to_numpy()
        ymd = coerce_yyyymmdd_series(pcp_ts_df["YYYYMMDD"]).to_numpy(dtype=str)
        from_dt = observed_dt.dt.strftime("%Y%m%d").fillna("").to_numpy(dtype=str)
        ymd_ok = np.array([len(str(x)) == 8 and str(x).isdigit() for x in ymd], dtype=bool)
        ymd = np.where(ymd_ok, ymd, from_dt)
        ymd = np.array(
            [x if len(str(x)) == 8 and str(x).isdigit() else "19700101" for x in ymd],
            dtype=str,
        )
        hour = pd.to_numeric(imp_ts_df["HOUR"], errors="coerce").fillna(0).clip(0, 23).astype(int).to_numpy()
        minute = pd.to_numeric(imp_ts_df["MINUTE"], errors="coerce").fillna(0).clip(0, 59).astype(int).to_numpy()
        second = pd.to_numeric(imp_ts_df["SECOND"], errors="coerce").fillna(0).clip(0, 59).astype(int).to_numpy()
        stamp = [
            f"{d} {int(h):02d}:{int(m):02d}:{int(s):02d}"
            for d, h, m, s in zip(ymd, hour, minute, second)
        ]
        reconstructed = pd.to_datetime(stamp, format="%Y%m%d %H:%M:%S", errors="coerce")
        new_dt = reconstructed.to_numpy()
        obs_np = observed_dt.to_numpy()
        missing = pd.isna(new_dt)
        if missing.any():
            new_dt = new_dt.copy()
            new_dt[missing] = obs_np[missing]
        if len(new_dt) == len(observed_mask):
            new_dt = new_dt.copy()
            keep_obs = observed_mask & pd.notna(obs_np)
            new_dt[keep_obs] = obs_np[keep_obs]
        imp_ts_df["DT_1"] = new_dt
        if str(config.verb).upper() == "T":
            print(LR, "Recomputing overlapped precipitation events...", end="")
        # R combines equal timestamps and trims DUR. Do not retime observed storms.
        imp_ts_df = combine_duplicate_precip_times(imp_ts_df)
        imp_ts_df = trim_overlap_durations(imp_ts_df)

    keep_ts = [c for c in ("DT_1", "CLUSTER", "DUR", "PRECIP") if c in imp_ts_df.columns]
    imp_ts_df = imp_ts_df[keep_ts]
    pcp_out = imp_ts_df[["DT_1", "DUR", "PRECIP"]].copy()

    if str(config.verb).upper() == "T":
        print(LR, "Recomputing antecedent times...", end="")
    dif = imp_ts_df["DT_1"].diff().dt.total_seconds() / 60
    dif.iloc[0] = 0
    pcp_out["ANT"] = dif
    pcp_out["CUM"] = pcp_out["ANT"].cumsum()
    data[pcp_ts_out] = pcp_out

    if str(config.verb).upper() == "T":
        print(LR, "Recomputing daily precipitation amounts...", end="")
    dly_out = data.slot_df(dly_ts_out)
    dly_out["BPS"] = dly_out["BPS"].fillna(0)
    dly_out["ELEN"] = dly_out["ELEN"].fillna(0)
    dly_out["PRECIP"] = dly_out["PRECIP"].fillna(0)
    e_row = dly_out["BPS"].cumsum().astype(int).tolist()
    s_row = [1] + [e + 1 for e in e_row[:-1]]
    i_pcp = []
    pvals = pcp_out["PRECIP"].values
    for s, e in zip(s_row, e_row):
        i_pcp.append(float(np.nansum(pvals[s - 1:e])))
    dly_out["PRECIP"] = i_pcp
    data[dly_ts_out] = dly_out

    if str(config.verb).upper() == "T":
        print(LR, "Returning imputed data...", end="")
    if use_phigi:
        from Source.phigi.pipeline import finalize_phigi_gap_fill

        finalize_phigi_gap_fill(config)
    return data