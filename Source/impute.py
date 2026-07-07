"""Gap filling and cluster imputation (PyMICE FCS or mean-fill fallback)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from Source.config import LR, get_config
from Source.data_model import WeppcliffData
from Source.time_bounds import parse_user_datetime

try:
    from Source.weppcliff_impute import impute_by_cluster as _pymice_impute_by_cluster
    from Source.weppcliff_impute import impute_by_df as _pymice_impute_by_df
    from Source.weppcliff_impute import impute_subdaily_timeseries as _pymice_impute_subdaily
    from Source.weppcliff_impute.cluster import select_cluster_granularity
    from Source.weppcliff_impute.diagnostics import format_report as _format_impute_report
    from Source.weppcliff_impute.ei_validate import EIValidationResult
    from Source.weppcliff_impute.overlap import resolve_overlaps, trim_overlap_durations
    from Source.weppcliff_impute.setup import dataframe_to_matrix

    _PYMICE_AVAILABLE = True
except ImportError:
    _PYMICE_AVAILABLE = False
    EIValidationResult = None  # type: ignore

    def resolve_overlaps(df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[misc]
        return df

    def trim_overlap_durations(df: pd.DataFrame) -> pd.DataFrame:  # type: ignore[misc]
        return df


def _max_iter(fil_df: pd.DataFrame, config) -> int:
    complete = fil_df.dropna(how="all").shape[0]
    total = len(fil_df)
    max_iter = max(20, int(np.ceil(100 - 100 * complete / max(total, 1))))
    return min(int(config.io), max_iter)


def _mean_impute_by_cluster(fil_df: pd.DataFrame, imp_method: str) -> pd.DataFrame:
    """Legacy mean-fill within CLUSTER (used when ``-pm f`` or PyMICE unavailable)."""
    del imp_method
    config = get_config()
    out = fil_df.copy()
    value_cols = [c for c in out.columns if c != "CLUSTER"]

    for cluster in out["CLUSTER"].dropna().unique():
        locs = out.index[out["CLUSTER"] == cluster]
        subset = out.loc[locs, value_cols].copy()
        for col in value_cols:
            if subset[col].dtype == object or str(subset[col].dtype) == "category":
                continue
            col_mean = subset[col].mean()
            if pd.isna(col_mean):
                col_mean = out[col].mean()
            if pd.isna(col_mean):
                col_mean = 0
            out.loc[locs, col] = subset[col].fillna(col_mean)

    for col in value_cols:
        if out[col].dtype == object:
            continue
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
                df_dates = df[dt_col_name].astype(str)
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
        df[ymd_col] = df[dt_col_name].astype(str)

    cluster_lut = out_df[["YYYYMMDD", "CLUSTER"]].astype({"YYYYMMDD": str})
    if df[ymd_col].duplicated(keep=False).any():
        merge_df = df.merge(cluster_lut, on=ymd_col, how="left")
    else:
        merge_df = cluster_lut.merge(df, on=ymd_col, how="left")
    return merge_df


def find_missing_precip(in_df: pd.DataFrame) -> pd.DataFrame:
    out = in_df.copy()
    out["YYYYMM"] = out["YYYYMMDD"].astype(str).str[:6].astype(int)
    yrs_observed = out["YYYYMMDD"].astype(str).str[:4].nunique()
    out["EVENT"] = pd.to_numeric(out["EVENT"], errors="coerce")

    monthly_events = out.groupby("YYYYMM")["EVENT"].sum(min_count=1).reset_index()
    cluster_events = out.groupby("CLUSTER")["EVENT"].sum(min_count=1).reset_index()

    vals = monthly_events["EVENT"].fillna(0).values
    changes = np.where(vals[:-1] != vals[1:])[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(vals)]])
    lengths = ends - starts
    run_vals = vals[starts]
    zero_streaks = lengths[run_vals == 0]

    streak_max = 0
    if len(zero_streaks) > 6:
        bins = pd.cut(zero_streaks, 3)
        zero_bins = bins.value_counts().sort_index().values
        if len(zero_bins) == 3:
            if zero_bins[0] < zero_bins[1] and zero_bins[2] < zero_bins[1]:
                streak_max = int(np.quantile(zero_streaks, 0.75))
            elif zero_bins[0] >= zero_bins[1]:
                streak_max = 0

    zero_idx = np.where(run_vals == 0)[0]
    missing_months = []
    for zi in zero_idx:
        if lengths[zi] > streak_max:
            s, e = starts[zi], ends[zi]
            missing_months.extend(monthly_events["YYYYMM"].iloc[s:e].tolist())
    if missing_months:
        out.loc[out["YYYYMM"].isin(missing_months), "EVENT"] = np.nan

    abs_limit = 7
    rel_limit = 0.025
    dry_clusters = set(
        cluster_events.loc[
            (cluster_events["EVENT"] <= abs_limit)
            | (cluster_events["EVENT"] / max(yrs_observed, 1) < rel_limit),
            "CLUSTER",
        ].tolist()
    )
    missing_days = out.index[out["EVENT"].isna()]
    dry_days = out.index[out["CLUSTER"].isin(dry_clusters)]
    zero_days = missing_days.intersection(dry_days)
    out.loc[zero_days, "EVENT"] = 0
    return out


def fill_data(
    data: WeppcliffData,
    pcp_ts_in: int,
    dly_ts_in: int,
    dly_ts_out: int,
    pcp_ts_out: int,
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
        .assign(YYYYMMDD=lambda d: d["YYYYMMDD"].astype(str))
    )
    pcp_ymd = pd.to_datetime(pcp_in["DT_1"], errors="coerce").dt.strftime("%Y%m%d")
    pcp_cluster_body = pcp_in.copy()
    pcp_cluster_body["YYYYMMDD"] = pcp_ymd
    pcp_cluster_body = pcp_cluster_body.merge(day_clusters, on="YYYYMMDD", how="left")
    pcp_cluster = pcp_cluster_body.dropna(subset=["DT_1", "CLUSTER"]).reset_index(drop=True)
    data[pcp_ts_out] = pcp_cluster

    dly_var_df = data.slot_df(dly_ts_out).iloc[:, [1, 4, 3, 2, 7, 5, 6]].copy()
    dly_var_df.columns = list(dly_var_df.columns)
    ncol = data.slot_df(dly_ts_out).shape[1]
    dly_pcp_df = data.slot_df(dly_ts_out).iloc[:, [1] + list(range(ncol - 5, ncol))].copy()

    dly_var_df["EVENT"] = 0
    if "PRECIP" in dly_pcp_df.columns:
        dly_var_df.loc[dly_pcp_df["PRECIP"] > 0, "EVENT"] = 1
    dly_var_df["EVENT"] = dly_var_df["EVENT"].astype("category")
    dly_pcp_df["BPS"] = dly_pcp_df["BPS"].fillna(0)

    dly_pcp_df["COUNT"] = range(1, len(dly_pcp_df) + 1)
    pcp_present = dly_pcp_df["BPS"].notna()
    split_parts = []
    for _, row in dly_pcp_df[pcp_present].iterrows():
        split_parts.extend([row["COUNT"]] * int(row["BPS"]))
    total_bps = int(dly_pcp_df["BPS"].fillna(0).sum())
    pcp_source = data.slot_df(pcp_ts_out).iloc[:total_bps]
    if split_parts:
        chunks = np.array_split(pcp_source, len(split_parts))
        pcp_ts_list = {
            k: (c if isinstance(c, pd.DataFrame) else pd.DataFrame(c, columns=pcp_source.columns))
            for k, c in zip(split_parts, chunks)
        }
    else:
        pcp_ts_list = {}

    dly_mis_df = pd.concat([
        data.slot_df(dly_ts_out).iloc[:, [0, 12]],
        dly_var_df,
    ], axis=1)
    dly_mis_df.columns = ["YYYYMMDD", "PRECIP"] + list(dly_var_df.columns)
    dly_mis_df = find_missing_precip(dly_mis_df)
    config.fill_dates = dly_mis_df["YYYYMMDD"].astype(str).tolist()

    cluster_col = dly_var_df.iloc[:, 0]
    dly_var_df = pd.concat([cluster_col, dly_mis_df.iloc[:, 3:10]], axis=1)
    dly_var_df.columns = ["CLUSTER"] + list(dly_mis_df.columns[3:10])

    if str(config.verb).upper() == "T":
        print(LR, "Imputing daily solar radiation data...", end="")
    if getattr(config, "do_phigi", str(config.gf).lower() == "phigi"):
        from Source.phigi.pipeline import (
            phigi_fit_weather_types,
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
    dly_var_df = impute_by_cluster(dly_var_df, "midastouch", data=data)

    if str(config.verb).upper() == "T":
        print(LR, "Imputing daily precipitation event data...", end="")
    dly_var_df = impute_by_cluster(dly_var_df, "cart", data=data)

    event_mask = dly_var_df["EVENT"].astype(float) == 1
    date_match = data.slot_df(dly_ts_out).loc[event_mask, data.slot_df(dly_ts_out).columns[0]]
    cluster_match = data.slot_df(dly_ts_out).loc[event_mask, "CLUSTER"]
    dly_eve_var_df = dly_var_df[event_mask].copy()
    dly_eve_pcp_df = dly_pcp_df.loc[event_mask].iloc[:, [1, 2, 5]].copy()
    dly_eve_con_df = dly_pcp_df.loc[event_mask, dly_pcp_df.columns[2:5]].copy()
    dly_eve_pcp_df.loc[dly_eve_pcp_df["BPS"] == 0, "BPS"] = np.nan

    dly_eve_df = pd.concat([dly_eve_var_df.drop(columns=["CLUSTER"], errors="ignore"), dly_eve_pcp_df, date_match.reset_index(drop=True)], axis=1)
    dly_eve_df.columns = list(dly_eve_var_df.drop(columns=["CLUSTER"], errors="ignore").columns) + list(dly_eve_pcp_df.columns) + ["YYYYMMDD"]
    dly_eve_df["YYYYMMDD"] = dly_eve_df["YYYYMMDD"].astype(str)
    if getattr(config, "do_phigi", str(config.gf).lower() == "phigi") and getattr(
        config, "phigi_cluster_lut", None
    ) is not None:
        from Source.phigi.pipeline import merge_phigi_clusters

        dly_eve_df = merge_phigi_clusters(dly_eve_df, config=config, date_col="YYYYMMDD")
    else:
        dly_eve_df = create_cluster_col(dly_eve_df, "YYYYMMDD", "%Y%m%d")
    keep_events = dly_eve_df.iloc[:, :9].notna().all(axis=1)
    event_dates = dly_eve_df.loc[keep_events, "YYYYMMDD"].astype(str).reset_index(drop=True)
    cluster_events = cluster_match.loc[event_mask].reset_index(drop=True).loc[keep_events].reset_index(drop=True)
    dly_eve_df = dly_eve_df.loc[keep_events].iloc[:, 1:].reset_index(drop=True)

    if str(config.verb).upper() == "T":
        print(LR, "Imputing daily event characteristics data...", end="")
    dly_imp_df = pd.concat([event_dates, impute_by_cluster(dly_eve_df, "midastouch", data=data)], axis=1)

    dly_pcp_out = dly_imp_df.iloc[:, [0] + list(range(9, dly_imp_df.shape[1]))]
    dly_pcp_out.columns = ["YYYYMMDD", "BPS", "ELEN", "PRECIP"][: dly_pcp_out.shape[1]]
    dly_var_out = pd.DataFrame({
        "YYYYMMDD": dly_mis_df["YYYYMMDD"],
        "MIN_TEMP": dly_var_df["MAX_TEMP"],
        "MAX_TEMP": dly_var_df["MIN_TEMP"],
        "SO_RAD": dly_var_df["W_VEL"],
        "W_VEL": dly_var_df["W_DIR"],
        "W_DIR": dly_var_df["DP_TEMP"],
        "DP_TEMP": dly_var_df["SO_RAD"],
    })
    # Match R column reorder: dly.var.df[,c(4:2, 6, 7, 5)]
    dly_var_out = pd.DataFrame({
        "YYYYMMDD": dly_mis_df["YYYYMMDD"],
        "MIN_TEMP": dly_var_df.iloc[:, 3],
        "MAX_TEMP": dly_var_df.iloc[:, 2],
        "SO_RAD": dly_var_df.iloc[:, 1],
        "W_VEL": dly_var_df.iloc[:, 5],
        "W_DIR": dly_var_df.iloc[:, 6],
        "DP_TEMP": dly_var_df.iloc[:, 4],
    })
    dly_out_df = pd.merge(dly_var_out, dly_pcp_out, on="YYYYMMDD", how="left")
    data[dly_ts_out] = dly_out_df

    # Recombine precipitation time series (simplified list rebuild)
    fil_ts_list: dict[int, pd.DataFrame] = {}
    pcp_list = list(range(len(dly_eve_df)))
    empty_days = dly_imp_df.index[dly_imp_df.isna().any(axis=1)].tolist()
    pcp_list = [i for i in pcp_list if i not in empty_days]
    ts_items = list(pcp_ts_list.values())
    for i, pl in enumerate(pcp_list):
        if i < len(ts_items):
            fil_ts_list[pl] = ts_items[i]
    if empty_days:
        col_names = list(data.slot_df(pcp_ts_out).columns)
        for idx, ed in enumerate(empty_days):
            nrows = int(dly_eve_df.iloc[ed]["BPS"]) if ed < len(dly_eve_df) else 1
            fil_ts_list[ed] = pd.DataFrame(index=range(nrows), columns=col_names)
            fil_ts_list[ed]["DT_1"] = pd.to_datetime(str(event_dates.iloc[ed]), format="%Y%m%d")
            if len(col_names) > 1:
                fil_ts_list[ed][col_names[1]] = event_dates.iloc[ed]
            if len(col_names) > 2:
                fil_ts_list[ed][col_names[2]] = cluster_events.iloc[ed]

    ordered = [fil_ts_list[k] for k in sorted(fil_ts_list.keys())]
    pcp_ts_df = pd.concat(ordered, ignore_index=True) if ordered else data.slot_df(pcp_ts_out)

    pcp_ts_df["YYYYMMDD"] = pcp_ts_df["YYYYMMDD"].astype(str)
    if getattr(config, "do_phigi", str(config.gf).lower() == "phigi") and getattr(
        config, "phigi_cluster_lut", None
    ) is not None:
        from Source.phigi.pipeline import merge_phigi_clusters

        pcp_ts_df["YYYYMMDD"] = pcp_ts_df["YYYYMMDD"].astype(str)
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
    imp_ts_df["HOUR"] = datetimes.str[11:13].astype(float)
    imp_ts_df["MINUTE"] = datetimes.str[14:16].astype(float)
    imp_ts_df["SECOND"] = datetimes.str[17:19].astype(float)
    imp_ts_df["DT_1"] = pd.to_datetime(pcp_ts_df["DT_1"])
    imp_ts_df.loc[pcp_ts_df["PRECIP"].isna(), ["DUR", "PRECIP", "HOUR", "MINUTE", "SECOND"]] = np.nan

    use_storm_template = (
        str(config.pm).upper() == "T"
        and str(config.st).upper() == "T"
        and _PYMICE_AVAILABLE
    )

    if str(config.verb).upper() == "T":
        print(LR, "Imputing precipitation time series...", end="")

    if use_storm_template:
        daily_events = dly_pcp_out.copy()
        daily_events["YYYYMMDD"] = daily_events["YYYYMMDD"].astype(str)
        daily_events["CLUSTER"] = cluster_events.reset_index(drop=True).values[: len(daily_events)]
        daily_events["BPS"] = pd.to_numeric(daily_events["BPS"], errors="coerce").fillna(1).astype(int)
        daily_events["ELEN"] = pd.to_numeric(daily_events["ELEN"], errors="coerce").fillna(1).astype(int)
        daily_events["PRECIP"] = pd.to_numeric(daily_events["PRECIP"], errors="coerce").fillna(0.0)
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
        row_df = imp_ts_df.drop(columns=["DT_1", "YYYYMMDD"], errors="ignore")
        row_df = impute_by_cluster(row_df, "midastouch", data=data)
        for col in row_df.columns:
            imp_ts_df[col] = row_df[col].to_numpy()
        imp_ts_df["DT_1"] = pd.to_datetime(
            pcp_ts_df["YYYYMMDD"].astype(str) + " " +
            imp_ts_df["HOUR"].astype(int).astype(str).str.zfill(2) + ":" +
            imp_ts_df["MINUTE"].astype(int).astype(str).str.zfill(2) + ":" +
            imp_ts_df["SECOND"].astype(int).astype(str).str.zfill(2),
            format="%Y%m%d %H:%M:%S",
        )
        if _PYMICE_AVAILABLE:
            imp_ts_df = resolve_overlaps(imp_ts_df)
            imp_ts_df = trim_overlap_durations(imp_ts_df)
        else:
            counter = 0
            continue_loop = True
            if str(config.verb).upper() == "T":
                print(LR, "Recomputing overlapped precipitation events...", end="")
            while continue_loop and counter < 10000:
                counter += 1
                imp_ts_df = imp_ts_df.sort_values("DT_1").reset_index(drop=True)
                dup_locs = imp_ts_df.index[imp_ts_df["DT_1"].duplicated(keep="last")]
                if len(dup_locs) == 0:
                    continue_loop = False
                else:
                    for loc in dup_locs:
                        if loc > 0:
                            imp_ts_df.loc[loc - 1, "PRECIP"] += imp_ts_df.loc[loc, "PRECIP"]
                    imp_ts_df = imp_ts_df.drop(index=dup_locs).reset_index(drop=True)
                    pcp_ts_df = pcp_ts_df.drop(index=dup_locs).reset_index(drop=True)
            imp_ts_df = trim_overlap_durations(imp_ts_df)

    imp_ts_df = imp_ts_df[["DT_1", "CLUSTER", "DUR", "PRECIP"]]

    pcp_out = pcp_ts_df.drop(columns=[c for c in pcp_ts_df.columns[:3] if c in pcp_ts_df.columns], errors="ignore")
    pcp_out = pd.concat([imp_ts_df[["DT_1", "DUR", "PRECIP"]], pcp_out.iloc[:, 3:]], axis=1)

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
    if getattr(config, "do_ir", config.is_true("ir")) and getattr(
        config, "do_phigi", str(config.gf).lower() == "phigi"
    ):
        from Source.phigi.pipeline import write_imputation_report

        write_imputation_report(config)
    return data