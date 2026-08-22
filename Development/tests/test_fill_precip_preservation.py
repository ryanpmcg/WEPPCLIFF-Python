"""Regression tests for WEPPCLIFF fill_data precip preservation.

The 2026-08-21 KMQE diagnosis found three interacting bugs that dropped
observed storm mass from ~23,874 mm to ~227 mm under ``-id``:

1. Breakpoint grouping used a COUNT-keyed dict + ``np.array_split``, so only
   the last 1-row chunk per wet day survived (R ``split()`` keeps BPS rows).
2. EVENT-day frames were ``concat``-ed on the original integer index while
   dates were ``reset_index``, pairing the wrong days.
3. PHIGI ``enforce_dry_days`` zeroed PRECIP on every EVENT==0 row, including
   days that already had observed positive precip.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.config import WeppcliffConfig, set_config  # noqa: E402
from Source.data_model import WeppcliffData  # noqa: E402
from Source.impute import (  # noqa: E402
    aligned_empty_storm_bps,
    assemble_event_table,
    assign_existing_event_timeseries,
    attach_event_dates,
    binarize_filled_event,
    coerce_precip_ts_numeric,
    combine_duplicate_precip_times,
    empty_breakpoint_frame,
    event_predictor_complete_mask,
    fill_data,
    impute_by_cluster,
    split_precip_by_daily_bps,
)
from Source.phigi.l4_precip_hurdle import enforce_dry_days  # noqa: E402


def test_split_precip_by_daily_bps_matches_r_split() -> None:
    """Day with BPS=3,1,2 must receive 3, then 1, then 2 consecutive rows."""
    n = 3 + 1 + 2
    src = pd.DataFrame(
        {
            "DT_1": pd.date_range("2000-01-01", periods=n, freq="h"),
            "PRECIP": np.arange(1, n + 1, dtype=float),
            "DUR": np.full(n, 10.0),
        }
    )
    bps = pd.Series([3, 0, 1, 2, 0])
    chunks = split_precip_by_daily_bps(src, bps)
    assert [len(c) for c in chunks] == [3, 1, 2]
    assert chunks[0]["PRECIP"].tolist() == [1.0, 2.0, 3.0]
    assert chunks[1]["PRECIP"].tolist() == [4.0]
    assert chunks[2]["PRECIP"].tolist() == [5.0, 6.0]


def test_split_precip_by_daily_bps_does_not_overwrite_by_count() -> None:
    """A COUNT-keyed dict would keep only the last row of each day."""
    src = pd.DataFrame({"PRECIP": [10.0, 20.0, 30.0, 40.0]})
    bps = pd.Series([2, 2])
    chunks = split_precip_by_daily_bps(src, bps)
    assert chunks[0]["PRECIP"].tolist() == [10.0, 20.0]
    assert chunks[1]["PRECIP"].tolist() == [30.0, 40.0]


def test_assemble_event_table_aligns_dates_with_wet_days() -> None:
    """Wet days at sparse indices must keep their own YYYYMMDD, not 0..n-1 dates."""
    daily = pd.DataFrame(
        {
            "YYYYMMDD": ["20000101", "20000102", "20000103", "20000104"],
            "CLUSTER": [1, 1, 1, 1],
        }
    )
    dly_var = pd.DataFrame(
        {
            "CLUSTER": [1, 1, 1, 1],
            "SO_RAD": [1.0, 2.0, 3.0, 4.0],
            "EVENT": [0.0, 1.0, 0.0, 1.0],
        }
    )
    dly_pcp = pd.DataFrame(
        {
            "CLUSTER": [1, 1, 1, 1],
            "BPS": [0, 2, 0, 1],
            "ELEN": [0, 1, 0, 1],
            "EMAX": [np.nan, 5.0, np.nan, 3.0],
            "EMIN": [np.nan, 1.0, np.nan, 1.0],
            "PRECIP": [0.0, 8.0, 0.0, 2.0],
        }
    )
    event_mask = dly_var["EVENT"] == 1
    eve, dates, clusters, con = assemble_event_table(
        dly_var, dly_pcp, daily, event_mask
    )
    assert dates.tolist() == ["20000102", "20000104"]
    assert eve["PRECIP"].tolist() == [8.0, 2.0]
    assert eve["SO_RAD"].tolist() == [2.0, 4.0]
    assert eve["YYYYMMDD"].tolist() == ["20000102", "20000104"]
    assert "CLUSTER" in eve.columns
    assert clusters.tolist() == [1, 1]
    assert con["EMAX"].tolist() == [5.0, 3.0]


def test_assign_existing_event_timeseries_positional_like_r() -> None:
    chunks = [
        pd.DataFrame({"PRECIP": [1.0, 2.0]}),
        pd.DataFrame({"PRECIP": [3.0]}),
    ]
    fil = assign_existing_event_timeseries(chunks, n_events=3, empty_days=[1])
    assert 0 in fil and 2 in fil and 1 not in fil
    assert fil[0]["PRECIP"].tolist() == [1.0, 2.0]
    assert fil[2]["PRECIP"].tolist() == [3.0]


def test_binarize_filled_event_keeps_observed_wet_and_dry() -> None:
    observed = pd.Series([1.0, 0.0, np.nan, 1.0, np.nan])
    filled = pd.Series([0.2, 0.9, 0.8, 0.1, 0.1])
    out = binarize_filled_event(filled, observed)
    assert out.tolist() == [1.0, 0.0, 1.0, 1.0, 0.0]


def test_enforce_dry_days_does_not_zero_observed_precip() -> None:
    var = pd.DataFrame({"EVENT": [0.0, 0.0, 1.0]})
    pcp = pd.DataFrame(
        {
            "BPS": [2.0, np.nan, 1.0],
            "ELEN": [1.0, np.nan, 1.0],
            "PRECIP": [12.5, np.nan, 3.0],
        }
    )
    out = enforce_dry_days(var, pcp)
    assert out["PRECIP"].iloc[0] == 12.5
    assert out["BPS"].iloc[0] == 2.0
    assert out["PRECIP"].iloc[1] == 0.0
    assert out["PRECIP"].iloc[2] == 3.0


def test_event_predictor_complete_mask_allows_missing_bps() -> None:
    """R complete.cases on weather+EVENT keeps rows whose BPS is NA to impute."""
    eve = pd.DataFrame(
        {
            "CLUSTER": [1.0, 1.0, 1.0],
            "SO_RAD": [1.0, 2.0, np.nan],
            "EVENT": [1.0, 1.0, 1.0],
            "BPS": [np.nan, 2.0, 1.0],
            "ELEN": [np.nan, 1.0, 1.0],
            "PRECIP": [np.nan, 8.0, 2.0],
            "YYYYMMDD": ["20000101", "20000102", "20000103"],
        }
    )
    keep = event_predictor_complete_mask(eve)
    assert keep.tolist() == [True, True, False]


def test_combine_duplicate_precip_times_sums_depth() -> None:
    base = pd.Timestamp("2000-01-04 21:55:01")
    src = pd.DataFrame(
        {
            "DT_1": [base, base, base + pd.Timedelta(minutes=1)],
            "PRECIP": [1.0, 2.0, 3.0],
            "DUR": [5.0, 5.0, 5.0],
        }
    )
    out = combine_duplicate_precip_times(src)
    assert len(out) == 2
    assert float(out.iloc[0]["PRECIP"]) == 3.0
    assert float(out.iloc[1]["PRECIP"]) == 3.0
    triple = pd.DataFrame(
        {
            "DT_1": [base, base, base],
            "PRECIP": [1.0, 2.0, 4.0],
            "DUR": [5.0, 6.0, 4.0],
        }
    )
    merged = combine_duplicate_precip_times(triple)
    assert len(merged) == 1
    assert float(merged.iloc[0]["PRECIP"]) == 7.0
    assert float(merged.iloc[0]["DUR"]) == 6.0


def test_attach_event_dates_overwrites_existing_column() -> None:
    """Impute keeps YYYYMMDD; assigning again must not raise duplicate-insert."""
    imp = pd.DataFrame(
        {
            "CLUSTER": [1.0, 1.0],
            "YYYYMMDD": ["19990101", "19990102"],
            "BPS": [1.0, 2.0],
            "ELEN": [1.0, 1.0],
            "PRECIP": [3.0, 4.0],
        }
    )
    dates = pd.Series(["20000102", "20000104"])
    out = attach_event_dates(imp, dates)
    assert out["YYYYMMDD"].tolist() == ["20000102", "20000104"]
    assert list(out.columns).count("YYYYMMDD") == 1


def _synthetic_fill_station(n_days: int = 400, seed: int = 7) -> tuple[WeppcliffData, float]:
    """Daily climate + 2-breakpoint wet days spanning >1 year."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_days, freq="D")
    ymd = dates.strftime("%Y%m%d")
    wet = np.zeros(n_days, dtype=bool)
    wet[::4] = True
    wet_n = int(wet.sum())
    min_t = rng.normal(5.0, 6.0, n_days)
    max_t = min_t + rng.uniform(4.0, 12.0, n_days)
    daily = pd.DataFrame(
        {
            "YYYYMMDD": ymd,
            "MIN_TEMP": min_t,
            "MAX_TEMP": max_t,
            "SO_RAD": rng.uniform(80.0, 350.0, n_days),
            "W_VEL": rng.uniform(1.0, 8.0, n_days),
            "W_DIR": rng.uniform(0.0, 360.0, n_days),
            "DP_TEMP": min_t - 1.5,
            "BPS": np.where(wet, 2.0, 0.0),
            "ELEN": np.where(wet, 1.0, np.nan),
            "EMAX": np.where(wet, 1.0, np.nan),
            "EMIN": np.where(wet, 1.0, np.nan),
            "PRECIP": np.where(wet, 10.0, np.nan),
        }
    )
    daily.loc[10:24, "MIN_TEMP"] = np.nan
    rows = []
    for day, is_wet in zip(dates, wet):
        if not is_wet:
            continue
        rows.append({"DT_1": day + pd.Timedelta(hours=10), "DUR": 15.0, "PRECIP": 6.0})
        rows.append(
            {
                "DT_1": day + pd.Timedelta(hours=10, minutes=15),
                "DUR": 15.0,
                "PRECIP": 4.0,
            }
        )
    pcp = pd.DataFrame(rows)
    data = WeppcliffData()
    data[5] = pcp
    data[10] = daily
    data.metadata["LAT"] = 42.2
    data.metadata["LON"] = -71.1
    data.metadata["ELEV"] = 25.0
    return data, float(10.0 * wet_n)


def test_aligned_empty_storm_bps_uses_imputed_counts() -> None:
    """New EVENT days must get post-impute BPS rows, not a single NA placeholder."""
    bps = pd.Series([3.0, np.nan, 4.2, 0.0, 2.0])
    out = aligned_empty_storm_bps(bps, empty_days=[1, 3])
    assert out.tolist() == [3.0, 1.0, 4.0, 1.0, 2.0]
    multi = aligned_empty_storm_bps(pd.Series([np.nan, 6.7, np.nan]), empty_days=[0, 1, 2])
    assert multi.tolist() == [1.0, 7.0, 1.0]


def test_empty_breakpoint_frame_keeps_numeric_precip() -> None:
    """All-object empty frames were the reason missing-event PRECIP never filled."""
    template = pd.DataFrame(
        {
            "DT_1": pd.to_datetime(["2000-01-01 12:00:00"]),
            "YYYYMMDD": ["20000101"],
            "CLUSTER": [1.0],
            "DUR": [15.0],
            "PRECIP": [3.2],
        }
    )
    frame = empty_breakpoint_frame(template, 2, yyyymmdd="20000102", cluster=1.0)
    assert pd.api.types.is_numeric_dtype(frame["PRECIP"])
    assert pd.api.types.is_numeric_dtype(frame["DUR"])
    assert frame["PRECIP"].isna().all()
    cat = pd.concat([template, frame], ignore_index=True)
    cat = coerce_precip_ts_numeric(cat)
    assert pd.api.types.is_numeric_dtype(cat["PRECIP"])
    assert pd.api.types.is_numeric_dtype(cat["DUR"])
    assert cat["PRECIP"].isna().sum() == 2


def test_impute_by_cluster_fills_object_na_precip_rows() -> None:
    """Object-dtype PRECIP/DUR (empty-event concat) must still be mean-filled."""
    cfg = WeppcliffConfig(pm="f", gf="legacy", verb="f")
    set_config(cfg)
    df = pd.DataFrame(
        {
            "CLUSTER": [1.0, 1.0, 1.0, 1.0],
            "DUR": [15.0, 10.0, np.nan, np.nan],
            "PRECIP": [2.0, 4.0, np.nan, np.nan],
            "HOUR": [10.0, 11.0, np.nan, np.nan],
            "MINUTE": [0.0, 15.0, np.nan, np.nan],
            "SECOND": [0.0, 0.0, np.nan, np.nan],
        }
    )
    df["PRECIP"] = df["PRECIP"].astype(object)
    df["DUR"] = df["DUR"].astype(object)
    out = impute_by_cluster(df, "midastouch")
    assert pd.api.types.is_numeric_dtype(out["PRECIP"])
    assert out["PRECIP"].notna().all()
    assert out["DUR"].notna().all()
    assert float(out["PRECIP"].iloc[2]) > 0


def test_fill_data_imputes_missing_zero_event_months() -> None:
    """Months with no observed storms must gain filled wet days and extra mass."""
    cfg = WeppcliffConfig(pm="t", gf="legacy", st="f", qi="t", verb="f", io=20)
    set_config(cfg)
    cfg.apply_runtime_flags()
    data, _obs_mass = _synthetic_fill_station(n_days=800, seed=11)
    daily_in = data.slot_df(10).copy()
    ymd = daily_in["YYYYMMDD"].astype(str)
    gap = ymd.str.startswith("201807") | ymd.str.startswith("201808")
    daily_in.loc[gap, "BPS"] = 0.0
    daily_in.loc[gap, ["ELEN", "EMAX", "EMIN", "PRECIP"]] = np.nan
    data[10] = daily_in
    gap_dates = set(ymd[gap].tolist())
    pcp = data.slot_df(5).copy()
    pcp_ymd = pd.to_datetime(pcp["DT_1"]).dt.strftime("%Y%m%d")
    data[5] = pcp.loc[~pcp_ymd.isin(gap_dates)].reset_index(drop=True)
    remaining_mass = float(pd.to_numeric(data.slot_df(5)["PRECIP"], errors="coerce").sum())
    n_obs_bp = len(data.slot_df(5))
    pre_wet = int((pd.to_numeric(daily_in["PRECIP"], errors="coerce") > 0).sum())
    filled = fill_data(data, 5, 10, 18, 15)
    daily = filled.slot_df(18)
    pcp_out = filled.slot_df(15)
    post_p = pd.to_numeric(daily["PRECIP"], errors="coerce")
    post_wet = int((post_p > 0).sum())
    gap_daily = daily.loc[daily["YYYYMMDD"].astype(str).isin(gap_dates)]
    gap_wet = int((pd.to_numeric(gap_daily["PRECIP"], errors="coerce") > 0).sum())
    new_mass = float(post_p.sum(skipna=True))
    assert post_wet > pre_wet, f"wet days {post_wet} vs pre {pre_wet}"
    assert gap_wet > 0, "missing July-August months were not filled"
    assert new_mass > remaining_mass, f"mass {new_mass:.1f} vs remaining {remaining_mass:.1f}"
    assert float(post_p.max()) < 500.0, f"date-like PRECIP? max={float(post_p.max())}"
    bp_p = pd.to_numeric(pcp_out["PRECIP"], errors="coerce")
    assert bp_p.notna().all()
    assert float(bp_p[bp_p > 0].sum()) > 0
    extra_bp = len(pcp_out) - n_obs_bp
    extra_wet = post_wet - pre_wet
    assert extra_wet > 0
    assert extra_bp >= extra_wet, (
        f"new storms got {extra_bp} extra breakpoint rows for {extra_wet} extra wet days"
    )
    merged = pd.DataFrame(
        {
            "YYYYMMDD": daily_in["YYYYMMDD"].astype(str),
            "pre": pd.to_numeric(daily_in["PRECIP"], errors="coerce"),
        }
    ).merge(
        pd.DataFrame(
            {
                "YYYYMMDD": daily["YYYYMMDD"].astype(str),
                "post": post_p,
            }
        ),
        on="YYYYMMDD",
        how="inner",
    )
    wet_pre = merged["pre"] > 0
    pre_mass = float(merged.loc[wet_pre, "pre"].sum())
    post_obs = float(merged.loc[wet_pre, "post"].sum())
    assert post_obs / pre_mass > 0.85, f"observed wet mass {post_obs:.1f} vs {pre_mass:.1f}"


def test_fill_data_mean_preserves_observed_precip_mass() -> None:
    """Legacy mean-fill must not wipe synthetic observed storm mass."""
    cfg = WeppcliffConfig(pm="f", gf="legacy", st="f", qi="f", verb="f", io=20)
    set_config(cfg)
    cfg.apply_runtime_flags()
    data, obs_mass = _synthetic_fill_station()
    filled = fill_data(data, 5, 10, 18, 15)
    daily = filled.slot_df(18)
    new_mass = float(pd.to_numeric(daily["PRECIP"], errors="coerce").sum(skipna=True))
    assert obs_mass > 100
    assert new_mass / obs_mass > 0.85, f"precip mass {new_mass:.1f} vs observed {obs_mass:.1f}"
    orig = data.slot_df(10)
    orig["YYYYMMDD"] = orig["YYYYMMDD"].astype(str)
    daily["YYYYMMDD"] = daily["YYYYMMDD"].astype(str)
    merged = orig.merge(daily, on="YYYYMMDD", suffixes=("_obs", "_fil"))
    wet = pd.to_numeric(merged["PRECIP_obs"], errors="coerce") > 0
    obs_wet = float(pd.to_numeric(merged.loc[wet, "PRECIP_obs"], errors="coerce").sum())
    fil_wet = float(pd.to_numeric(merged.loc[wet, "PRECIP_fil"], errors="coerce").sum())
    assert fil_wet / obs_wet > 0.85, f"wet-day mass {fil_wet:.1f} vs {obs_wet:.1f}"


def test_fill_data_phigi_preserves_observed_precip_mass() -> None:
    """PHIGI fill must not zero observed wet-day PRECIP on the synthetic station."""
    cfg = WeppcliffConfig(pm="f", gf="phigi", st="f", qi="f", verb="f", io=20)
    set_config(cfg)
    cfg.apply_runtime_flags()
    data, obs_mass = _synthetic_fill_station()
    filled = fill_data(data, 5, 10, 18, 15)
    daily = filled.slot_df(18)
    new_mass = float(pd.to_numeric(daily["PRECIP"], errors="coerce").sum(skipna=True))
    assert obs_mass > 100
    assert new_mass / obs_mass > 0.85, f"precip mass {new_mass:.1f} vs observed {obs_mass:.1f}"


def _run_kmqe_fill_guard(gf: str, file_stem: str, summary_name: str) -> None:
    """Run full-record KMQE ``-id`` fill and compare precip mass to Autotest.1."""
    import subprocess

    python = ROOT / ".venv" / "bin" / "python"
    if not python.exists():
        pytest.skip("project venv missing")
    autotest_daily = ROOT / "Autotest" / "Output" / "AutoTest.1-Daily-Time-Series.csv"
    if not autotest_daily.exists():
        pytest.skip("Autotest.1 daily reference missing")
    out_dir = ROOT / "Autotest" / "Output" / "FillPrecipGuard"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python),
        str(ROOT / "weppcliff.py"),
        "-f",
        "ASOS_BPT_KMQE.csv",
        "-la",
        "y",
        "-qc",
        "-id",
        "-pm",
        "-gf",
        gf,
        "-qi",
        "-ed",
        "6",
        "-o",
        str(out_dir),
        "-e",
        str(out_dir),
        "-fn",
        file_stem,
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    filled = pd.read_csv(out_dir / f"{file_stem}-Daily-Time-Series.csv")
    ref = pd.read_csv(autotest_daily)
    obs = pd.to_numeric(ref["PRECIP"], errors="coerce")
    new = pd.to_numeric(filled["PRECIP"], errors="coerce")
    obs_sum = float(obs.sum(skipna=True))
    new_sum = float(new.sum(skipna=True))
    ref["year"] = ref["YYYYMMDD"].astype(str).str[:4]
    filled["year"] = filled["YYYYMMDD"].astype(str).str[:4]
    yearly = (
        pd.DataFrame({"obs": obs, "year": ref["year"]})
        .groupby("year")["obs"]
        .sum(min_count=1)
        .to_frame()
        .join(
            pd.DataFrame({"new": new, "year": filled["year"]}).groupby("year")["new"].sum(min_count=1),
            how="outer",
        )
    )
    cmp = pd.DataFrame(
        {
            "YYYYMMDD": ref["YYYYMMDD"].astype(str),
            "obs": obs.to_numpy(),
        }
    ).merge(
        pd.DataFrame(
            {
                "YYYYMMDD": filled["YYYYMMDD"].astype(str),
                "new": new.to_numpy(),
            }
        ),
        on="YYYYMMDD",
        how="left",
    )
    zeroed = int(((cmp["obs"] > 0) & cmp["new"].fillna(0).le(0)).sum())
    summary = {
        "gf": gf,
        "obs_sum_mm": obs_sum,
        "new_sum_mm": new_sum,
        "ratio": None if obs_sum == 0 else new_sum / obs_sum,
        "obs_wet_days": int((obs > 0).sum()),
        "new_wet_days": int((new > 0).sum()),
        "observed_wet_days_zeroed": zeroed,
        "obs_bps_sum": float(pd.to_numeric(ref["BPS"], errors="coerce").sum(skipna=True)),
        "new_bps_sum": float(pd.to_numeric(filled["BPS"], errors="coerce").sum(skipna=True)),
        "yearly": {
            str(idx): {"obs": float(row["obs"]), "new": float(row["new"])}
            for idx, row in yearly.fillna(0).iterrows()
        },
        "returncode": result.returncode,
    }
    (out_dir / summary_name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert obs_sum > 1000
    assert new_sum / obs_sum > 0.85, f"precip mass {new_sum:.1f} vs observed {obs_sum:.1f}"
    assert zeroed == 0, f"{zeroed} observed wet days were zeroed"


@pytest.mark.slow
def test_kmqe_legacy_fill_preserves_observed_precip_mass() -> None:
    """Full-record ``-id -pm -gf legacy -qi`` must not wipe Autotest.1 precip mass."""
    _run_kmqe_fill_guard("legacy", "KMQE-fill-guard", "MassCompare.json")


@pytest.mark.slow
def test_kmqe_phigi_fill_preserves_observed_precip_mass() -> None:
    """Full-record ``-id -pm -gf phigi -qi`` must not wipe Autotest.1 precip mass."""
    _run_kmqe_fill_guard("phigi", "KMQE-fill-guard-phigi", "MassCompare-phigi.json")
