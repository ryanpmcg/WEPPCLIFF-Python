"""Month-of-year gap vs dry detection (no CLUSTER count limits)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.impute import binarize_filled_event, find_missing_precip  # noqa: E402
from Source.missing_precip import (  # noqa: E402
    detect_missing_precip_months,
    expected_dry_season_run,
    occurrence_p_wet,
    poisson_gamma_p_zero,
)
from Source.phigi.l4_precip_hurdle import compute_precip_constraints  # noqa: E402


def _daily_panel(
    start: str = "2000-01-01",
    end: str = "2009-12-31",
    *,
    dry_months: tuple[int, ...] = (6, 7, 8),
    wet_every: int = 4,
    extra_zero: tuple[str, str] | None = None,
) -> tuple[pd.Series, np.ndarray]:
    dates = pd.date_range(start, end, freq="D")
    event = np.zeros(len(dates), dtype=np.float64)
    for i, day in enumerate(dates):
        if day.month in dry_months:
            event[i] = 0.0
        else:
            event[i] = 1.0 if (day.day % wet_every) == 1 else 0.0
    if extra_zero is not None:
        z0, z1 = pd.Timestamp(extra_zero[0]), pd.Timestamp(extra_zero[1])
        event[(dates >= z0) & (dates <= z1)] = 0.0
    ymd = pd.Series(dates.strftime("%Y%m%d"))
    return ymd, event


def test_poisson_gamma_p_zero_all_zeros_is_high() -> None:
    assert poisson_gamma_p_zero(np.zeros(10)) > 0.8


def test_poisson_gamma_p_zero_wet_month_is_low() -> None:
    assert poisson_gamma_p_zero(np.full(10, 8.0)) < 0.05


def test_poisson_gamma_p_zero_no_years_is_dry() -> None:
    assert poisson_gamma_p_zero(np.array([])) == 1.0


def test_occurrence_p_wet_laplace() -> None:
    assert occurrence_p_wet(np.array([])) == 0.5
    assert occurrence_p_wet(np.zeros(10)) < 0.2
    assert occurrence_p_wet(np.full(10, 3.0)) > 0.8
    mixed = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    assert occurrence_p_wet(mixed) > 0.5


def test_expected_dry_season_run_wraps_year() -> None:
    assert expected_dry_season_run({6, 7, 8}) == 3
    assert expected_dry_season_run({11, 12, 1, 2}) == 4
    assert expected_dry_season_run(set()) == 0
    assert expected_dry_season_run(set(range(1, 13))) == 12


def test_isolated_dry_season_zeros_not_flagged() -> None:
    ymd, event = _daily_panel()
    scan = detect_missing_precip_months(ymd.to_numpy(), event)
    assert {6, 7, 8}.issubset(scan.dry_calendar_months)
    assert scan.expected_dry_run == 3
    assert scan.missing_months == []
    assert scan.unusual_months == []
    jja = [ym for ym in scan.force_dry_months if (ym % 100) in (6, 7, 8)]
    assert len(jja) == 10 * 3


def test_multi_year_outage_including_dry_months_is_flagged() -> None:
    ymd, event = _daily_panel(extra_zero=("2005-01-01", "2006-12-31"))
    scan = detect_missing_precip_months(ymd.to_numpy(), event)
    hole = [ym for ym in scan.missing_months if 200501 <= ym <= 200612]
    assert hole == [year * 100 + month for year in (2005, 2006) for month in range(1, 13)]
    other_jja = [
        ym
        for ym in scan.force_dry_months
        if (ym % 100) in (6, 7, 8) and not (200501 <= ym <= 200612)
    ]
    assert len(other_jja) == 8 * 3


def test_detection_independent_of_cluster_granularity() -> None:
    ymd, event = _daily_panel(extra_zero=("2005-01-01", "2006-12-31"))
    a = detect_missing_precip_months(ymd.to_numpy(), event)
    frame = pd.DataFrame(
        {
            "YYYYMMDD": ymd,
            "EVENT": event,
            "CLUSTER": np.ones(len(ymd)),
        }
    )
    seasonal = np.where(
        ymd.str[4:6].astype(int).isin([12, 1, 2]),
        1,
        np.where(ymd.str[4:6].astype(int).isin([3, 4, 5]), 2, 3),
    )
    b = find_missing_precip(frame)
    frame_s = frame.copy()
    frame_s["CLUSTER"] = seasonal
    c = find_missing_precip(frame_s)
    frame_q = frame.copy()
    frame_q["CLUSTER"] = (
        ymd.str[4:6].astype(int) * 4
        + ((ymd.str[6:8].astype(int) - 1) // 8)
    )
    d = find_missing_precip(frame_q)
    na_b = set(b.loc[b["EVENT"].isna(), "YYYYMM"].astype(int))
    na_c = set(c.loc[c["EVENT"].isna(), "YYYYMM"].astype(int))
    na_d = set(d.loc[d["EVENT"].isna(), "YYYYMM"].astype(int))
    assert na_b == na_c == na_d == set(a.missing_months)


def test_observed_event_preserved_outside_gap_months() -> None:
    ymd, event = _daily_panel(extra_zero=("2005-01-01", "2006-12-31"))
    frame = pd.DataFrame({"YYYYMMDD": ymd, "EVENT": event, "CLUSTER": 1.0})
    out = find_missing_precip(frame)
    missing = set(
        detect_missing_precip_months(ymd.to_numpy(), event).missing_months
    )
    yyyymm = ymd.str[:6].astype(int)
    keep = ~yyyymm.isin(missing)
    assert (out.loc[keep, "EVENT"].to_numpy() == event[keep.to_numpy()]).all()
    assert out.loc[yyyymm.isin(missing), "EVENT"].isna().all()
    wet_keep = keep.to_numpy() & (event == 1.0)
    assert (out.loc[wet_keep, "EVENT"] == 1.0).all()


def test_wet_season_zero_month_is_unusual() -> None:
    ymd, event = _daily_panel(dry_months=(), extra_zero=("2005-01-01", "2005-01-31"))
    scan = detect_missing_precip_months(ymd.to_numpy(), event)
    assert 200501 in scan.unusual_months
    assert 200501 in scan.missing_months
    assert 200502 not in scan.missing_months


def test_clim_window_stops_pre_window_years_from_reclassifying() -> None:
    """Extra wet years before the clim window must not change 1980+ gap flags."""
    ymd, event = _daily_panel(start="1980-01-01", end="1989-12-31", dry_months=())
    # Quiet 1985-03: all EVENT 0 that March.
    dates = pd.to_datetime(ymd, format="%Y%m%d")
    event = event.copy()
    event[(dates.dt.year == 1985) & (dates.dt.month == 3)] = 0.0
    base = detect_missing_precip_months(ymd.to_numpy(), event)

    pre_dates = pd.date_range("1977-01-01", "1979-12-31", freq="D")
    pre_event = np.ones(len(pre_dates), dtype=np.float64)
    pre_ymd = pd.Series(pre_dates.strftime("%Y%m%d"))
    all_ymd = pd.concat([pre_ymd, ymd], ignore_index=True)
    all_event = np.concatenate([pre_event, event])
    leaked = detect_missing_precip_months(all_ymd.to_numpy(), all_event)
    pinned = detect_missing_precip_months(
        all_ymd.to_numpy(),
        all_event,
        clim_year_min=1980,
        clim_year_max=1989,
    )
    base_1980 = {ym for ym in base.missing_months if ym // 100 >= 1980}
    leaked_1980 = {ym for ym in leaked.missing_months if ym // 100 >= 1980}
    pinned_1980 = {ym for ym in pinned.missing_months if ym // 100 >= 1980}
    assert pinned_1980 == base_1980
    # Without a clim window, 1977-1979 (all wet) can change 1980+ flags.
    assert leaked_1980 != pinned_1980 or leaked.p_wet.get(3, 0) != pinned.p_wet.get(3, 0)


def test_gap_zeros_do_not_define_dry_climatology() -> None:
    """Zeros mixed into a month that usually rains are gaps, not dry season.

    Poisson-Gamma on all years (including the gaps) leaves P(N=0) too high
    when wet years only have a few events. Occurrence uses P(month is wet).
    """
    dates = pd.date_range("2000-01-01", "2009-12-31", freq="D")
    event = np.zeros(len(dates), dtype=np.float64)
    for i, day in enumerate(dates):
        if day.month == 1 and day.year in {2001, 2002, 2003, 2004, 2005, 2006}:
            event[i] = 1.0 if day.day in (3, 10, 18) else 0.0
        elif day.month != 1:
            event[i] = 1.0 if (day.day % 4) == 1 else 0.0
    ymd = dates.strftime("%Y%m%d")
    scan = detect_missing_precip_months(ymd, event)
    jan_zeros = [200001, 200701, 200801, 200901]
    assert scan.p_wet[1] > 0.5
    for ym in jan_zeros:
        assert ym in scan.missing_months, f"{ym} should be a gap"
    assert 1 not in scan.dry_calendar_months


def test_binarize_keeps_observed_after_month_fill() -> None:
    observed = pd.Series([1.0, 0.0, np.nan, 1.0])
    filled = pd.Series([0.1, 0.9, 0.7, 0.2])
    out = binarize_filled_event(filled, observed)
    assert out.tolist() == [1.0, 0.0, 1.0, 1.0]


def test_phigi_constraints_match_calendar_scan() -> None:
    ymd, event = _daily_panel(extra_zero=("2005-01-01", "2006-12-31"))
    cluster = np.ones(len(ymd), dtype=float)
    cluster[:400] = 99.0
    cons = compute_precip_constraints(ymd.to_numpy(), cluster, event)
    scan = detect_missing_precip_months(ymd.to_numpy(), event)
    assert cons.dry_clusters == set()
    assert set(cons.missing_months) == set(scan.missing_months)
    assert cons.force_dry.sum() == ymd.str[:6].astype(int).isin(scan.force_dry_months).sum()


def test_all_na_months_do_not_count_as_dry_climatology() -> None:
    """Missing Januarys must not pull P(wet) down until January looks like dry season."""
    dates = pd.date_range("2000-01-01", "2009-12-31", freq="D")
    event = np.zeros(len(dates), dtype=np.float64)
    for i, day in enumerate(dates):
        if day.month == 1 and day.year in {2000, 2001, 2002, 2003}:
            event[i] = 1.0 if day.day in (3, 10, 18) else 0.0
        elif day.month == 1:
            event[i] = np.nan
        else:
            event[i] = 1.0 if (day.day % 4) == 1 else 0.0
    ymd = dates.strftime("%Y%m%d")
    scan = detect_missing_precip_months(ymd, event)
    assert scan.p_wet[1] > 0.5
    assert 1 not in scan.dry_calendar_months
    for ym in (200501, 200601, 200701, 200801, 200901):
        assert ym in scan.missing_months, f"{ym} should be a wet-season hole"


def test_three_state_preserves_observed_and_fills_wet_holes() -> None:
    """EVENT 0/1 stay; NA in wet-season months stays NA; NA in dry season becomes 0."""
    ymd, event = _daily_panel()
    dates = pd.to_datetime(ymd, format="%Y%m%d")
    event = event.copy()
    jan_hole = (dates.dt.year == 2005) & (dates.dt.month == 1)
    event[jan_hole.to_numpy()] = np.nan
    jja_missing = (dates.dt.year == 2004) & (dates.dt.month == 7) & (dates.dt.day <= 10)
    event[jja_missing.to_numpy()] = np.nan
    frame = pd.DataFrame({"YYYYMMDD": ymd, "EVENT": event, "CLUSTER": 1.0})
    out = find_missing_precip(frame)
    jan = (dates.dt.year == 2005) & (dates.dt.month == 1)
    assert out.loc[jan.to_numpy(), "EVENT"].isna().all()
    jja = (dates.dt.year == 2004) & (dates.dt.month == 7) & (dates.dt.day <= 10)
    assert (out.loc[jja.to_numpy(), "EVENT"] == 0.0).all()
    observed = np.isfinite(event)
    assert (out.loc[observed, "EVENT"].to_numpy() == event[observed]).all()


def test_partial_month_fills_missing_days_not_observed_storms() -> None:
    """Two observed wet days must stay wet; the rest of a gappy wet month is fillable."""
    dates = pd.date_range("2000-01-01", "2009-12-31", freq="D")
    event = np.zeros(len(dates), dtype=np.float64)
    for i, day in enumerate(dates):
        if day.month == 1:
            if day.day in (3, 10):
                event[i] = 1.0
            elif day.year == 2005 and day.day >= 11:
                event[i] = np.nan
            else:
                event[i] = 0.0
        else:
            event[i] = 1.0 if (day.day % 4) == 1 else 0.0
    ymd = pd.Series(dates.strftime("%Y%m%d"))
    frame = pd.DataFrame({"YYYYMMDD": ymd, "EVENT": event, "CLUSTER": 1.0})
    out = find_missing_precip(frame)
    y2005jan = (dates.year == 2005) & (dates.month == 1)
    storm_days = y2005jan & dates.day.isin([3, 10])
    hole_days = y2005jan & (dates.day >= 11)
    dry_obs = y2005jan & ~dates.day.isin([3, 10]) & (dates.day < 11)
    assert (out.loc[np.asarray(storm_days), "EVENT"] == 1.0).all()
    assert (out.loc[np.asarray(dry_obs), "EVENT"] == 0.0).all()
    assert out.loc[np.asarray(hole_days), "EVENT"].isna().all()


def test_assign_daily_event_three_state() -> None:
    from Source.missing_precip import assign_daily_event

    ymd = pd.Series(["20000101", "20000102", "20000103"])
    precip = pd.Series([5.0, np.nan, np.nan])
    observed = {"20000101", "20000102"}
    event = assign_daily_event(precip, ymd, observed)
    assert event.tolist()[0] == 1.0
    assert event.tolist()[1] == 0.0
    assert np.isnan(event.tolist()[2])


def test_snapshot_observed_precip_days_requires_enough_valid_slots() -> None:
    from Source.missing_precip import snapshot_observed_precip_days

    rows = []
    # Full dry day: 96 zeros.
    for minute in range(0, 1440, 15):
        rows.append(
            {
                "DT_1": pd.Timestamp("2000-01-01") + pd.Timedelta(minutes=minute),
                "PRECIP": 0.0,
            }
        )
    # Mostly invalid day: 10 valid zeros, rest NA.
    for minute in range(0, 1440, 15):
        rows.append(
            {
                "DT_1": pd.Timestamp("2000-01-02") + pd.Timedelta(minutes=minute),
                "PRECIP": 0.0 if minute < 150 else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    observed = snapshot_observed_precip_days(df)
    assert "20000101" in observed
    assert "20000102" not in observed
