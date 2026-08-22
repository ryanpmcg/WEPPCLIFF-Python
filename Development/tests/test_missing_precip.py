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
