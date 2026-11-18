"""Calendar-month gap vs dry detection for daily EVENT series.

Replaces CLUSTER count limits (abs=7 / rel=0.025). CLUSTER is a donor pool.

Breakpoint daily tables store non-storm days as PRECIP NA, so EVENT is 0 for
both true dry days and gauge gaps. Distinguishing them from the zeros themselves
is circular if P(N=0) is fit to all years.

Rain-process climatology uses years that recorded events (hurdle) plus the
fraction of years that were wet (Beta-Binomial occurrence). A zero year-month
is a gap when other years of that calendar month are usually wet, or when a
Poisson-Gamma fit to positive counts makes N=0 rare, or when it sits in a
zero streak longer than the climatological dry-season run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from Source.phigi.dates import as_yyyymmdd_strings

MISSING_PRECIP_ALPHA = 0.05
MISSING_PRECIP_OCCURRENCE = 0.5
_GAMMA_SHAPE_PRIOR = 1.0
_GAMMA_RATE_PRIOR = 1.0


@dataclass
class MissingPrecipScan:
    """Result of month-of-year gap detection."""

    missing_months: list[int] = field(default_factory=list)
    unusual_months: list[int] = field(default_factory=list)
    dry_calendar_months: set[int] = field(default_factory=set)
    expected_dry_run: int = 0
    p_zero: dict[int, float] = field(default_factory=dict)
    p_wet: dict[int, float] = field(default_factory=dict)
    force_dry_months: list[int] = field(default_factory=list)


def poisson_gamma_p_zero(
    counts: NDArray[np.float64] | list[float] | pd.Series,
    *,
    a0: float = _GAMMA_SHAPE_PRIOR,
    b0: float = _GAMMA_RATE_PRIOR,
) -> float:
    """Posterior predictive P(N=0) under Poisson-Gamma (negative binomial).

    ``λ ~ Gamma(a0, rate=b0)``, ``N|λ ~ Poisson(λ)``. With no years
    (``k=0``) the month is treated as dry (return 1.0) so it is not flagged
    as unusual.
    """
    arr = np.asarray(counts, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    k = int(arr.size)
    if k == 0:
        return 1.0
    a = float(a0) + float(arr.sum())
    b = float(b0) + k
    log_p = a * (np.log(b) - np.log(b + 1.0))
    return float(np.exp(np.clip(log_p, -700.0, 0.0)))


def occurrence_p_wet(
    counts: NDArray[np.float64] | list[float] | pd.Series,
) -> float:
    """Laplace P(month is wet) = (1 + n_wet) / (2 + n_years)."""
    arr = np.asarray(counts, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    k = int(arr.size)
    if k == 0:
        return 0.5
    n_wet = int(np.sum(arr > 0))
    return (1.0 + n_wet) / (2.0 + k)


def expected_dry_season_run(dry_calendar_months: set[int]) -> int:
    """Longest consecutive calendar-month dry run, wrapping Dec–Jan."""
    if not dry_calendar_months:
        return 0
    flags = [m in dry_calendar_months for m in range(1, 13)]
    best = 0
    cur = 0
    for flag in flags + flags:
        if flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(min(best, 12))


def _month_adjacent(a: int, b: int) -> bool:
    if b == a + 1:
        return True
    return a % 100 == 12 and b == (a // 100 + 1) * 100 + 1


def monthly_event_counts(
    yyyymmdd: NDArray[np.str_] | pd.Series | list,
    event: NDArray[np.float64] | pd.Series,
) -> pd.DataFrame:
    """One row per YYYYMM with event count, year, and month-of-year."""
    dates = as_yyyymmdd_strings(yyyymmdd)
    evt = pd.to_numeric(pd.Series(np.asarray(event)), errors="coerce").to_numpy(
        dtype=np.float64
    )
    n = min(len(dates), len(evt))
    dates = dates[:n]
    evt = evt[:n]
    yyyymm = np.zeros(n, dtype=np.int64)
    for i, d in enumerate(dates):
        if isinstance(d, str) and len(d) >= 6 and d[:6].isdigit():
            yyyymm[i] = int(d[:6])
    valid = yyyymm > 0
    df = pd.DataFrame({"YYYYMM": yyyymm[valid], "EVENT": evt[valid]})
    if df.empty:
        return pd.DataFrame(columns=["YYYYMM", "YEAR", "MOY", "EVENT"])
    monthly = df.groupby("YYYYMM", as_index=False)["EVENT"].sum(min_count=1)
    # Leave all-NA months as NA. Filling them with 0 treats gauge holes as dry
    # evidence and can reclassify a wet-season month as dry season.
    monthly["YEAR"] = (monthly["YYYYMM"] // 100).astype(int)
    monthly["MOY"] = (monthly["YYYYMM"] % 100).astype(int)
    return monthly.sort_values("YYYYMM").reset_index(drop=True)


def _zero_streaks(monthly: pd.DataFrame) -> list[list[int]]:
    """Calendar-adjacent runs of zero-event year-months."""
    if monthly.empty:
        return []
    yyyymm = monthly["YYYYMM"].astype(int).tolist()
    counts = monthly["EVENT"].to_numpy(dtype=np.float64)
    streaks: list[list[int]] = []
    current: list[int] = []
    prev: int | None = None
    for ym, n_evt in zip(yyyymm, counts):
        is_zero = (not np.isfinite(n_evt)) or n_evt <= 0
        if is_zero and prev is not None and current and not _month_adjacent(prev, ym):
            streaks.append(current)
            current = []
        if is_zero:
            current.append(int(ym))
        elif current:
            streaks.append(current)
            current = []
        prev = int(ym)
    if current:
        streaks.append(current)
    return streaks


def _is_dry_calendar_month(
    counts: NDArray[np.float64],
    *,
    alpha: float,
    occurrence: float,
) -> bool:
    """True when a typical year has no events and zeros are not rare given λ+."""
    pw = occurrence_p_wet(counts)
    pos = np.asarray(counts, dtype=np.float64)
    pos = pos[np.isfinite(pos) & (pos > 0)]
    ph = poisson_gamma_p_zero(pos)
    return pw <= occurrence and ph >= alpha


def _zero_is_unusual(
    others: NDArray[np.float64],
    *,
    alpha: float,
    occurrence: float,
) -> bool:
    """Gap if other years are usually wet, or N=0 is rare given positive counts."""
    pw = occurrence_p_wet(others)
    pos = np.asarray(others, dtype=np.float64)
    pos = pos[np.isfinite(pos) & (pos > 0)]
    ph = poisson_gamma_p_zero(pos)
    return pw > occurrence or ph < alpha


def detect_missing_precip_months(
    yyyymmdd: NDArray[np.str_] | pd.Series | list,
    event: NDArray[np.float64] | pd.Series,
    *,
    alpha: float = MISSING_PRECIP_ALPHA,
    occurrence: float = MISSING_PRECIP_OCCURRENCE,
    clim_year_min: int | None = None,
    clim_year_max: int | None = None,
) -> MissingPrecipScan:
    """Flag year-months whose zeros look like gauge gaps, not dry season.

    CLUSTER is intentionally unused. Positive-count (hurdle) years estimate
    the rain process; zeros are not treated as dry-season evidence.

    *clim_year_min* / *clim_year_max* pin p_wet, dry-season months, and the
    "other years" comparison to a reference window. Extra years are still
    classified, but they cannot change the climatology or join a zero-streak
    that leaks into the reference window (CCPT 1971-2025 vs Stage A 1980-2023).
    """
    monthly = monthly_event_counts(yyyymmdd, event)
    if monthly.empty:
        return MissingPrecipScan()

    if clim_year_min is None and clim_year_max is None:
        clim = monthly
        extra = monthly.iloc[0:0]
    else:
        ymin = clim_year_min if clim_year_min is not None else int(monthly["YEAR"].min())
        ymax = clim_year_max if clim_year_max is not None else int(monthly["YEAR"].max())
        in_clim = (monthly["YEAR"] >= ymin) & (monthly["YEAR"] <= ymax)
        clim = monthly.loc[in_clim].reset_index(drop=True)
        extra = monthly.loc[~in_clim].reset_index(drop=True)
        if clim.empty:
            clim = monthly
            extra = monthly.iloc[0:0]

    p_zero: dict[int, float] = {}
    p_wet: dict[int, float] = {}
    dry_calendar: set[int] = set()
    by_moy = {
        int(m): g["EVENT"].to_numpy(dtype=np.float64) for m, g in clim.groupby("MOY")
    }
    for moy, counts in by_moy.items():
        pos = counts[np.isfinite(counts) & (counts > 0)]
        p_wet[int(moy)] = occurrence_p_wet(counts)
        p_zero[int(moy)] = poisson_gamma_p_zero(pos)
        if _is_dry_calendar_month(counts, alpha=alpha, occurrence=occurrence):
            dry_calendar.add(int(moy))

    expected = expected_dry_season_run(dry_calendar)

    unusual: list[int] = []
    unusual_set: set[int] = set()
    for _, row in monthly.iterrows():
        evt = float(row["EVENT"])
        moy = int(row["MOY"])
        year = int(row["YEAR"])
        ym = int(row["YYYYMM"])
        others = clim.loc[
            (clim["MOY"] == moy) & (clim["YEAR"] != year), "EVENT"
        ].to_numpy(dtype=np.float64)
        if np.isfinite(evt) and evt > 0:
            continue
        if np.isfinite(evt) and evt <= 0:
            if _zero_is_unusual(others, alpha=alpha, occurrence=occurrence):
                unusual.append(ym)
                unusual_set.add(ym)
            continue
        if p_wet.get(moy, 0.5) > occurrence:
            unusual.append(ym)
            unusual_set.add(ym)

    def _streak_flags(part: pd.DataFrame) -> tuple[list[int], list[int]]:
        missing_part: list[int] = []
        dry_part: list[int] = []
        if part.empty:
            return missing_part, dry_part
        for streak in _zero_streaks(part):
            is_gap = any(ym in unusual_set for ym in streak) or len(streak) > expected
            if is_gap:
                missing_part.extend(streak)
            else:
                dry_part.extend(streak)
        return missing_part, dry_part

    missing, force_dry = _streak_flags(clim)
    extra_missing, extra_dry = _streak_flags(extra)
    missing.extend(extra_missing)
    force_dry.extend(extra_dry)

    return MissingPrecipScan(
        missing_months=missing,
        unusual_months=unusual,
        dry_calendar_months=dry_calendar,
        expected_dry_run=expected,
        p_zero=p_zero,
        p_wet=p_wet,
        force_dry_months=force_dry,
    )


# A Complete 15-min day has 96 slots. Below this fraction of valid slots the
# day is treated as missing (partial/invalid QPCP), not observed dry.
OBSERVED_SLOT_FRACTION = 90.0 / 96.0
# Breakpoint files may have only a few rows on a wet day.
SPARSE_DAY_SLOT_MAX = 48


def snapshot_observed_precip_days(
    precip_df: pd.DataFrame,
    *,
    min_frac: float = OBSERVED_SLOT_FRACTION,
    sparse_slot_max: int = SPARSE_DAY_SLOT_MAX,
) -> set[str]:
    """YYYYMMDD set for days that were actually in the 15-min/breakpoint file.

    Call on slot 1 *before* duration drops all-zero dry days. A regular 15-min
    day must have enough valid (non-NaN) slots; a sparse breakpoint day is
    observed if it has at least one valid value.
    """
    if precip_df is None or precip_df.empty or "DT_1" not in precip_df.columns:
        return set()
    dt = pd.to_datetime(precip_df["DT_1"], errors="coerce")
    ymd = dt.dt.strftime("%Y%m%d")
    if "PRECIP" in precip_df.columns:
        precip = pd.to_numeric(precip_df["PRECIP"], errors="coerce")
    else:
        precip = pd.Series(np.zeros(len(precip_df), dtype=np.float64))
    tmp = pd.DataFrame({"YYYYMMDD": ymd, "valid": precip.notna().to_numpy()})
    tmp = tmp[tmp["YYYYMMDD"].notna() & (tmp["YYYYMMDD"] != "NaT")]
    if tmp.empty:
        return set()
    stats = tmp.groupby("YYYYMMDD", sort=False).agg(
        n=("valid", "size"),
        n_valid=("valid", "sum"),
    )
    observed: set[str] = set()
    for day, row in stats.iterrows():
        n = int(row["n"])
        n_valid = int(row["n_valid"])
        if n <= sparse_slot_max:
            if n_valid >= 1:
                observed.add(str(day))
        elif n_valid / n >= min_frac:
            observed.add(str(day))
    return observed


def assign_daily_event(
    precip: pd.Series,
    yyyymmdd: pd.Series,
    observed_ymd: set[str] | None,
) -> pd.Series:
    """Three-state daily EVENT: 1 wet, 0 observed dry, NA missing.

    When *observed_ymd* is None, NA precip is treated as dry (legacy 0/1).
    """
    p = pd.to_numeric(precip, errors="coerce")
    ymd = as_yyyymmdd_strings(yyyymmdd)
    n = min(len(p), len(ymd))
    p = p.iloc[:n] if hasattr(p, "iloc") else pd.Series(np.asarray(p)[:n])
    ymd = np.asarray(ymd[:n], dtype=object)
    wet = (p.to_numpy(dtype=np.float64) > 0) & np.isfinite(p.to_numpy(dtype=np.float64))
    if not observed_ymd:
        return pd.Series(np.where(wet, 1.0, 0.0), dtype=np.float64)
    obs = np.array([str(d) in observed_ymd for d in ymd], dtype=bool)
    event = np.full(n, np.nan, dtype=np.float64)
    event[obs & wet] = 1.0
    event[obs & ~wet] = 0.0
    return pd.Series(event, dtype=np.float64)
