"""Amputation scenarios for the experiment matrix.

Extends the base helpers in ``validation.py`` with powered short gaps,
season-rotated long gaps, MNAR patterns, and explicit hole masks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from Source.weppcliff_impute.missingness import MissingnessKind
from Source.weppcliff_impute.validation import (
    ScenarioResult,
    WeatherScenario,
    apply_long_gap_all,
    apply_short_gaps,
    apply_variable_absent,
)
from pymice.ampute import ampute

SEASON_MONTHS: dict[str, set[int]] = {
    "DJF": {12, 1, 2},
    "MAM": {3, 4, 5},
    "JJA": {6, 7, 8},
    "SON": {9, 10, 11},
}


class MatrixScenario(str, Enum):
    """Named masks used by experiment matrix A–F."""

    MCAR = "mcar_ampute"
    SHORT_GAPS = "short_gaps"  # legacy thin short gaps
    SHORT_GAPS_POWERED = "short_gaps_powered"
    LONG_GAP_ALL = "long_gap_all"  # fixed window (legacy)
    LONG_GAP_DJF = "long_gap_DJF"
    LONG_GAP_MAM = "long_gap_MAM"
    LONG_GAP_JJA = "long_gap_JJA"
    LONG_GAP_SON = "long_gap_SON"
    VARIABLE_ABSENT_SO_RAD = "variable_absent:SO_RAD"
    VARIABLE_ABSENT_MIN_TEMP = "variable_absent:MIN_TEMP"
    MNAR_COLD = "mnar_cold_sensor"
    PERSISTENT_SEASONAL = "persistent_seasonal"
    BREAKPOINT_MCAR = "breakpoint_mcar"


@dataclass(frozen=True)
class AppliedMask:
    """Amputed panel plus row- and cell-level hole indicators."""

    amputed: pd.DataFrame
    missing_mask: pd.DataFrame
    """Boolean DataFrame (same columns as value_cols) of introduced NAs."""

    hole_day: np.ndarray
    """True for calendar days that contain any introduced missingness."""

    scenario: str
    expected_kind: MissingnessKind
    notes: tuple[str, ...] = ()


def _mask_frame(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(False, index=df.index, columns=value_cols)


def _hole_day_from_mask(missing_mask: pd.DataFrame) -> np.ndarray:
    if missing_mask.empty:
        return np.zeros(0, dtype=bool)
    return missing_mask.to_numpy(dtype=bool).any(axis=1)


def _from_scenario_result(result: ScenarioResult, scenario: str) -> AppliedMask:
    hole = _hole_day_from_mask(result.missing_mask)
    return AppliedMask(
        amputed=result.amputed,
        missing_mask=result.missing_mask,
        hole_day=hole,
        scenario=scenario,
        expected_kind=result.expected_kind,
    )


def apply_mcar_mixed_panel(
    df: pd.DataFrame,
    value_cols: list[str],
    *,
    prop: float = 0.12,
    seed: int = 0,
    complete_cols: tuple[str, ...] | None = None,
) -> AppliedMask:
    """
    MCAR on mixed complete/partial columns (PRECIP/storm often naturally NA on dry days).

    Matches ``autotest1_ampute_impute_analysis.apply_mcar_all_variables`` semantics.
    """
    out = df.copy()
    mask = _mask_frame(df, value_cols)
    complete_set = set(complete_cols or ())
    if not complete_set:
        # Heuristic: columns that are ≥95% observed are "complete"
        for c in value_cols:
            if c in df.columns and df[c].notna().mean() >= 0.95:
                complete_set.add(c)
    complete = [c for c in value_cols if c in complete_set and c in df.columns]
    partial = [c for c in value_cols if c not in complete and c in df.columns]

    if complete:
        matrix = np.array(df[complete].to_numpy(dtype=np.float64), copy=True)
        for j in range(matrix.shape[1]):
            miss = np.isnan(matrix[:, j])
            if miss.any():
                med = np.nanmedian(matrix[:, j])
                matrix[miss, j] = med if np.isfinite(med) else 0.0
        result = ampute(matrix, prop=prop, seed=seed)
        if result.amp is not None:
            out[complete] = result.amp
            mask[complete] = np.isnan(result.amp)

    rng = np.random.default_rng(seed + 17)
    for col in partial:
        observed = df[col].notna().to_numpy()
        idx = np.flatnonzero(observed)
        if len(idx) == 0:
            continue
        n_drop = max(1, int(round(prop * len(idx))))
        drop = rng.choice(idx, size=min(n_drop, len(idx)), replace=False)
        out.loc[drop, col] = np.nan
        mask.loc[drop, col] = True

    return AppliedMask(
        amputed=out,
        missing_mask=mask,
        hole_day=_hole_day_from_mask(mask),
        scenario=MatrixScenario.MCAR.value,
        expected_kind=MissingnessKind.SHORT_GAPS,
        notes=(f"prop={prop}", f"n_complete={len(complete)}", f"n_partial={len(partial)}"),
    )


def apply_powered_short_gaps(
    df: pd.DataFrame,
    value_cols: list[str],
    *,
    target_frac: float = 0.10,
    gap_len: int = 5,
    seed: int = 0,
    min_gaps: int = 20,
) -> AppliedMask:
    """
    Contiguous multi-day holes covering approximately ``target_frac`` of the record.

    Default legacy short_gaps used 10×4 days (~0.6% of a 6940-day panel) which
    under-powered distributional metrics. This targets ≥10% of days by default.
    """
    n = len(df)
    if n == 0:
        empty = _mask_frame(df, value_cols)
        return AppliedMask(
            amputed=df.copy(),
            missing_mask=empty,
            hole_day=np.zeros(0, dtype=bool),
            scenario=MatrixScenario.SHORT_GAPS_POWERED.value,
            expected_kind=MissingnessKind.SHORT_GAPS,
        )
    n_gaps = max(min_gaps, int(np.ceil(target_frac * n / max(gap_len, 1))))
    # Cap so we do not wipe the entire panel
    n_gaps = min(n_gaps, max(1, n // max(gap_len, 1)))
    result = apply_short_gaps(
        df, value_cols, gap_len=gap_len, n_gaps=n_gaps, seed=seed
    )
    applied = _from_scenario_result(result, MatrixScenario.SHORT_GAPS_POWERED.value)
    frac = float(applied.hole_day.mean()) if len(applied.hole_day) else 0.0
    return AppliedMask(
        amputed=applied.amputed,
        missing_mask=applied.missing_mask,
        hole_day=applied.hole_day,
        scenario=applied.scenario,
        expected_kind=applied.expected_kind,
        notes=(
            f"target_frac={target_frac}",
            f"gap_len={gap_len}",
            f"n_gaps={n_gaps}",
            f"realized_day_frac={frac:.4f}",
        ),
    )


def apply_long_gap_season(
    df: pd.DataFrame,
    value_cols: list[str],
    *,
    season: str,
    year: int | None = None,
) -> AppliedMask:
    """
    Mask all variables on days belonging to one meteorological season.

    If ``year`` is None, uses the first full occurrence of that season in the
    panel (preferring an interior year when multiple years exist).
    """
    season = season.upper()
    if season not in SEASON_MONTHS:
        raise ValueError(f"unknown season {season!r}; expected one of {list(SEASON_MONTHS)}")
    out = df.copy()
    mask = _mask_frame(df, value_cols)
    ymd = out["YYYYMMDD"].astype(str)
    years = ymd.str.slice(0, 4).astype(int)
    months = ymd.str.slice(4, 6).astype(int)
    in_season = months.isin(SEASON_MONTHS[season]).to_numpy()

    if year is not None:
        sel = in_season & (years.to_numpy() == int(year))
    else:
        # Prefer second calendar year if available so donors exist on both sides
        uniq = sorted(years.unique().tolist())
        pick_year = uniq[1] if len(uniq) >= 2 else uniq[0]
        # For DJF, winter spans year boundary — use year of January as tag
        if season == "DJF":
            # Select Dec of previous year + Jan/Feb of pick_year when possible
            y_arr = years.to_numpy()
            m_arr = months.to_numpy()
            sel = np.zeros(len(df), dtype=bool)
            for i in range(len(df)):
                if not in_season[i]:
                    continue
                y, m = int(y_arr[i]), int(m_arr[i])
                if m in (1, 2) and y == pick_year:
                    sel[i] = True
                elif m == 12 and y == pick_year - 1:
                    sel[i] = True
                elif m == 12 and pick_year == uniq[0] and y == pick_year:
                    sel[i] = True
            if not sel.any():
                sel = in_season & (y_arr == pick_year)
        else:
            sel = in_season & (years.to_numpy() == pick_year)
        year = int(pick_year)

    out.loc[sel, value_cols] = np.nan
    mask.loc[sel, value_cols] = True
    scen = f"long_gap_{season}"
    return AppliedMask(
        amputed=out,
        missing_mask=mask,
        hole_day=_hole_day_from_mask(mask),
        scenario=scen,
        expected_kind=MissingnessKind.LONG_GAP_ALL,
        notes=(f"season={season}", f"year={year}", f"n_days={int(sel.sum())}"),
    )


def apply_mnar_cold_sensor(
    df: pd.DataFrame,
    value_cols: list[str],
    *,
    temp_col: str = "MIN_TEMP",
    threshold_c: float = -5.0,
    drop_prob: float = 0.85,
    seed: int = 0,
) -> AppliedMask:
    """
    MNAR: temperature/wind sensors fail preferentially on cold mornings.

    Drops selected variables with high probability when ``temp_col < threshold``.
    """
    out = df.copy()
    mask = _mask_frame(df, value_cols)
    if temp_col not in df.columns:
        raise ValueError(f"{temp_col} required for MNAR cold scenario")
    rng = np.random.default_rng(seed)
    cold = (df[temp_col].to_numpy(dtype=float) < threshold_c) & df[temp_col].notna().to_numpy()
    targets = [c for c in value_cols if c in df.columns]
    for i in np.flatnonzero(cold):
        if rng.random() > drop_prob:
            continue
        idx = out.index[i]
        for col in targets:
            if pd.notna(out.at[idx, col]):
                out.at[idx, col] = np.nan
                mask.at[idx, col] = True
    return AppliedMask(
        amputed=out,
        missing_mask=mask,
        hole_day=_hole_day_from_mask(mask),
        scenario=MatrixScenario.MNAR_COLD.value,
        expected_kind=MissingnessKind.SHORT_GAPS,
        notes=(f"threshold_c={threshold_c}", f"drop_prob={drop_prob}", f"n_cold={int(cold.sum())}"),
    )


def apply_breakpoint_mcar(
    df: pd.DataFrame,
    value_cols: list[str],
    *,
    prop: float = 0.12,
    seed: int = 0,
) -> AppliedMask:
    """MCAR on a breakpoint / sub-daily precip panel (observed rows only)."""
    out = df.copy()
    mask = _mask_frame(df, value_cols)
    rng = np.random.default_rng(seed)
    for col in value_cols:
        if col not in df.columns:
            continue
        observed = df[col].notna().to_numpy()
        idx = np.flatnonzero(observed)
        if len(idx) == 0:
            continue
        n_drop = max(1, int(round(prop * len(idx))))
        drop = rng.choice(idx, size=min(n_drop, len(idx)), replace=False)
        out.loc[drop, col] = np.nan
        mask.loc[drop, col] = True
    return AppliedMask(
        amputed=out,
        missing_mask=mask,
        hole_day=_hole_day_from_mask(mask),
        scenario=MatrixScenario.BREAKPOINT_MCAR.value,
        expected_kind=MissingnessKind.SHORT_GAPS,
        notes=(f"prop={prop}",),
    )


def apply_matrix_scenario(
    df: pd.DataFrame,
    scenario: str | MatrixScenario,
    value_cols: list[str],
    *,
    seed: int = 0,
    complete_cols: tuple[str, ...] | None = None,
) -> AppliedMask:
    """Dispatch a matrix scenario name to an amputation function."""
    name = scenario.value if isinstance(scenario, MatrixScenario) else str(scenario)

    if name == MatrixScenario.MCAR.value:
        return apply_mcar_mixed_panel(
            df, value_cols, prop=0.12, seed=seed, complete_cols=complete_cols
        )
    if name == MatrixScenario.SHORT_GAPS.value:
        result = apply_short_gaps(df, value_cols, seed=seed)
        return _from_scenario_result(result, name)
    if name == MatrixScenario.SHORT_GAPS_POWERED.value:
        return apply_powered_short_gaps(df, value_cols, seed=seed)
    if name == MatrixScenario.LONG_GAP_ALL.value:
        result = apply_long_gap_all(df, value_cols)
        return _from_scenario_result(result, name)
    if name.startswith("long_gap_") and name.split("_")[-1] in SEASON_MONTHS:
        season = name.split("_")[-1]
        return apply_long_gap_season(df, value_cols, season=season)
    if name == MatrixScenario.VARIABLE_ABSENT_SO_RAD.value:
        result = apply_variable_absent(df, value_cols, column="SO_RAD")
        return _from_scenario_result(result, name)
    if name == MatrixScenario.VARIABLE_ABSENT_MIN_TEMP.value:
        result = apply_variable_absent(df, value_cols, column="MIN_TEMP")
        return _from_scenario_result(result, name)
    if name == MatrixScenario.MNAR_COLD.value:
        return apply_mnar_cold_sensor(df, value_cols, seed=seed)
    if name == MatrixScenario.PERSISTENT_SEASONAL.value:
        from Source.weppcliff_impute.validation import apply_persistent_seasonal

        result = apply_persistent_seasonal(df, value_cols)
        return _from_scenario_result(result, name)
    if name == MatrixScenario.BREAKPOINT_MCAR.value:
        return apply_breakpoint_mcar(df, value_cols, seed=seed)

    # Fallback: map WeatherScenario enum values
    try:
        ws = WeatherScenario(name)
    except ValueError as exc:
        raise ValueError(f"unknown matrix scenario: {name}") from exc
    from Source.weppcliff_impute.validation import apply_weather_scenario

    result = apply_weather_scenario(df, ws, value_cols, seed=seed)
    return _from_scenario_result(result, name)
