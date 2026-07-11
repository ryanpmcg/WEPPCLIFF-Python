"""Fair PHIGI vs legacy comparison on real gap-fill pipelines (KMQE)."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Source.config import WeppcliffConfig, get_config, set_config
from Source.data_model import WeppcliffData
from Source.phigi.dates import as_yyyymmdd_strings, coerce_yyyymmdd_series
from Source.phigi.holdout_metrics import (
    event_brier_score,
    monthly_precip_mass_bias,
    rmse,
)
from Source.phigi.l4_precip_hurdle import score_precip_holdout
from Source.workflow import fill_workflow
from Source.weppcliff_impute.ei_validate import i30_proxy

DAILY_VARS = ("MIN_TEMP", "MAX_TEMP", "SO_RAD", "W_VEL", "W_DIR", "DP_TEMP")
PRECIP_DAILY_VARS = ("PRECIP", "BPS", "ELEN")
ALL_DAILY_SCORE_VARS = DAILY_VARS + PRECIP_DAILY_VARS


class CompareScenario(str, Enum):
    HOLDOUT_RANDOM = "holdout_random"
    SORAD_CORRUPT_SUMMER = "sorad_corrupt_summer"
    SEASONAL_GAP = "seasonal_gap"
    SPARSE_DONORS = "sparse_donors"


@dataclass
class HoldoutPlan:
    """Days and variables targeted by a comparison scenario."""

    scenario: CompareScenario
    holdout_ymd: set[str]
    corrupt_ymd: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)


@dataclass
class TruthSnapshot:
    """Observed values before artificial masking or corruption."""

    daily: pd.DataFrame
    precip: pd.DataFrame

    def daily_on(self, ymd: str) -> pd.Series | None:
        row = self.daily.loc[self.daily["YYYYMMDD"] == ymd]
        return None if row.empty else row.iloc[0]


@dataclass
class EngineScore:
    engine: str
    daily_rmse: dict[str, float]
    temp_order_violations: int
    precip: dict[str, float]
    i30_mean_rel_bias: float
    ei_mean_rel_bias: float
    n_holdout_days: int
    n_wet_holdout_days: int
    notes: list[str] = field(default_factory=list)


@dataclass
class FairCompareResult:
    scenario: str
    phigi: EngineScore
    legacy: EngineScore
    winners: dict[str, str]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        def _score_dict(s: EngineScore) -> dict[str, Any]:
            return {
                "engine": s.engine,
                "daily_rmse": s.daily_rmse,
                "temp_order_violations": s.temp_order_violations,
                "precip": s.precip,
                "i30_mean_rel_bias": s.i30_mean_rel_bias,
                "ei_mean_rel_bias": s.ei_mean_rel_bias,
                "n_holdout_days": s.n_holdout_days,
                "n_wet_holdout_days": s.n_wet_holdout_days,
                "notes": s.notes,
            }

        return {
            "scenario": self.scenario,
            "phigi": _score_dict(self.phigi),
            "legacy": _score_dict(self.legacy),
            "winners": self.winners,
            "notes": self.notes,
        }


def deep_copy_data(data: WeppcliffData) -> WeppcliffData:
    """Deep copy DataFrame slots and storm lists."""
    out = data.deep_copy()
    for key in list(out.keys()):
        val = out[key]
        if isinstance(val, list):
            out[key] = [x.copy() if isinstance(x, pd.DataFrame) else copy.copy(x) for x in val]
    return out


def capture_truth_snapshot(data: WeppcliffData) -> TruthSnapshot:
    daily = data.slot_df(10).copy()
    daily["YYYYMMDD"] = coerce_yyyymmdd_series(daily["YYYYMMDD"])
    precip = data.slot_df(5).copy()
    return TruthSnapshot(daily=daily, precip=precip)


def build_pre_gapfill_data(
    input_path: Path,
    config: WeppcliffConfig,
) -> WeppcliffData:
    """Run the real workflow up to (but not including) ``fill_workflow``."""
    from Source.daily import annual_summary, process_daily_data
    from Source.duration import create_duration_data
    from Source.erosion import calculate_erosion_indices
    from Source.io import check_input_format, load_input_file
    from Source.preprocess import (
        convert_data,
        preserve_missing_data,
        preprocess_file,
        remove_missing_datetime_rows,
        store_data,
    )
    from Source.precip import aggregate_precip_data, process_precip_data
    from Source.quality import initialize_qc_data, quality_check_inputs, quality_check_outputs
    from Source.storms import break_storms, trim_data
    from Source.units import convert_units
    from Source.workflow import _load_and_preprocess

    set_config(config)
    config.apply_runtime_flags()
    config.do_id = True
    config.id = "t"

    config.f = input_path.name
    if not config.d:
        config.d = str(input_path.parent)
    file = _load_and_preprocess(config.f, config)

    x = [1, 3]
    data = store_data(file, config)
    data = convert_data(data, config)
    data = remove_missing_datetime_rows(data, config)
    data = preserve_missing_data(data, x, config)
    data = create_duration_data(data, x, config)
    if config.do_ai:
        data = aggregate_precip_data(data, config)
    data = convert_units(data, config)
    data = initialize_qc_data(data, config)
    if config.do_qc:
        data = quality_check_inputs(data, config)
    data = trim_data(data, 5, 6, 7, 8, config)
    if config.do_ei:
        data = calculate_erosion_indices(data, 8, 9, config)
    if not config.cv_off:
        data = process_daily_data(data, 7, 6, 10, 11, 12, 13, config)
    data = process_precip_data(data, 5, 10, 11, 12, 13, config)
    if config.do_qc:
        data = quality_check_outputs(data, 10, config)
    data = annual_summary(data, 12, 14, config)
    data = preserve_missing_data(data, [13], config)
    return data


def _complete_day_mask(daily: pd.DataFrame) -> np.ndarray:
    cols = [c for c in ALL_DAILY_SCORE_VARS if c in daily.columns]
    return daily[cols].notna().all(axis=1).to_numpy(dtype=bool)


def build_holdout_plan(
    truth: TruthSnapshot,
    scenario: CompareScenario,
    *,
    fraction: float = 0.10,
    seed: int = 0,
    gap_days: int = 60,
) -> HoldoutPlan:
    daily = truth.daily
    ymd = coerce_yyyymmdd_series(daily["YYYYMMDD"])
    complete = _complete_day_mask(daily)
    eligible = ymd[complete].tolist()
    rng = np.random.default_rng(seed)

    if scenario == CompareScenario.HOLDOUT_RANDOM:
        n = max(5, int(len(eligible) * fraction))
        pick = set(rng.choice(eligible, size=min(n, len(eligible)), replace=False).tolist())
        return HoldoutPlan(scenario=scenario, holdout_ymd=pick, notes=[f"random {n} complete days"])

    if scenario == CompareScenario.SEASONAL_GAP:
        dt = pd.to_datetime(ymd, format="%Y%m%d", errors="coerce")
        summer = dt.dt.month.isin([6, 7, 8]).to_numpy() & complete
        idx = np.where(summer)[0]
        if len(idx) < gap_days:
            start = max(0, len(eligible) // 3)
            pick = set(ymd.iloc[start : start + gap_days].tolist())
        else:
            start = int(rng.integers(0, max(len(idx) - gap_days, 1)))
            pick = set(ymd.iloc[idx[start] : idx[start] + gap_days].tolist())
        return HoldoutPlan(
            scenario=scenario,
            holdout_ymd=pick,
            notes=[f"{gap_days}-day contiguous summer gap"],
        )

    if scenario == CompareScenario.SPARSE_DONORS:
        month = pd.to_datetime(ymd, format="%Y%m%d", errors="coerce").dt.month
        target_month = int(rng.choice([1, 2, 7, 8]))
        in_month = (month == target_month).to_numpy() & complete
        pool = ymd[in_month].tolist()
        n = max(10, int(len(pool) * min(fraction * 2.5, 0.5)))
        pick = set(rng.choice(pool, size=min(n, len(pool)), replace=False).tolist())
        return HoldoutPlan(
            scenario=scenario,
            holdout_ymd=pick,
            notes=[f"sparse mask in month {target_month} ({n} days)"],
        )

    if scenario == CompareScenario.SORAD_CORRUPT_SUMMER:
        dt = pd.to_datetime(ymd, format="%Y%m%d", errors="coerce")
        summer = dt.dt.month.isin([6, 7, 8]).to_numpy() & complete
        corrupt = set(ymd[summer].tolist())
        n_score = max(5, int(len(corrupt) * fraction))
        score_days = set(rng.choice(sorted(corrupt), size=min(n_score, len(corrupt)), replace=False).tolist())
        return HoldoutPlan(
            scenario=scenario,
            holdout_ymd=score_days,
            corrupt_ymd=corrupt,
            notes=[f"corrupt SO_RAD x0.1 on {len(corrupt)} summer days; score {n_score}"],
        )

    raise ValueError(f"unknown scenario: {scenario}")


def apply_holdout_plan(data: WeppcliffData, plan: HoldoutPlan) -> None:
    """Mask or corrupt slots 5 and 10 in place."""
    daily = data.slot_df(10).copy()
    daily["YYYYMMDD"] = coerce_yyyymmdd_series(daily["YYYYMMDD"])
    precip = data.slot_df(5).copy()
    pcp_ymd = pd.to_datetime(precip["DT_1"], errors="coerce").dt.strftime("%Y%m%d")

    if plan.corrupt_ymd:
        corrupt = daily["YYYYMMDD"].isin(plan.corrupt_ymd)
        if "SO_RAD" in daily.columns:
            daily.loc[corrupt, "SO_RAD"] = pd.to_numeric(daily.loc[corrupt, "SO_RAD"], errors="coerce") * 0.1

    if plan.holdout_ymd:
        hold = daily["YYYYMMDD"].isin(plan.holdout_ymd)
        for col in ALL_DAILY_SCORE_VARS:
            if col in daily.columns:
                daily.loc[hold, col] = np.nan
        pcp_hold = pcp_ymd.isin(plan.holdout_ymd)
        if "PRECIP" in precip.columns:
            precip.loc[pcp_hold, "PRECIP"] = np.nan

    data[10] = daily
    data[5] = precip


def run_gapfill_engine(data: WeppcliffData, *, gf: str) -> WeppcliffData:
    """Execute ``fill_workflow`` with ``-gf phigi`` or ``-gf legacy``."""
    config = get_config()
    config.gf = gf
    config.apply_runtime_flags()
    config.phigi_report = None
    set_config(config)
    return fill_workflow(data, config)


def _daily_after_fill(data: WeppcliffData) -> pd.DataFrame:
    dly = data.slot_df(18).copy()
    dly["YYYYMMDD"] = coerce_yyyymmdd_series(dly["YYYYMMDD"])
    return dly


def _precip_after_fill(data: WeppcliffData) -> pd.DataFrame:
    pcp = data.slot_df(15).copy()
    pcp["YYYYMMDD"] = pd.to_datetime(pcp["DT_1"], errors="coerce").dt.strftime("%Y%m%d")
    return pcp


def _temp_order_violations(dly: pd.DataFrame, ymd_set: set[str]) -> int:
    if "MIN_TEMP" not in dly.columns or "MAX_TEMP" not in dly.columns:
        return 0
    sub = dly[dly["YYYYMMDD"].isin(ymd_set)]
    if sub.empty:
        return 0
    bad = sub["MIN_TEMP"] > sub["MAX_TEMP"]
    return int(bad.sum())


def _wet_holdout_days(truth: TruthSnapshot, ymd_set: set[str]) -> list[str]:
    wet: list[str] = []
    for ymd in ymd_set:
        row = truth.daily_on(ymd)
        if row is None:
            continue
        precip = float(row.get("PRECIP", 0) or 0)
        if precip > 0:
            wet.append(ymd)
    return wet


def _precip_day_frame(pcp: pd.DataFrame, ymd: str) -> pd.DataFrame:
    frame = pcp.copy()
    frame["YYYYMMDD"] = as_yyyymmdd_strings(
        pd.to_datetime(frame["DT_1"], errors="coerce").dt.strftime("%Y%m%d")
    )
    rows = frame.loc[frame["YYYYMMDD"] == ymd, ["DT_1", "DUR", "PRECIP"]].copy()
    if "DUR" in rows.columns:
        rows["DUR"] = pd.to_numeric(rows["DUR"], errors="coerce").fillna(15.0)
    rows["PRECIP"] = pd.to_numeric(rows["PRECIP"], errors="coerce").fillna(0.0)
    return rows.dropna(subset=["DT_1"])


def _score_i30_ei(
    truth: TruthSnapshot,
    filled: WeppcliffData,
    wet_days: list[str],
) -> tuple[float, float, list[str]]:
    """Mean relative I30 and daily precip-mass bias on wet holdout days."""
    imputed_pcp = _precip_after_fill(filled)
    notes: list[str] = []
    i30_biases: list[float] = []
    ei_biases: list[float] = []

    for ymd in wet_days:
        t_rows = _precip_day_frame(truth.precip, ymd)
        i_rows = _precip_day_frame(imputed_pcp, ymd)
        if t_rows.empty or i_rows.empty:
            continue
        t_i30 = i30_proxy(t_rows)
        i_i30 = i30_proxy(i_rows)
        if t_i30 > 1e-9:
            i30_biases.append(abs(i_i30 - t_i30) / t_i30)

        t_ke = float(t_rows["PRECIP"].sum())
        i_ke = float(i_rows["PRECIP"].sum())
        if t_ke > 1e-9:
            ei_biases.append(abs(i_ke - t_ke) / t_ke)

    if not i30_biases:
        notes.append("no wet holdout days with I30 pairs")
    if not ei_biases:
        notes.append("no wet holdout days with precip-mass pairs")

    i30_bias = float(np.mean(i30_biases)) if i30_biases else float("nan")
    ei_bias = float(np.mean(ei_biases)) if ei_biases else float("nan")
    return i30_bias, ei_bias, notes


def score_engine(
    truth: TruthSnapshot,
    filled: WeppcliffData,
    plan: HoldoutPlan,
    *,
    engine: str,
) -> EngineScore:
    dly = _daily_after_fill(filled)
    score_days = plan.holdout_ymd if plan.scenario != CompareScenario.SORAD_CORRUPT_SUMMER else plan.corrupt_ymd
    ymd_list = sorted(score_days)
    truth_sub = truth.daily[truth.daily["YYYYMMDD"].isin(ymd_list)].copy()
    fill_sub = dly[dly["YYYYMMDD"].isin(ymd_list)].copy()
    daily_rmse: dict[str, float] = {}
    for var in ALL_DAILY_SCORE_VARS:
        if var not in truth_sub.columns or var not in fill_sub.columns:
            continue
        merged = truth_sub[["YYYYMMDD", var]].merge(
            fill_sub[["YYYYMMDD", var]],
            on="YYYYMMDD",
            suffixes=("_t", "_f"),
        )
        tvals = pd.to_numeric(merged[f"{var}_t"], errors="coerce").to_numpy(dtype=np.float64)
        fvals = pd.to_numeric(merged[f"{var}_f"], errors="coerce").to_numpy(dtype=np.float64)
        daily_rmse[var] = rmse(tvals, fvals)

    wet_days = _wet_holdout_days(truth, plan.holdout_ymd)
    precip_cmp = truth.daily.loc[truth.daily["YYYYMMDD"].isin(ymd_list), ["YYYYMMDD", "PRECIP"]].merge(
        dly.loc[dly["YYYYMMDD"].isin(ymd_list), ["YYYYMMDD", "PRECIP", "BPS", "ELEN"]],
        on="YYYYMMDD",
        suffixes=("_t", "_f"),
        how="inner",
    )
    actual_precip = pd.to_numeric(precip_cmp["PRECIP_t"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    predicted_precip = pd.to_numeric(precip_cmp["PRECIP_f"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
    event_t = (actual_precip > 0).astype(float)
    event_p_bin = (predicted_precip > 0).astype(float)
    precip_scores = score_precip_holdout(
        dates=np.asarray(precip_cmp["YYYYMMDD"].astype(str), dtype=str),
        actual_event=event_t,
        predicted_event=event_p_bin,
        predicted_prob=np.clip(event_p_bin, 0.02, 0.98),
        actual_precip=actual_precip,
        predicted_precip=predicted_precip,
        holdout_mask=np.ones(len(event_t), dtype=bool),
    )

    i30_bias, ei_bias, er_notes = _score_i30_ei(truth, filled, wet_days)

    return EngineScore(
        engine=engine,
        daily_rmse=daily_rmse,
        temp_order_violations=_temp_order_violations(dly, score_days),
        precip=precip_scores,
        i30_mean_rel_bias=i30_bias,
        ei_mean_rel_bias=ei_bias,
        n_holdout_days=len(ymd_list),
        n_wet_holdout_days=len(wet_days),
        notes=er_notes,
    )


def _pick_winner(phigi_val: float, legacy_val: float, *, lower_is_better: bool = True) -> str:
    if not np.isfinite(phigi_val) or not np.isfinite(legacy_val):
        return "tie"
    if abs(phigi_val - legacy_val) < 1e-9:
        return "tie"
    if lower_is_better:
        return "phigi" if phigi_val < legacy_val else "legacy"
    return "phigi" if phigi_val > legacy_val else "legacy"


def compare_engines_on_kmqe(
    input_path: Path,
    scenario: CompareScenario,
    *,
    fraction: float = 0.10,
    seed: int = 0,
    config: WeppcliffConfig | None = None,
) -> FairCompareResult:
    """Run full ``fill_workflow`` for PHIGI and legacy on the same masked KMQE record."""
    from Source.args import assign_empty_args

    cfg = config or WeppcliffConfig(
        f=input_path.name,
        id="t",
        qc="t",
        pd="f",
        verb="f",
        la="y",
        qi="t",
        pm="t",
        ed="0",
        gf="legacy",
        d=str(input_path.parent),
    )
    repo_root = input_path.resolve().parents[1] if input_path.parent.name == "Input" else input_path.parent
    cfg.home_dir = str(repo_root)
    assign_empty_args(cfg)

    base = build_pre_gapfill_data(input_path, cfg)
    truth = capture_truth_snapshot(base)
    plan = build_holdout_plan(truth, scenario, fraction=fraction, seed=seed)

    phigi_data = deep_copy_data(base)
    legacy_data = deep_copy_data(base)
    apply_holdout_plan(phigi_data, plan)
    apply_holdout_plan(legacy_data, plan)

    phigi_out = run_gapfill_engine(phigi_data, gf="phigi")
    legacy_out = run_gapfill_engine(legacy_data, gf="legacy")

    phigi_score = score_engine(truth, phigi_out, plan, engine="phigi")
    legacy_score = score_engine(truth, legacy_out, plan, engine="legacy")

    winners: dict[str, str] = {}
    for var in ALL_DAILY_SCORE_VARS:
        if var in phigi_score.daily_rmse and var in legacy_score.daily_rmse:
            winners[f"rmse_{var.lower()}"] = _pick_winner(
                phigi_score.daily_rmse[var],
                legacy_score.daily_rmse[var],
            )
    winners["temp_order"] = _pick_winner(
        float(phigi_score.temp_order_violations),
        float(legacy_score.temp_order_violations),
    )
    if phigi_score.precip and legacy_score.precip:
        for key in ("event_brier", "monthly_precip_mass_bias"):
            if key in phigi_score.precip and key in legacy_score.precip:
                winners[key] = _pick_winner(phigi_score.precip[key], legacy_score.precip[key])
    winners["i30_bias"] = _pick_winner(phigi_score.i30_mean_rel_bias, legacy_score.i30_mean_rel_bias)
    winners["ei_bias"] = _pick_winner(phigi_score.ei_mean_rel_bias, legacy_score.ei_mean_rel_bias)

    return FairCompareResult(
        scenario=scenario.value,
        phigi=phigi_score,
        legacy=legacy_score,
        winners=winners,
        notes=list(plan.notes),
    )


def compare_all_scenarios(
    input_path: Path,
    *,
    fraction: float = 0.10,
    seed: int = 0,
    config: WeppcliffConfig | None = None,
) -> list[FairCompareResult]:
    return [
        compare_engines_on_kmqe(
            input_path,
            scenario,
            fraction=fraction,
            seed=seed,
            config=config,
        )
        for scenario in CompareScenario
    ]