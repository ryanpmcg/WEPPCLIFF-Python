"""L5 storm-template sub-daily imputation with parametric fallback."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from Source.phigi.l1_priors import load_default_priors
from Source.phigi.l5_ei_validate import (
    EIValidationResult,
    select_best_i30_candidate,
    validate_i30_against_donors,
)
from Source.weppcliff_impute.adapter import ImputeConfig, _config_from_any, _run_mice_on_frame
from Source.weppcliff_impute.overlap import resolve_overlaps, trim_overlap_durations
from Source.weppcliff_impute.storm_template import (
    apply_template_to_day,
    cluster_median_start,
    extract_day_templates,
    select_donor_template,
)

if TYPE_CHECKING:
    from Source.phigi.report import ImputationReport

_DEFAULT_START_MINUTE = 720.0
_MIN_LOCAL_TEMPLATES = 5
_K_CANDIDATES = 5


def _pool_templates_for_cluster(
    templates: list,
    *,
    cluster: int,
    kg_zone: str | None = None,
) -> list:
    """
    L5.2: expand donor pool to same-season components when local templates are sparse.

    When fewer than ``_MIN_LOCAL_TEMPLATES`` exist for ``cluster``, include templates
    from neighboring weather-type components (same season bucket).
    """
    from Source.weppcliff_impute.storm_template import DayTemplate

    pool = [t for t in templates if isinstance(t, DayTemplate) and t.cluster == cluster]
    if len(pool) >= _MIN_LOCAL_TEMPLATES:
        return pool

    season = int(cluster) // 10
    season_pool = [t for t in templates if isinstance(t, DayTemplate) and int(t.cluster) // 10 == season]
    if len(season_pool) >= _MIN_LOCAL_TEMPLATES:
        if kg_zone and "kg_pooled_templates" not in str(kg_zone):
            pass
        return season_pool
    return templates


def generate_parametric_day(
    *,
    yyyymmdd: str,
    cluster: int,
    bps: int,
    daily_precip: float,
    elen: int = 1,
    start_minute: float = _DEFAULT_START_MINUTE,
) -> pd.DataFrame:
    """
    L5.3 parametric fallback: 1–2 pulse uniform storm when template donors absent.

    Scales breakpoint depths to ``daily_precip``; preserves timing fractions.
    """
    n = max(1, int(bps))
    base = pd.to_datetime(str(yyyymmdd), format="%Y%m%d")
    span_min = float(max(int(elen), 1) * 15.0)
    if n == 1:
        fracs = np.array([1.0], dtype=np.float64)
        offsets = np.array([0.0], dtype=np.float64)
        dur = np.array([max(span_min, 1.0)], dtype=np.float64)
    elif n == 2:
        fracs = np.array([0.7, 0.3], dtype=np.float64)
        offsets = np.array([0.0, span_min * 0.35], dtype=np.float64)
        dur = np.array([span_min * 0.4, span_min * 0.35], dtype=np.float64)
    else:
        fracs = np.ones(n, dtype=np.float64) / n
        offsets = np.linspace(0.0, span_min * 0.75, n)
        dur = np.full(n, span_min / n, dtype=np.float64)

    total = float(max(daily_precip, 0.0))
    rows: list[dict] = []
    for off, d, frac in zip(offsets, dur, fracs, strict=True):
        t = base + pd.Timedelta(minutes=float(start_minute + off))
        rows.append(
            {
                "DT_1": t,
                "YYYYMMDD": str(yyyymmdd),
                "CLUSTER": int(cluster),
                "DUR": float(max(d, 0.1)),
                "PRECIP": float(frac * total),
                "HOUR": t.hour,
                "MINUTE": t.minute,
                "SECOND": t.second,
            }
        )
    return pd.DataFrame(rows)


def impute_phigi_subdaily(
    imp_df: pd.DataFrame,
    *,
    observed_ts: pd.DataFrame,
    daily_events: pd.DataFrame,
    config: ImputeConfig | object,
    imp_method: str = "midastouch",
    report: ImputationReport | None = None,
) -> tuple[pd.DataFrame, list[str], list[EIValidationResult], int]:
    """
    L5 orchestrator: template donors → parametric pulse → PyMICE FCS fallback.

    Returns filled dataframe, strategy tags, EI validation results, and day count.
    """
    from Source.phigi.report import ImputationReport

    rep = report or ImputationReport()
    if "L5" not in rep.layers_used:
        rep.layers_used.append("L5")

    cfg = _config_from_any(config)
    strategies: list[str] = []
    ei_results: list[EIValidationResult] = []
    work = imp_df.copy()
    value_cols = [c for c in ("DUR", "PRECIP", "HOUR", "MINUTE", "SECOND") if c in work.columns]
    if not value_cols:
        return work, strategies, ei_results, 0

    templates = extract_day_templates(observed_ts)
    prior_store = load_default_priors()
    kg_zone = prior_store.resolve_zone(
        str(getattr(cfg, "kg", "AUTO")),
        float(getattr(cfg, "lat", 45.0)),
        float(getattr(cfg, "lon", 0.0)),
    )
    rng = np.random.default_rng(cfg.seed)
    dates = daily_events.copy()
    dates["YYYYMMDD"] = dates["YYYYMMDD"].astype(str)

    if "YYYYMMDD" not in work.columns and "DT_1" in work.columns:
        work["YYYYMMDD"] = pd.to_datetime(work["DT_1"]).dt.strftime("%Y%m%d")

    n_days = 0
    out_parts: list[pd.DataFrame] = []

    for ymd, day_rows in work.groupby(work["YYYYMMDD"].astype(str), sort=False):
        day_rows = day_rows.copy()
        needs_fill = day_rows[value_cols].isna().any(axis=1)
        if not needs_fill.any():
            out_parts.append(day_rows)
            continue

        meta = dates.loc[dates["YYYYMMDD"] == str(ymd)]
        if meta.empty:
            out_parts.append(day_rows)
            continue

        meta_row = meta.iloc[0]
        cluster = int(meta_row.get("CLUSTER", day_rows["CLUSTER"].iloc[0]))
        bps = int(meta_row.get("BPS", needs_fill.sum()) or needs_fill.sum())
        daily_p = float(meta_row.get("PRECIP", day_rows["PRECIP"].sum(skipna=True)) or 0.0)
        elen = int(meta_row.get("ELEN", 1) or 1)

        synth: pd.DataFrame | None = None
        tag = ""

        pooled_templates = _pool_templates_for_cluster(
            templates,
            cluster=cluster,
            kg_zone=kg_zone,
        )
        if len(pooled_templates) > len([t for t in templates if t.cluster == cluster]):
            if "kg_pooled_templates" not in rep.strategies:
                rep.strategies.append("kg_pooled_templates")

        if pooled_templates:
            candidates: list[pd.DataFrame] = []
            cluster_donors = [
                observed_ts.loc[observed_ts["CLUSTER"] == cluster].copy()
            ]
            for k in range(_K_CANDIDATES):
                donor = select_donor_template(
                    pooled_templates,
                    cluster=cluster,
                    bps=bps,
                    daily_precip=daily_p,
                    rng=rng,
                )
                if donor is None:
                    continue
                start_min = cluster_median_start(pooled_templates, cluster)
                candidate = apply_template_to_day(
                    donor,
                    yyyymmdd=str(ymd),
                    daily_precip=daily_p,
                    bps=bps,
                    cluster=cluster,
                    start_minute=start_min,
                )
                candidates.append(candidate)

            if candidates:
                synth, ei_pick, pick_idx = select_best_i30_candidate(
                    candidates,
                    cluster_donors,
                )
                if synth is not None:
                    tag = f"storm_template_{ymd}"
                    if pick_idx >= 0 and ei_pick is not None and ei_pick.passed:
                        strategies.append(f"i30_pass_{ymd}")
                    elif pick_idx >= 0:
                        strategies.append(f"i30_best_{ymd}")
                        rep.mark_low_confidence(f"L5 I30 below donor threshold on {ymd}")

        if synth is None:
            start_min = (
                cluster_median_start(templates, cluster)
                if templates
                else _DEFAULT_START_MINUTE
            )
            synth = generate_parametric_day(
                yyyymmdd=str(ymd),
                cluster=cluster,
                bps=bps,
                daily_precip=daily_p,
                elen=elen,
                start_minute=start_min,
            )
            tag = f"parametric_pulse_{ymd}"

        strategies.append(tag)
        n_days += 1

        fill_locs = day_rows.index[needs_fill]
        n_fill = min(len(fill_locs), len(synth))
        for i in range(n_fill):
            loc = fill_locs[i]
            for col in value_cols:
                if col in synth.columns:
                    day_rows.loc[loc, col] = synth.iloc[i][col]
            if "DT_1" in day_rows.columns and "DT_1" in synth.columns:
                day_rows.loc[loc, "DT_1"] = synth.iloc[i]["DT_1"]

        still_missing = day_rows.loc[fill_locs, value_cols].isna().any(axis=1)
        remaining = fill_locs[still_missing.to_numpy()]
        if len(remaining) and str(cfg.pm).upper() == "T":
            strategies.append(f"fcs_fallback_{ymd}")
            rep.mark_low_confidence("L5 subdaily PyMICE FCS fallback; confidence=low")
            filled = _run_mice_on_frame(
                day_rows.loc[remaining, value_cols],
                imp_method=imp_method,
                config=cfg,
            )
            for col in value_cols:
                day_rows.loc[remaining, col] = filled[col].to_numpy()

        if "DT_1" in day_rows.columns and not any(s.startswith(("i30_pass_", "i30_best_")) for s in strategies if str(ymd) in s):
            cluster_donors = [
                observed_ts.loc[observed_ts["CLUSTER"] == cluster].copy()
            ]
            ei = validate_i30_against_donors(
                day_rows.loc[needs_fill, ["DT_1", "DUR", "PRECIP"]].dropna(subset=["DT_1"]),
                cluster_donors,
            )
            ei_results.append(ei)
            if not ei.passed:
                strategies.append(f"i30_warn_{ymd}")
                rep.mark_low_confidence(f"L5 I30 validation warn on {ymd}")

        out_parts.append(day_rows)

    result = pd.concat(out_parts, ignore_index=True) if out_parts else work
    if "DUR" in result.columns:
        result["DUR"] = pd.to_numeric(result["DUR"], errors="coerce").fillna(15.0)
    if "PRECIP" in result.columns:
        result["PRECIP"] = pd.to_numeric(result["PRECIP"], errors="coerce").fillna(0.0)
    if "DT_1" in result.columns:
        result = resolve_overlaps(result)
        result = trim_overlap_durations(result)

    n_cells = int(result[value_cols].notna().sum().sum()) if value_cols else 0
    rep.n_subdaily_days_imputed += n_days
    rep.n_subdaily_cells_imputed += n_cells
    if any(s.startswith("storm_template_") for s in strategies) and "storm_template" not in rep.strategies:
        rep.strategies.append("storm_template")
    if any(s.startswith("parametric_pulse_") for s in strategies) and "parametric_pulse" not in rep.strategies:
        rep.strategies.append("parametric_pulse")
    if any(s.startswith("fcs_fallback_") for s in strategies) and "subdaily_fcs_fallback" not in rep.strategies:
        rep.strategies.append("subdaily_fcs_fallback")
    if any(s.startswith("i30_warn_") for s in strategies) and "i30_validation_warn" not in rep.strategies:
        rep.strategies.append("i30_validation_warn")
    if any(s.startswith("i30_pass_") for s in strategies) and "i30_k_candidate_pass" not in rep.strategies:
        rep.strategies.append("i30_k_candidate_pass")
    if n_days:
        rep.notes.append(f"L5 imputed {n_days} wet-day sub-daily sequences")
    return result, strategies, ei_results, n_days