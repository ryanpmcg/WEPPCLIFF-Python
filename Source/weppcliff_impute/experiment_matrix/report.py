"""Markdown / paper report writers for the experiment matrix."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from Source.weppcliff_impute.experiment_matrix.catalog import EXPERIMENT_CATALOG


def _fmt(mean: float, std: float) -> str:
    if pd.isna(mean):
        return "—"
    if pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


def write_summary_markdown(
    agg: pd.DataFrame,
    gates: pd.DataFrame,
    out_path: Path,
    *,
    profile: str,
    experiments: tuple[str, ...],
) -> str:
    """Human-readable assessment report emphasizing whole vs hole."""
    lines: list[str] = [
        "# Ampute–impute experiment matrix report",
        "",
        f"**Profile:** `{profile}`  ",
        f"**Experiments:** {', '.join(experiments)}",
        "",
        "Metrics use dual **scope**:",
        "",
        "- `whole` — reconstructed full series vs truth (monthly/seasonal/annual climate)",
        "- `hole` — only amputed calendar days (reconstruction skill)",
        "- `holdout_cells` — cell-level hold-out (occurrence, KS, pointwise RMSE)",
        "",
        "Periods: `daily`, `monthly`, `seasonal`, `annual`, `sample`.",
        "",
    ]

    for eid in experiments:
        spec = EXPERIMENT_CATALOG.get(eid)
        if spec is None:
            continue
        lines.append(f"## Experiment {eid}: {spec.title}")
        lines.append("")
        lines.append(spec.description)
        lines.append("")
        if agg.empty or "experiment" not in agg.columns:
            lines.append("*No aggregate rows.*")
            lines.append("")
            continue
        sub = agg[agg["experiment"] == eid].copy()
        if sub.empty:
            lines.append("*No results for this experiment.*")
            lines.append("")
            continue

        # Highlight key metrics
        key_metrics = [
            ("PRECIP", "wet_f1", "holdout_cells", "daily"),
            ("PRECIP", "holdout_mass_ratio", "holdout_cells", "daily"),
            ("PRECIP", "mass_ratio", "hole", "daily"),
            ("PRECIP", "mass_rel_bias", "hole", "monthly"),
            ("PRECIP", "mass_rel_bias", "whole", "monthly"),
            ("PRECIP", "mass_ratio", "whole", "annual"),
            ("PRECIP", "mass_ratio", "hole", "annual"),
            ("PRECIP", "wet_intensity_ks", "holdout_cells", "daily"),
            ("MIN_TEMP", "rmse", "holdout_cells", "daily"),
            ("SO_RAD", "rmse", "holdout_cells", "daily"),
            ("EMAX", "mean_ratio", "holdout_cells", "sample"),
            ("W_VEL", "speed_ks", "holdout_cells", "sample"),
            ("ANT", "mean_ratio", "holdout_cells", "sample"),
            ("BTW", "mean_ratio", "holdout_cells", "sample"),
            ("I30", "mean_rel_bias", "holdout_cells", "daily"),
        ]
        lines.append(
            "| Engine | Scenario | Variable | Metric | Scope | Period | Mean ± SD | N |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for engine in sorted(sub["engine"].unique()):
            for scenario in sorted(sub["scenario"].unique()):
                for var, metric, scope, period in key_metrics:
                    hit = sub[
                        (sub["engine"] == engine)
                        & (sub["scenario"] == scenario)
                        & (sub["variable"] == var)
                        & (sub["metric"] == metric)
                        & (sub["scope"] == scope)
                        & (sub["period"] == period)
                    ]
                    if hit.empty:
                        continue
                    r = hit.iloc[0]
                    lines.append(
                        f"| {engine} | {scenario} | {var} | {metric} | {scope} | "
                        f"{period} | {_fmt(r['mean'], r['std'])} | {int(r['n'])} |"
                    )
        lines.append("")

        # Whole vs hole contrast for PRECIP mass
        lines.append("### Whole vs hole PRECIP mass (midastouch when present)")
        lines.append("")
        lines.append("| Scenario | Scope | Period | mass_ratio / mass_rel_bias | N |")
        lines.append("| --- | --- | --- | --- | --- |")
        mid = sub[sub["engine"] == "midastouch"] if "midastouch" in set(sub["engine"]) else sub
        for scenario in sorted(mid["scenario"].unique()):
            for scope in ("whole", "hole"):
                for period, metric in (
                    ("daily", "mass_ratio"),
                    ("monthly", "mass_rel_bias"),
                    ("annual", "mass_ratio"),
                ):
                    hit = mid[
                        (mid["scenario"] == scenario)
                        & (mid["variable"] == "PRECIP")
                        & (mid["metric"] == metric)
                        & (mid["scope"] == scope)
                        & (mid["period"] == period)
                    ]
                    if hit.empty:
                        continue
                    r = hit.iloc[0]
                    lines.append(
                        f"| {scenario} | {scope} | {period} ({metric}) | "
                        f"{_fmt(r['mean'], r['std'])} | {int(r['n'])} |"
                    )
        lines.append("")

    if gates is not None and not gates.empty:
        lines.append("## Gates")
        lines.append("")
        lines.append("| Gate | Description | Observed | Threshold | Pass |")
        lines.append("| --- | --- | --- | --- | --- |")
        for _, g in gates.iterrows():
            obs = g["observed"]
            obs_s = f"{obs:.4f}" if isinstance(obs, float) else str(obs)
            passed = "yes" if g["passed"] else "**no**"
            lines.append(
                f"| {g['gate_id']} | {g['description']} | {obs_s} | "
                f"{g['threshold']} | {passed} |"
            )
        lines.append("")

    lines.append("## Full tables")
    lines.append("")
    lines.append("- `replicates_long.csv` — every seed × metric")
    lines.append("- `aggregate_mean_std.csv` — mean/std by factor")
    lines.append("- `gates.csv` — CI assessment gates")
    lines.append("")
    lines.append("See `Documentation/Planning/AmputeImputeExperimentMatrix.md`.")
    lines.append("")

    text = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return text


def write_paper_snippet(agg: pd.DataFrame, out_path: Path) -> str:
    """Compact markdown table for JOSS paper update (experiment A midastouch)."""
    if agg.empty:
        text = "_No aggregate data._\n"
        out_path.write_text(text, encoding="utf-8")
        return text
    sub = agg[
        (agg.get("experiment") == "A")
        & (agg.get("engine") == "midastouch")
        & (agg.get("scenario") == "mcar_ampute")
    ] if "experiment" in agg.columns else agg
    lines = [
        "# Experiment A (MCAR midastouch) dual-scope snapshot",
        "",
        "| Variable | Metric | Scope | Period | Mean ± SD |",
        "| --- | --- | --- | --- | --- |",
    ]
    if not sub.empty:
        for _, r in sub.sort_values(["variable", "metric", "scope", "period"]).iterrows():
            lines.append(
                f"| {r['variable']} | {r['metric']} | {r['scope']} | {r['period']} | "
                f"{_fmt(r['mean'], r['std'])} |"
            )
    text = "\n".join(lines) + "\n"
    out_path.write_text(text, encoding="utf-8")
    return text
