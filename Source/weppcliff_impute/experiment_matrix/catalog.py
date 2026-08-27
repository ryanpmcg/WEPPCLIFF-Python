"""Experiment matrix catalog (A–F) and smoke / paper profiles."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExperimentSpec:
    """One experiment in the assessment matrix."""

    id: str
    title: str
    description: str
    scenarios: tuple[str, ...]
    engines: tuple[str, ...]
    panel: str  # daily | breakpoint | workflow
    default_n_seeds: int
    paper_n_seeds: int
    smoke_n_seeds: int = 1
    include_storm_metrics: bool = True
    notes: tuple[str, ...] = ()
    workflow_scenarios: tuple[str, ...] = ()


EXPERIMENT_CATALOG: dict[str, ExperimentSpec] = {
    "A": ExperimentSpec(
        id="A",
        title="MCAR baseline (midastouch)",
        description=(
            "12% MCAR on AutoTest.1 daily panel with midastouch (and mean/cart "
            "ablation). Dual whole/hole scoring. Replaces paper multiseed core."
        ),
        scenarios=("mcar_ampute",),
        engines=("midastouch", "mean", "cart"),
        panel="daily",
        default_n_seeds=10,
        paper_n_seeds=30,
        smoke_n_seeds=1,
        notes=("Primary reliability baseline",),
    ),
    "B": ExperimentSpec(
        id="B",
        title="Powered short gaps",
        description=(
            "Contiguous short gaps covering ~10% of days (powered protocol). "
            "Hole-only mass and wet-day KS are primary; whole-series mass is secondary."
        ),
        scenarios=("short_gaps_powered",),
        engines=("midastouch", "midastouch_post_cluster", "mean"),
        panel="daily",
        default_n_seeds=10,
        paper_n_seeds=30,
        smoke_n_seeds=1,
        notes=("Addresses under-powered legacy short_gaps",),
    ),
    "C": ExperimentSpec(
        id="C",
        title="Long gap × four seasons",
        description=(
            "Full-variable long gaps rotated through DJF/MAM/JJA/SON. "
            "Engines: midastouch, post-cluster midastouch, mean. "
            "EMAX/EMIN explosion gated as storm metrics."
        ),
        scenarios=(
            "long_gap_DJF",
            "long_gap_MAM",
            "long_gap_JJA",
            "long_gap_SON",
        ),
        engines=("midastouch", "midastouch_post_cluster", "mean"),
        panel="daily",
        default_n_seeds=5,
        paper_n_seeds=15,
        smoke_n_seeds=1,
        notes=("Season rotation fixes fixed-window bias of long_gap_all",),
    ),
    "D": ExperimentSpec(
        id="D",
        title="Variable-absent sensors",
        description=(
            "Entire SO_RAD (and MIN_TEMP) sensor loss. Tests prior/resilience "
            "and motivates PHIGI solar physics."
        ),
        scenarios=("variable_absent:SO_RAD", "variable_absent:MIN_TEMP"),
        engines=("midastouch", "mean", "midastouch_post_cluster"),
        panel="daily",
        default_n_seeds=5,
        paper_n_seeds=10,
        smoke_n_seeds=1,
        notes=("SO_RAD absence is the smoking-gun failure for physics-free fill",),
    ),
    "E": ExperimentSpec(
        id="E",
        title="Breakpoint MCAR (timing / I30)",
        description=(
            "MCAR on AutoTest.1 precipitation breakpoint panel. Engines: "
            "midastouch vs storm-template FCS. Scores ANT/BTW/DUR/INT + I30 proxy."
        ),
        scenarios=("breakpoint_mcar",),
        engines=("midastouch", "storm_template_fcs", "mean"),
        panel="breakpoint",
        default_n_seeds=5,
        paper_n_seeds=15,
        smoke_n_seeds=1,
        include_storm_metrics=False,
        notes=("Validates -st claim for EI-critical timing fields",),
    ),
    "F": ExperimentSpec(
        id="F",
        title="Full workflow legacy vs PHIGI",
        description=(
            "End-to-end fill_workflow on raw KMQE with fair_compare hold-out "
            "scenarios. Reports daily RMSE, precip scores, I30/EI relative bias."
        ),
        scenarios=(),  # uses workflow_scenarios
        engines=("workflow_legacy", "workflow_phigi"),
        panel="workflow",
        default_n_seeds=2,
        paper_n_seeds=5,
        smoke_n_seeds=0,  # expensive; opt-in
        workflow_scenarios=(
            "holdout_random",
            "sorad_corrupt_summer",
            "seasonal_gap",
            "sparse_donors",
        ),
        notes=("Requires Input/ASOS_BPT_KMQE.csv; slow",),
    ),
    "G": ExperimentSpec(
        id="G",
        title="MNAR cold-sensor + method ablation",
        description=(
            "MNAR cold-morning dropouts plus optional JOMO ablation under MCAR."
        ),
        scenarios=("mnar_cold_sensor", "mcar_ampute"),
        engines=("midastouch", "jomo", "mean"),
        panel="daily",
        default_n_seeds=5,
        paper_n_seeds=10,
        smoke_n_seeds=1,
        notes=("Recommended improvement: MNAR + method ablation",),
    ),
}


def list_experiments() -> list[ExperimentSpec]:
    return [EXPERIMENT_CATALOG[k] for k in sorted(EXPERIMENT_CATALOG)]


def resolve_seed_count(spec: ExperimentSpec, profile: str) -> int:
    profile = profile.lower()
    if profile == "smoke":
        return spec.smoke_n_seeds
    if profile == "paper":
        return spec.paper_n_seeds
    if profile == "default":
        return spec.default_n_seeds
    raise ValueError(f"unknown profile {profile!r}; use smoke|default|paper")
