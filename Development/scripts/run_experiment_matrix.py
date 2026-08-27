#!/usr/bin/env python3
"""
Run the WEPPCLIFF ampute–impute experiment matrix (A–G).

Dual-scope scoring reports whole-series and hole-only statistics at daily,
monthly, seasonal, and annual periods. See:

  Documentation/Planning/AmputeImputeExperimentMatrix.md

Examples
--------
  # CI smoke (tiny seeds, no workflow F)
  .venv/bin/python Development/scripts/run_experiment_matrix.py --profile smoke --serial

  # Default assessment profile (recommended local run)
  .venv/bin/python Development/scripts/run_experiment_matrix.py --profile default --workers 4

  # Paper multiseed counts
  .venv/bin/python Development/scripts/run_experiment_matrix.py --profile paper --workers 4

  # Single experiment
  .venv/bin/python Development/scripts/run_experiment_matrix.py --experiments A,B --profile smoke

  # Include full fill_workflow PHIGI vs legacy (slow)
  .venv/bin/python Development/scripts/run_experiment_matrix.py --experiments F --include-workflow --profile default
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.weppcliff_impute.experiment_matrix.catalog import (  # noqa: E402
    EXPERIMENT_CATALOG,
    list_experiments,
)
from Source.weppcliff_impute.experiment_matrix.report import (  # noqa: E402
    write_paper_snippet,
    write_summary_markdown,
)
from Source.weppcliff_impute.experiment_matrix.runner import (  # noqa: E402
    MatrixRunConfig,
    run_matrix,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="WEPPCLIFF ampute–impute experiment matrix (dual whole/hole scoring).",
    )
    parser.add_argument(
        "--profile",
        choices=("smoke", "default", "paper"),
        default="default",
        help="Seed counts per experiment (smoke=tiny, default=moderate, paper=full)",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="A,B,C,D,E,G",
        help="Comma-separated experiment ids (A–G). Default omits F (workflow).",
    )
    parser.add_argument(
        "--include-workflow",
        action="store_true",
        help="Enable experiment F (full fill_workflow legacy vs PHIGI; slow)",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        help="Override seed count for all selected experiments",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--io", type=int, default=15, help="PyMICE iterations")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default Autotest/Output/ExperimentMatrix)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List experiment catalog and exit",
    )
    args = parser.parse_args(argv)

    if args.list:
        for spec in list_experiments():
            print(f"{spec.id}: {spec.title}")
            print(f"   panel={spec.panel} engines={spec.engines}")
            print(f"   scenarios={spec.scenarios or spec.workflow_scenarios}")
            print(f"   seeds smoke/default/paper={spec.smoke_n_seeds}/{spec.default_n_seeds}/{spec.paper_n_seeds}")
            print(f"   {spec.description[:100]}...")
            print()
        return 0

    exp_ids = tuple(x.strip().upper() for x in args.experiments.split(",") if x.strip())
    for e in exp_ids:
        if e not in EXPERIMENT_CATALOG:
            print(f"Unknown experiment {e}; choose from {sorted(EXPERIMENT_CATALOG)}", file=sys.stderr)
            return 2

    out_dir = args.out or (ROOT / "Autotest" / "Output" / "ExperimentMatrix")
    config = MatrixRunConfig(
        experiments=exp_ids,
        profile=args.profile,
        seed_start=args.seed_start,
        n_seeds_override=args.n_seeds,
        workers=args.workers,
        serial=args.serial or args.workers <= 1,
        io=args.io,
        out_dir=out_dir,
        include_workflow=args.include_workflow or "F" in exp_ids,
    )

    print(
        f"Running experiments {exp_ids} profile={args.profile} "
        f"io={args.io} workers={1 if config.serial else args.workers}"
    )
    replicates, agg, gates, written = run_matrix(config)

    report_path = written / "REPORT.md"
    write_summary_markdown(
        agg,
        gates,
        report_path,
        profile=args.profile,
        experiments=exp_ids,
    )
    write_paper_snippet(agg, written / "TABLE_experiment_A_dual_scope.md")

    # Mirror paper-facing tables when full paper profile
    paper_dir = ROOT / "JOSS-PAPER" / "results"
    if args.profile == "paper":
        paper_dir.mkdir(parents=True, exist_ok=True)
        if not replicates.empty:
            replicates.to_csv(paper_dir / "matrix_replicates_long.csv", index=False)
        if not agg.empty:
            agg.to_csv(paper_dir / "matrix_aggregate_mean_std.csv", index=False)
        write_summary_markdown(
            agg,
            gates,
            paper_dir / "TABLE_experiment_matrix.md",
            profile=args.profile,
            experiments=exp_ids,
        )

    n_fail = 0
    if not gates.empty and "passed" in gates.columns:
        n_fail = int((~gates["passed"]).sum())
        print(f"Gates: {int(gates['passed'].sum())} passed, {n_fail} failed")
    print(f"Report: {report_path}")
    # Smoke / CI: fail only on job success gate or total job failure
    if replicates.empty:
        return 1
    jobs = replicates[
        (replicates.get("variable") == "_JOB") & (replicates.get("metric") == "status")
    ] if "variable" in replicates.columns else replicates
    if not jobs.empty and float(jobs["value"].mean()) < 0.5:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
