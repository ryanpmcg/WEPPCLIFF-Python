#!/usr/bin/env python3
"""Fair PHIGI vs legacy hold-out comparison on real KMQE gap-fill pipelines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.args import assign_empty_args  # noqa: E402
from Source.config import WeppcliffConfig  # noqa: E402
from Source.phigi.fair_compare import (  # noqa: E402
    CompareScenario,
    compare_all_scenarios,
    compare_engines_on_kmqe,
)

DEFAULT_INPUT = ROOT / "Input" / "ASOS_BPT_KMQE.csv"


def _default_config(input_path: Path) -> WeppcliffConfig:
    cfg = WeppcliffConfig(
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
    cfg.home_dir = str(ROOT)
    assign_empty_args(cfg)
    return cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare PHIGI vs legacy via full fill_workflow on KMQE.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Station CSV or CLIGEN path (default: KMQE)",
    )
    parser.add_argument(
        "--scenario",
        choices=[s.value for s in CompareScenario] + ["all"],
        default="all",
        help="Hold-out scenario (default: all four)",
    )
    parser.add_argument("--fraction", type=float, default=0.10, help="Hold-out fraction")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    config = _default_config(args.input)

    if args.scenario == "all":
        results = compare_all_scenarios(args.input, fraction=args.fraction, seed=args.seed)
        payload = [r.to_dict() for r in results]
    else:
        scenario = CompareScenario(args.scenario)
        result = compare_engines_on_kmqe(
            args.input,
            scenario,
            fraction=args.fraction,
            seed=args.seed,
            config=config,
        )
        payload = result.to_dict()

    text = json.dumps(payload, indent=2)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())