#!/usr/bin/env python3
"""Run PyMICE weather-panel validation suite (Phase 6d)."""

from __future__ import annotations

import sys
from pathlib import Path

MICE_ROOT = Path(__file__).resolve().parents[2].parent / "MICE"
sys.path.insert(0, str(MICE_ROOT / "src"))

from Source.weppcliff_impute.adapter import ImputeConfig  # noqa: E402
from Source.weppcliff_impute.validation import (  # noqa: E402
    format_validation_report,
    run_weather_validation_suite,
)


def main() -> int:
    config = ImputeConfig(pm="t", io=30, seed=42, lat=40.0, kg="Cfa")
    results = run_weather_validation_suite(config, seed=42)
    print(format_validation_report(results))
    failed = [
        r for r in results
        if r.n_filled != r.n_missing or not r.temp_order_ok or r.rmse > 10.0
    ]
    if failed:
        print(f"\n{len(failed)} scenario(s) outside thresholds.", file=sys.stderr)
        return 1
    print("\nAll scenarios passed thresholds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())