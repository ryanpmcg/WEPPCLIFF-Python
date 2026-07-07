#!/usr/bin/env python3
"""
Calibrate PHIGI Köppen zone × season priors (v2 schema).

Fits marginal means/stds and Pearson correlation matrices from observational
panels keyed by Köppen zone and meteorological season. Intended inputs are
GHCN-Daily, GSOD, or DAYMET extracts with columns:

  zone, season, TAU, MIN_TEMP, MAX_TEMP, W_VEL, DP_TEMP

Until a network extract is wired, run with ``--demo`` to emit a schema-valid
skeleton from the bundled v2 file (provenance stamp only).

Example (future):
  python Development/scripts/calibrate_koppen_priors.py \\
    --input data/ghcn_koppen_panel.parquet \\
    --output priors/koppen_priors_v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "priors" / "koppen_priors_v2.json"
DEFAULT_VARIABLES = ("TAU", "MIN_TEMP", "MAX_TEMP", "W_VEL", "DP_TEMP")


def meteorological_season(month: int, lat_deg: float = 45.0) -> int:
    """DJF=1, MAM=2, JJA=3, SON=4 with hemisphere flip."""
    m = int(month)
    if lat_deg < 0.0:
        m = ((m + 5) % 12) + 1
    if m in (12, 1, 2):
        return 1
    if m in (3, 4, 5):
        return 2
    if m in (6, 7, 8):
        return 3
    return 4


def fit_zone_season(
    frame: pd.DataFrame,
    *,
    variables: tuple[str, ...] = DEFAULT_VARIABLES,
    min_n: int = 30,
) -> dict[str, Any] | None:
    """Fit means, stds, and correlation matrix for one zone × season slice."""
    cols = [c for c in variables if c in frame.columns]
    if not cols:
        return None
    clean = frame[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(clean) < min_n:
        return None

    means = {v: float(clean[v].mean()) for v in cols}
    stds = {v: float(max(clean[v].std(ddof=1), 1e-3)) for v in cols}
    corr = clean[cols].corr(method="pearson").to_numpy(dtype=np.float64)
    np.fill_diagonal(corr, 1.0)
    return {"means": means, "stds": stds, "correlation": corr.tolist()}


def build_prior_db(
    panel: pd.DataFrame,
    *,
    variables: tuple[str, ...] = DEFAULT_VARIABLES,
    min_n: int = 30,
    provenance: str = "",
) -> dict[str, Any]:
    """Aggregate zone × season fits into v2 JSON structure."""
    zones: dict[str, Any] = defaultdict(lambda: {"seasons": {}})
    grouped = panel.groupby(["zone", "season"], dropna=True)
    for (zone, season), grp in grouped:
        entry = fit_zone_season(grp, variables=variables, min_n=min_n)
        if entry is None:
            continue
        zones[str(zone)]["seasons"][str(int(season))] = entry

    return {
        "version": 2,
        "description": (
            "PHIGI Köppen zone × season priors (τ + daily weather); "
            f"calibrated {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        ),
        "provenance": provenance,
        "variables": list(variables),
        "zones": dict(zones),
    }


def load_panel(path: Path) -> pd.DataFrame:
    """Load a calibration panel (CSV or Parquet)."""
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_prior_db(data: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


def demo_export(output: Path) -> None:
    """Copy bundled priors with a calibration provenance stamp (no data fit)."""
    src = DEFAULT_OUTPUT
    with src.open(encoding="utf-8") as fh:
        data = json.load(fh)
    data["provenance"] = (
        "demo export — re-stamp bundled priors; "
        "run with --input when network extract is available"
    )
    write_prior_db(data, output)
    print(f"Wrote demo prior DB to {output}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="Calibration panel with zone, season, and weather columns",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output koppen_priors_v2.json path",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=30,
        help="Minimum rows per zone × season to emit an entry",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Re-export bundled priors with provenance stamp (no fitting)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.demo:
        demo_export(args.output)
        return 0
    if args.input is None:
        print("Provide --input <panel> or use --demo", file=sys.stderr)
        return 2
    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    panel = load_panel(args.input)
    required = {"zone", "season"}
    missing = required - set(panel.columns)
    if missing:
        print(f"Panel missing columns: {sorted(missing)}", file=sys.stderr)
        return 1

    provenance = f"fitted from {args.input.name} (n={len(panel)})"
    data = build_prior_db(panel, min_n=args.min_n, provenance=provenance)
    write_prior_db(data, args.output)
    n_zones = len(data["zones"])
    print(f"Wrote {args.output} ({n_zones} zones)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())