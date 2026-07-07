"""Unit tests for PHIGI L1 Köppen zone resolution."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.phigi.l1_koppen import resolve_koppen_zone  # noqa: E402


def test_kmqe_resolves_dfb() -> None:
    zone = resolve_koppen_zone(42.2167, -71.1167, kg="AUTO")
    assert zone == "Dfb"


def test_manual_kg_override() -> None:
    zone = resolve_koppen_zone(42.2167, -71.1167, kg="Cfa")
    assert zone == "Cfa"


def test_tropical_station_am() -> None:
    zone = resolve_koppen_zone(1.35, 103.82, kg="AUTO")
    assert zone == "Am"


def test_grid_lookup_matches_bundled_csv() -> None:
    grid = ROOT / "priors" / "koppen_grid_2deg.csv"
    assert grid.is_file()
    zone = resolve_koppen_zone(33.45, -112.07, kg="AUTO", grid_path=grid)
    assert zone in {"BSk", "BWk", "Cfa"}


def test_rule_fallback_when_grid_missing() -> None:
    zone = resolve_koppen_zone(51.51, -0.13, kg="AUTO", grid_path="/nonexistent/grid.csv")
    assert zone == "Cfb"