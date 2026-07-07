"""PHIGI hold-out harness scaffold (Phase 0)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.config import FLAG_NAMES, WeppcliffConfig  # noqa: E402
from Source.phigi.report import ImputationReport  # noqa: E402


def test_gf_flag_registered() -> None:
    assert "gf" in FLAG_NAMES
    assert "tauqc" in FLAG_NAMES
    assert "ir" in FLAG_NAMES


def test_phigi_runtime_flags() -> None:
    config = WeppcliffConfig(gf="phigi", ir="t")
    config.apply_runtime_flags()
    assert config.do_phigi is True
    assert config.do_ir is True


def test_imputation_report_serializes() -> None:
    rep = ImputationReport(layers_used=["L0"], n_sorad_imputed=3)
    data = rep.to_dict()
    assert data["engine"] == "phigi"
    assert data["n_sorad_imputed"] == 3