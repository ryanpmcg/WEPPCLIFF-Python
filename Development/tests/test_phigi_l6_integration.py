"""L6 PHIGI integration: dispatcher, finalize, confidence tier."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.config import WeppcliffConfig  # noqa: E402
from Source.impute import fill_data, fill_data_legacy, fill_data_phigi  # noqa: E402
from Source.phigi.pipeline import finalize_phigi_gap_fill  # noqa: E402
from Source.phigi.report import ImputationReport  # noqa: E402


def test_fill_data_entry_points_exist() -> None:
    assert callable(fill_data)
    assert callable(fill_data_phigi)
    assert callable(fill_data_legacy)


def test_finalize_writes_report_on_ir(tmp_path: Path) -> None:
    config = WeppcliffConfig(gf="phigi", ir="t", e=str(tmp_path), ex_dir=str(tmp_path))
    config.apply_runtime_flags()
    config.phigi_report = ImputationReport(layers_used=["L0", "L2"])
    finalize_phigi_gap_fill(config)
    out = tmp_path / "Imputation-Report.json"
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["engine"] == "phigi"
    assert "L6" in data["layers_used"]


def test_mark_low_confidence() -> None:
    rep = ImputationReport()
    rep.mark_low_confidence("test fallback")
    assert rep.confidence_tier == "low"
    assert "test fallback" in rep.notes


def test_finalize_preserves_low_confidence() -> None:
    config = WeppcliffConfig(gf="phigi", ir="f", verb="f")
    config.apply_runtime_flags()
    rep = ImputationReport(confidence_tier="low", strategies=["subdaily_fcs_fallback"])
    config.phigi_report = rep
    finalize_phigi_gap_fill(config)
    assert config.phigi_report.confidence_tier == "low"