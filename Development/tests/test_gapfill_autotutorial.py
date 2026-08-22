"""AutoTutorial gap-fill smoke tests (Tutorial/AutoTutorial.sh lines 26–27)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
from Source.cli_validate import validate_cli_file  # noqa: E402

VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
PYTHON = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)
INPUT = ROOT / "Input" / "ASOS_BPT_KMQE.csv"

GAPFILL_CASES = [
    (
        "AutoTutorial.quick",
        [
            "-f", "ASOS_BPT_KMQE.csv",
            "-la", "y", "-verb", "-qc", "-pd",
            "-id", "-qi", "-pm", "-st", "-ed", "1",
        ],
    ),
    (
        "AutoTutorial.verbose",
        [
            "-f", "ASOS_BPT_KMQE.csv",
            "-la", "y", "-verb", "-qc", "-pd",
            "-id", "-iv", "-pm", "-st", "-ed", "1",
        ],
    ),
]


def _run_gapfill(name: str, extra_args: list[str]) -> Path:
    out_dir = ROOT / "Autotest" / "Gapfill"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON),
        str(ROOT / "weppcliff.py"),
        "-o", str(out_dir),
        "-e", str(out_dir),
        "-fn", name,
        *extra_args,
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(
            f"weppcliff failed ({name}):\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return out_dir / f"{name}.cli"


@pytest.fixture(scope="module", autouse=True)
def _require_input():
    if not INPUT.exists():
        pytest.skip(f"Input file missing: {INPUT}")


@pytest.mark.parametrize("name,args", GAPFILL_CASES)
def test_autotutorial_gapfill_produces_valid_cli(name: str, args: list[str]) -> None:
    """Gap-fill workflow from AutoTutorial must exit cleanly and write a valid .cli."""
    cli_path = _run_gapfill(name, args)
    assert cli_path.exists(), f"expected output: {cli_path}"
    violations = validate_cli_file(cli_path, errors_only=True)
    assert not violations, f"{name}.cli validation errors: {violations[:5]}"


def test_kmqe_phigi_gapfill_smoke() -> None:
    """L6 smoke: PHIGI gap fill on KMQE produces a structurally valid CLI."""
    cli_path = _run_gapfill(
        "KMQE-phigi-smoke",
        [
            "-f", "ASOS_BPT_KMQE.csv",
            "-gf", "phigi",
            "-ir",
            "-la", "y",
            "-id",
            "-pm",
            "-st",
            "-ed", "1",
        ],
    )
    violations = validate_cli_file(cli_path, errors_only=True)
    assert not violations, f"PHIGI gap-fill validation errors: {violations[:5]}"
    report = cli_path.parent / "Imputation-Report.json"
    assert report.exists()