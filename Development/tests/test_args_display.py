"""Tests for CLI argument summary formatting."""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.args import assign_args_to_vars, assign_empty_args, parse_args  # noqa: E402


def _capture_args(argv: list[str]) -> str:
    config = parse_args(argv)
    assign_empty_args(config)
    buf = io.StringIO()
    with redirect_stdout(buf):
        assign_args_to_vars(config, argv)
    return buf.getvalue()


def test_no_blank_line_between_argument_rows() -> None:
    text = _capture_args(["-f", "ASOS_BPT_KMQE.csv", "-pd", "t", "-la", "y"])
    lines = [ln for ln in text.splitlines() if ln.startswith(" ")]
    assert lines, "expected argument rows in summary"
    for prev, curr in zip(lines, lines[1:]):
        assert prev.strip(), "blank argument row"
        assert curr.strip(), "blank argument row"


def test_defaults_printed_for_unspecified_flags() -> None:
    text = _capture_args(["-f", "ASOS_BPT_KMQE.csv", "-pd", "t", "-la", "y"])
    assert "d:\t" in text and "Input" in text
    assert "qc:\tf" in text or "qc:\t f" in text.replace(" ", "")
    assert "pm:\tt" in text or "pm:\t t" in text.replace(" ", "")


def test_library_argument_removed() -> None:
    from Source.config import FLAG_NAMES

    text = _capture_args(["-f", "ASOS_BPT_KMQE.csv", "-la", "y"])
    assert "l" not in FLAG_NAMES
    assert "Library Directory" not in text
    assert f"of {len(FLAG_NAMES)} possible arguments" in text