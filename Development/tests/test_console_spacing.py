"""Tests for single-blank-line console formatting."""

from __future__ import annotations

import io
import re
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


def test_no_double_blank_lines_in_argument_summary() -> None:
    text = _capture_args(["-f", "ASOS_BPT_KMQE.csv", "-pd", "t", "-la", "y", "-verb", "t"])
    assert "\n\n\n" not in text


def test_at_most_one_blank_line_after_section_dividers() -> None:
    text = _capture_args(["-f", "ASOS_BPT_KMQE.csv", "-verb", "t"])
    blocks = re.findall(
        r"\*{44}\n(\n*)",
        text,
    )
    assert blocks, "expected section divider lines"
    for gap in blocks:
        assert gap in ("", "\n"), "expected at most one blank line after section divider"