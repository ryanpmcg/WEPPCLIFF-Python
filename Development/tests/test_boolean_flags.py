"""Presence/absence semantics for pure boolean CLI flags."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.args import assign_empty_args, parse_args  # noqa: E402
from Source.config import BOOLEAN_FLAGS  # noqa: E402


def test_boolean_presence_is_true() -> None:
    config = parse_args(["-qc", "-id", "-verb", "-pm", "-st"])
    assign_empty_args(config)
    assert config.is_true("qc")
    assert config.is_true("id")
    assert config.is_true("verb")
    assert config.is_true("pm")
    assert config.is_true("st")


def test_boolean_absence_is_false() -> None:
    config = parse_args(["-f", "x.csv", "-la", "y"])
    assign_empty_args(config)
    for flag in BOOLEAN_FLAGS:
        assert not config.is_true(flag), f"expected {flag} false when absent"


def test_legacy_t_suffix_still_enables() -> None:
    config = parse_args(["-qc", "t", "-pd", "t"])
    assign_empty_args(config)
    assert config.is_true("qc")
    assert config.is_true("pd")


def test_legacy_f_suffix_treated_as_present() -> None:
    """Presence model: -flag f still enables (omit the flag for false)."""
    config = parse_args(["-qc", "f"])
    assign_empty_args(config)
    assert config.is_true("qc")


def test_value_flags_still_take_arguments() -> None:
    config = parse_args(["-ed", "2", "-pi", "5", "-mp", "4", "-gf", "phigi"])
    assign_empty_args(config)
    assert str(config.ed) == "2"
    assert str(config.pi) == "5"
    assert str(config.mp) == "4"
    assert str(config.gf).lower() == "phigi"


def test_boolean_does_not_consume_next_value_flag() -> None:
    config = parse_args(["-verb", "-ed", "3", "-ei"])
    assign_empty_args(config)
    assert config.is_true("verb")
    assert config.is_true("ei")
    assert str(config.ed) == "3"
