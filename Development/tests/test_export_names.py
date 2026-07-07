"""Tests for human-readable export filenames."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.io import EXPORT_FILE_SUFFIXES, export_csv_path  # noqa: E402


def test_export_csv_path_uses_hyphenated_title_case() -> None:
    path = export_csv_path(Path("/tmp/Export"), "KMQE", "precip")
    assert path.name == "KMQE-Precipitation.csv"


def test_all_export_suffixes_are_title_case_hyphenated() -> None:
    for key, suffix in EXPORT_FILE_SUFFIXES.items():
        assert " " not in suffix, key
        assert suffix[0].isupper(), key
        assert not suffix.endswith("Data"), key