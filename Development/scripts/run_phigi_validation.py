#!/usr/bin/env python3
"""PHIGI validation harness (L6 scaffold)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PHIGI unit and integration tests.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all Development/tests (slow; includes autotest benchmarks).",
    )
    args = parser.parse_args(argv)

    python = ROOT / ".venv" / "bin" / "python"
    exe = str(python if python.exists() else sys.executable)

    if args.full:
        cmd = [exe, "-m", "pytest", str(ROOT / "Development" / "tests"), "-v"]
    else:
        cmd = [
            exe,
            "-m",
            "pytest",
            str(ROOT / "Development" / "tests"),
            "-k",
            "phigi",
            "-v",
        ]

    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())