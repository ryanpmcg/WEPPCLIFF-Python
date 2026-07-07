#!/usr/bin/env python3
"""Post-write CLI structural validation for Autotest runs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.cli_validate import validate_cli_file  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: validate_cli_output.py <file.cli> [file2.cli ...]", file=sys.stderr)
        return 2

    failed = 0
    for arg in argv[1:]:
        path = Path(arg)
        if not path.exists():
            print(f"MISSING: {path}")
            failed += 1
            continue
        violations = validate_cli_file(path, errors_only=True)
        if violations:
            print(f"FAIL: {path} ({len(violations)} errors)")
            for v in violations[:10]:
                print(f"  {v}")
            if len(violations) > 10:
                print(f"  ... {len(violations) - 10} more")
            failed += 1
        else:
            print(f"PASS: {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))