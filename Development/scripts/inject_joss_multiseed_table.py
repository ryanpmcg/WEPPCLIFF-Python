#!/usr/bin/env python3
"""Inject multiseed markdown table into JOSS-PAPER/paper.md."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "JOSS-PAPER" / "paper.md"
TABLE = ROOT / "JOSS-PAPER" / "results" / "TABLE_multiseed_summary.md"
START = "<!-- AUTO:MULTISEED_TABLE -->"
END = "<!-- /AUTO:MULTISEED_TABLE -->"


def main() -> int:
    if not TABLE.exists():
        raise SystemExit(f"Missing {TABLE}; run multiseed script first.")
    body = TABLE.read_text(encoding="utf-8")
    # Drop the H1 title line; paper already has a section heading
    lines = body.strip().splitlines()
    if lines and lines[0].startswith("#"):
        lines = lines[1:]
    # Keep the markdown table and a short note
    insert = "\n".join(lines).strip() + "\n"
    paper = PAPER.read_text(encoding="utf-8")
    if START not in paper or END not in paper:
        raise SystemExit("paper.md missing AUTO multiseed markers")
    pre, rest = paper.split(START, 1)
    _, post = rest.split(END, 1)
    PAPER.write_text(pre + START + "\n\n" + insert + "\n" + END + post, encoding="utf-8")
    print(f"Updated {PAPER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
