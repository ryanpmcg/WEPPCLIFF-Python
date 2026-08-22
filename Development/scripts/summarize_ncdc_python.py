#!/usr/bin/env python3
"""Summarize Python NCDC run vs legacy Results. Does not modify legacy files."""

from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY_WEPP = ROOT / "Export" / "NCDC-Python-2026" / "Complete" / "WEPP"
PY_DAILY = ROOT / "Export" / "NCDC-Python-2026" / "Complete" / "Daily"
LEG_WEPP = Path("/Users/home/Software/WEPPCLIFF/NCDC-Update/Results/WEPP")
LEG_DAILY = Path("/Users/home/Software/WEPPCLIFF/NCDC-Update/Results/Daily")
OUT = ROOT / "Export" / "NCDC-Python-2026" / "SUMMARY.md"


def parse_cli_header(path: Path) -> dict[str, object]:
    lines = path.read_text(errors="replace").splitlines()
    meta = {"begin": None, "sim": None, "obs": None, "precip": []}
    if len(lines) >= 13:
        parts = lines[4].split()
        if len(parts) >= 6:
            try:
                meta["obs"] = int(float(parts[3]))
                meta["begin"] = int(float(parts[4]))
                meta["sim"] = int(float(parts[5]))
            except ValueError:
                pass
        try:
            meta["precip"] = [float(x) for x in lines[12].split()]
        except ValueError:
            meta["precip"] = []
    return meta


def daily_precip_sum(path: Path) -> tuple[float, int, int]:
    total = 0.0
    n = 0
    na = 0
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            n += 1
            v = row.get("PRECIP", "")
            if v in ("", "NA", "NaN"):
                na += 1
                continue
            try:
                total += float(v)
            except ValueError:
                na += 1
    return total, n, na


def rmse_monthly(a: list[float], b: list[float]) -> float:
    if len(a) != 12 or len(b) != 12:
        return float("nan")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / 12.0)


def main() -> int:
    py_clis = {p.stem: p for p in PY_WEPP.glob("*.cli")}
    leg_clis = {p.stem: p for p in LEG_WEPP.glob("*.cli")}
    common = sorted(set(py_clis) & set(leg_clis))
    only_py = sorted(set(py_clis) - set(leg_clis))
    only_leg = sorted(set(leg_clis) - set(py_clis))

    rows = []
    for stem in common:
        ph = parse_cli_header(py_clis[stem])
        lh = parse_cli_header(leg_clis[stem])
        p_sum = sum(ph["precip"]) if ph["precip"] else float("nan")
        l_sum = sum(lh["precip"]) if lh["precip"] else float("nan")
        rows.append(
            {
                "station": stem,
                "py_begin": ph["begin"],
                "leg_begin": lh["begin"],
                "py_sim": ph["sim"],
                "leg_sim": lh["sim"],
                "py_p_ann": p_sum,
                "leg_p_ann": l_sum,
                "p_rmse": rmse_monthly(ph["precip"], lh["precip"]),
                "p_ratio": (p_sum / l_sum) if l_sum not in (0, float("nan")) else float("nan"),
            }
        )

    def mean(key: str) -> float:
        vals = [r[key] for r in rows if isinstance(r[key], (int, float)) and r[key] == r[key]]
        return sum(vals) / len(vals) if vals else float("nan")

    n = len(rows)
    ratios = [r["p_ratio"] for r in rows if r["p_ratio"] == r["p_ratio"]]
    low = sum(1 for r in ratios if r < 0.5)
    lines = [
        "# Python WEPPCLIFF vs legacy NCDC-Update",
        "",
        "Python outputs: `Export/NCDC-Python-2026/` (does **not** overwrite `~/Software/WEPPCLIFF/NCDC-Update`).",
        "Python flags: `-qc -ei -ed 6` (**no `-id`** — fill still wipes observed precip).",
        "",
        f"- Python `.cli` files: {len(py_clis)}",
        f"- Legacy `.cli` files: {len(leg_clis)}",
        f"- Common stations: {n}",
        f"- Python only: {len(only_py)}",
        f"- Legacy only: {len(only_leg)}",
        "",
        "## Climatology (monthly precip header, mm/year = sum of 12 monthly means)",
        "",
        f"- Mean Python annual P: {mean('py_p_ann'):.1f} mm",
        f"- Mean legacy annual P: {mean('leg_p_ann'):.1f} mm",
        f"- Mean Python/legacy P ratio: {mean('p_ratio'):.3f}",
        f"- Mean monthly-P RMSE: {mean('p_rmse'):.2f} mm",
        f"- Stations with Python P < 50% of legacy: {low} / {len(ratios)}",
        "",
        "## Sample stations (first 15 by id)",
        "",
        "| Station | Py years | Leg years | Py P mm/yr | Leg P mm/yr | Ratio |",
        "|---------|----------|-----------|------------|-------------|-------|",
    ]
    for r in rows[:15]:
        lines.append(
            f"| {r['station']} | {r['py_begin']}+{r['py_sim']} | {r['leg_begin']}+{r['leg_sim']} | "
            f"{r['py_p_ann']:.1f} | {r['leg_p_ann']:.1f} | {r['p_ratio']:.3f} |"
        )
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
