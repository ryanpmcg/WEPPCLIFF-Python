#!/usr/bin/env python3
"""Compare R, OldPy (TenStation), and NewPy fills for the ten-station set.

Does not modify R Results, NCDC-Python-2026, Fill Complete/Rerun, or TenStation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.export_store import read_npz_slots  # noqa: E402

LEGACY = Path("/Users/home/Software/WEPPCLIFF/NCDC-Update/Results")
OLD_PY = ROOT / "Export" / "NCDC-Python-Fill" / "TenStation"
NEW_PY = ROOT / "Export" / "NCDC-Python-Fill" / "NewPy"
OUT = NEW_PY / "Compare.csv"
PROTECTED = (
    LEGACY.resolve(),
    (ROOT / "Export" / "NCDC-Python-2026").resolve(),
    (ROOT / "Export" / "NCDC-Python-Fill" / "Complete").resolve(),
    (ROOT / "Export" / "NCDC-Python-Fill" / "Rerun").resolve(),
    (OLD_PY / "WEPP").resolve(),
    (OLD_PY / "NPZ").resolve(),
)


def _assert_out(path: Path) -> None:
    resolved = path.resolve()
    for blocked in PROTECTED:
        if resolved == blocked or blocked in resolved.parents or resolved in blocked.parents:
            raise SystemExit(f"Refusing to write under protected tree: {resolved}")


def _ymd(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str[:8]


def _num(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)


def _cli_p_sum(path: Path) -> float:
    try:
        line = path.read_text(errors="replace").splitlines()[12]
        return float(sum(float(x) for x in line.split()))
    except (OSError, ValueError, IndexError):
        return float("nan")


def _daily_metrics(daily: pd.DataFrame, r_daily: pd.DataFrame, prefix: str) -> dict[str, float]:
    d = daily.copy()
    d["YYYYMMDD"] = _ymd(d["YYYYMMDD"])
    py_p = _num(d["PRECIP"])
    r = r_daily.copy()
    r["YYYYMMDD"] = _ymd(r["YYYYMMDD"])
    merged = d[["YYYYMMDD", "PRECIP"]].merge(
        r[["YYYYMMDD", "PRECIP"]], on="YYYYMMDD", suffixes=("_py", "_r"), how="inner"
    )
    py = _num(merged["PRECIP_py"])
    rr = _num(merged["PRECIP_r"])
    wet_py = py > 0
    wet_r = rr > 0
    obs_wet = rr > 0
    out = {
        f"{prefix}_wet": int(np.nansum(wet_py)),
        f"{prefix}_mass": float(np.nansum(py_p)),
        f"{prefix}_r_wet": int(np.nansum(wet_r)),
        f"{prefix}_r_mass": float(np.nansum(rr)),
        f"{prefix}_mass_vs_r": float(np.nansum(py_p) / np.nansum(rr)) if np.nansum(rr) else float("nan"),
        f"{prefix}_obs_wet_mass_ratio": (
            float(np.nansum(py[obs_wet]) / np.nansum(rr[obs_wet]))
            if np.nansum(rr[obs_wet])
            else float("nan")
        ),
        f"{prefix}_max": float(np.nanmax(py_p)) if np.isfinite(py_p).any() else float("nan"),
    }
    return out


def _from_npz(npz: Path, r_daily: pd.DataFrame, prefix: str) -> dict[str, float]:
    slots = read_npz_slots(npz, {10, 18})
    pre = slots[10]
    post = slots[18]
    row = _daily_metrics(pre, r_daily, f"{prefix}_pre")
    row.update(_daily_metrics(post, r_daily, f"{prefix}_post"))
    pre_p = _num(pre["PRECIP"])
    post_p = _num(post["PRECIP"])
    row[f"{prefix}_wet_gain"] = int(np.nansum(post_p > 0) - np.nansum(pre_p > 0))
    row[f"{prefix}_mass_gain"] = float(np.nansum(post_p) - np.nansum(pre_p))
    row[f"{prefix}_exploded"] = int(row[f"{prefix}_post_max"] > 10000)
    pre_wet = pre_p > 0
    if np.nansum(pre_p[pre_wet]):
        row[f"{prefix}_obs_keep"] = float(np.nansum(post_p[pre_wet]) / np.nansum(pre_p[pre_wet]))
    else:
        row[f"{prefix}_obs_keep"] = float("nan")
    return row


def compare_station(stem: str) -> dict[str, object]:
    row: dict[str, object] = {"station": stem, "error": ""}
    r_daily_p = LEGACY / "Daily" / f"{stem} DAILY DATA.csv"
    r_cli = LEGACY / "WEPP" / f"{stem}.cli"
    old_npz = OLD_PY / "NPZ" / f"{stem}.npz"
    new_npz = NEW_PY / "NPZ" / f"{stem}.npz"
    old_cli = OLD_PY / "WEPP" / f"{stem}.cli"
    new_cli = NEW_PY / "WEPP" / f"{stem}.cli"
    if not r_daily_p.exists():
        row["error"] = "missing_r_daily"
        return row
    if not old_npz.exists() or not new_npz.exists():
        row["error"] = "missing_npz"
        return row
    r_daily = pd.read_csv(r_daily_p)
    r_p = _num(r_daily["PRECIP"])
    row["r_wet"] = int(np.nansum(r_p > 0))
    row["r_mass"] = float(np.nansum(r_p))
    row["r_cli_p"] = _cli_p_sum(r_cli)
    try:
        row.update(_from_npz(old_npz, r_daily, "old"))
        row.update(_from_npz(new_npz, r_daily, "new"))
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}:{exc}"
        return row
    row["old_cli_p"] = _cli_p_sum(old_cli)
    row["new_cli_p"] = _cli_p_sum(new_cli)
    row["old_cli_vs_r"] = (
        row["old_cli_p"] / row["r_cli_p"] if row["r_cli_p"] else float("nan")
    )
    row["new_cli_vs_r"] = (
        row["new_cli_p"] / row["r_cli_p"] if row["r_cli_p"] else float("nan")
    )
    return row


def main() -> int:
    _assert_out(OUT)
    stems = sorted(p.stem for p in (OLD_PY / "NPZ").glob("*.npz"))
    rows = [compare_station(s) for s in stems]
    NEW_PY.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {OUT}")
    ok = df[df["error"].astype(str) == ""]
    if len(ok):
        print(
            "median post mass vs R: "
            f"OldPy={ok['old_post_mass_vs_r'].median():.3f}  "
            f"NewPy={ok['new_post_mass_vs_r'].median():.3f}"
        )
        print(
            "median wet gain: "
            f"OldPy={ok['old_wet_gain'].median():.0f}  "
            f"NewPy={ok['new_wet_gain'].median():.0f}  "
            f"R extra vs OldPy pre uses r_wet"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
