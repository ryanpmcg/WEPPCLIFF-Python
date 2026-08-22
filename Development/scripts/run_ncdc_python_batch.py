#!/usr/bin/env python3
"""Run NCDC-Update Complete + Rerun through Python WEPPCLIFF with fill.

Writes ONLY under Export/NCDC-Python-Fill/. Never touches
~/Software/WEPPCLIFF/NCDC-Update/Results (legacy) or
Export/NCDC-Python-2026 (previous no-fill Python batch).

Keeps .cli (Output) and compressed .npz (Export, -ed 2 -ef npz).
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv" / "bin" / "python"
WEPPCLIFF = ROOT / "weppcliff.py"
LEGACY_ROOT = Path("/Users/home/Software/WEPPCLIFF/NCDC-Update")
OUT_ROOT = ROOT / "Export" / "NCDC-Python-Fill"
MIN_FREE_GB = 40.0
WORKERS = 8


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def assert_safe_out(path: Path) -> None:
    resolved = path.resolve()
    legacy = LEGACY_ROOT.resolve()
    nofill = (ROOT / "Export" / "NCDC-Python-2026").resolve()
    if resolved == legacy or legacy in resolved.parents or resolved in legacy.parents:
        raise SystemExit(f"Refusing to write under legacy tree: {resolved}")
    if "NCDC-Update" in resolved.parts and "NCDC-Python-Fill" not in resolved.parts:
        raise SystemExit(f"Refusing output path that looks like legacy NCDC-Update: {resolved}")
    if resolved == nofill or nofill in resolved.parents:
        raise SystemExit(f"Refusing to overwrite no-fill Python batch: {resolved}")


def clear_outputs(out_cli: Path, out_npz: Path) -> None:
    """Overwrite prior Python fill products in this tree only."""
    for path in out_cli.glob("*.cli"):
        path.unlink()
    for path in out_npz.glob("*.npz"):
        path.unlink()


def run_dir(input_dir: Path, out_cli: Path, out_npz: Path, label: str) -> int:
    assert_safe_out(out_cli)
    assert_safe_out(out_npz)
    out_cli.mkdir(parents=True, exist_ok=True)
    out_npz.mkdir(parents=True, exist_ok=True)
    n_old = len(list(out_cli.glob("*.cli"))) + len(list(out_npz.glob("*.npz")))
    if n_old:
        print(f"[{label}] removing {n_old} prior .cli/.npz (overwrite)", flush=True)
        clear_outputs(out_cli, out_npz)
    n = len(list(input_dir.glob("WEPP_*.csv")))
    free = free_gb(OUT_ROOT)
    print(f"[{label}] {n} files from {input_dir}", flush=True)
    print(f"[{label}] cli -> {out_cli}", flush=True)
    print(f"[{label}] npz -> {out_npz}", flush=True)
    print(f"[{label}] free space {free:.1f} GB (abort below {MIN_FREE_GB:.0f} GB)", flush=True)
    if free < MIN_FREE_GB:
        print(f"[{label}] NOT ENOUGH DISK. Aborting.", flush=True)
        return 2
    cmd = [
        str(PYTHON),
        str(WEPPCLIFF),
        "-d",
        str(input_dir),
        "-o",
        str(out_cli),
        "-e",
        str(out_npz),
        "-la",
        "y",
        "-qc",
        "-id",
        "-pm",
        "-qi",
        "-ei",
        "-ed",
        "2",
        "-ef",
        "npz",
        "-fn",
        "station",
        "-mp",
        str(WORKERS),
    ]
    log = OUT_ROOT / f"{label}.log"
    print(f"[{label}] log {log}", flush=True)
    with log.open("w", encoding="utf-8") as fh:
        fh.write(" ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(30)
            avail = free_gb(OUT_ROOT)
            if avail < MIN_FREE_GB:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                print(f"[{label}] KILLED: free space {avail:.1f} GB < {MIN_FREE_GB:.0f} GB", flush=True)
                return 3
        return int(proc.returncode or 0)


def main() -> int:
    if not PYTHON.exists():
        raise SystemExit(f"Missing venv python: {PYTHON}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rc = run_dir(
        LEGACY_ROOT / "Inputs" / "Complete",
        OUT_ROOT / "Complete" / "WEPP",
        OUT_ROOT / "Complete" / "NPZ",
        "Complete",
    )
    if rc != 0:
        return rc
    rerun = LEGACY_ROOT / "Inputs" / "Rerun"
    if rerun.is_dir() and any(rerun.glob("WEPP_*.csv")):
        rc = run_dir(
            rerun,
            OUT_ROOT / "Rerun" / "WEPP",
            OUT_ROOT / "Rerun" / "NPZ",
            "Rerun",
        )
        if rc != 0:
            return rc
    elapsed = time.perf_counter() - t0
    print(f"Batch finished in {elapsed/60:.1f} min. Free {free_gb(OUT_ROOT):.1f} GB", flush=True)
    n_cli = len(list((OUT_ROOT / "Complete" / "WEPP").glob("*.cli")))
    n_npz = len(list((OUT_ROOT / "Complete" / "NPZ").glob("*.npz")))
    print(f"Complete: {n_cli} .cli, {n_npz} .npz", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
