#!/usr/bin/env python3
"""Quick benchmark for Phase 0/1 refactor verification."""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def md5_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def bench_smart_window_sum() -> float:
    from Source.storms import smart_window_sum

    rng = np.random.default_rng(0)
    n = 50_000
    dur = np.full(n, 5.0)
    cum = np.cumsum(dur)
    pcp = rng.uniform(0, 2, n)
    rows = 72
    t0 = time.perf_counter()
    out = smart_window_sum(cum, pcp, dur, rows, "reverse", 360.0)
    elapsed = time.perf_counter() - t0
    print(f"smart_window_sum(n={n}): {elapsed:.4f}s  sum={np.nansum(out):.4f}")
    return elapsed


def bench_autotest1() -> tuple[float, str]:
    from Source.args import assign_empty_args, assign_final_args, parse_args
    from Source.config import set_config
    from Source.io import load_input_file
    from Source.preprocess import preprocess_file
    from Source.workflow import core_workflow

    out_dir = ROOT / "Autotest" / "Output" / "_bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    argv = [
        "-f",
        str(ROOT / "Input" / "ASOS_BPT_KMQE.csv"),
        "-la",
        "y",
        "-verb",
        "f",
        "-o",
        str(out_dir),
        "-fn",
        "Bench",
        "-d",
        str(ROOT / "Input"),
    ]
    config = parse_args(argv)
    assign_empty_args(config)
    set_config(config)
    assign_final_args(config)
    config.home_dir = str(ROOT)

    t0 = time.perf_counter()
    frame = preprocess_file(load_input_file(config.f, config), config)
    core_workflow(frame, config)
    elapsed = time.perf_counter() - t0
    cli = out_dir / "Bench.cli"
    digest = md5_file(cli) if cli.exists() else "missing"
    print(f"core_workflow AutoTest.1: {elapsed:.2f}s  md5={digest}")
    return elapsed, digest


if __name__ == "__main__":
    print("=== WEPPCLIFF benchmark ===")
    bench_smart_window_sum()
    bench_autotest1()