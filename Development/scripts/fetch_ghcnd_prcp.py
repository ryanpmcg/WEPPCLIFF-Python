#!/usr/bin/env python3
"""Download GHCNd daily PRCP for NCDC-Update station IDs.

Writes compact parquet under Export/NCDC-Python-Fill/GHCNd/. Does not
touch R Results or NCDC-Python-2026. Resumes existing station files.
"""

from __future__ import annotations

import gzip
import io
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LEGACY = Path("/Users/home/Software/WEPPCLIFF/NCDC-Update")
OUT = ROOT / "Export" / "NCDC-Python-Fill" / "GHCNd"
URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{stem}.csv.gz"
WORKERS = 16
RETRIES = 4


def station_stems() -> list[str]:
    stems: set[str] = set()
    for folder in (LEGACY / "Inputs" / "Complete", LEGACY / "Inputs" / "Rerun"):
        for path in folder.glob("WEPP_*.csv"):
            stems.add(path.stem.replace("WEPP_", "", 1))
    return sorted(stems)


def _parse_prcp(raw: bytes, stem: str) -> pd.DataFrame:
    rows: list[tuple[str, float]] = []
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as fh:
        for line in fh:
            parts = line.decode("utf-8", errors="replace").rstrip("\n").split(",")
            if len(parts) < 4 or parts[2] != "PRCP":
                continue
            qflag = parts[5] if len(parts) > 5 else ""
            if qflag not in {"", " "}:
                continue
            try:
                value = int(parts[3])
            except ValueError:
                continue
            if value == -9999:
                continue
            date = parts[1]
            if len(date) != 8 or not date.isdigit():
                continue
            rows.append((date, value / 10.0))
    if not rows:
        return pd.DataFrame(columns=["YYYYMMDD", "PRCP"])
    frame = pd.DataFrame(rows, columns=["YYYYMMDD", "PRCP"])
    frame["STATION"] = stem
    return frame[["STATION", "YYYYMMDD", "PRCP"]]


def fetch_one(stem: str) -> tuple[str, str, int]:
    dest = OUT / f"{stem}.parquet"
    if dest.exists() and dest.stat().st_size > 0:
        return stem, "exists", 0
    url = URL.format(stem=stem)
    last_err = ""
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "WEPPCLIFF-GHCNd-fetch/2.0"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
            frame = _parse_prcp(raw, stem)
            dest.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(dest, index=False)
            return stem, "ok", len(frame)
        except urllib.error.HTTPError as exc:
            last_err = f"HTTP {exc.code}"
            if exc.code == 404:
                return stem, "missing", 0
            time.sleep(1.5 * (attempt + 1))
        except Exception as exc:
            last_err = f"{type(exc).__name__}:{exc}"
            time.sleep(1.5 * (attempt + 1))
    return stem, last_err, 0


def main() -> int:
    stems = station_stems()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Fetching GHCNd PRCP for {len(stems)} stations -> {OUT}", flush=True)
    ok = missing = failed = skipped = 0
    rows = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(fetch_one, stem): stem for stem in stems}
        done = 0
        for fut in as_completed(futs):
            stem, status, n = fut.result()
            done += 1
            if status == "ok":
                ok += 1
                rows += n
            elif status == "exists":
                skipped += 1
            elif status == "missing":
                missing += 1
            else:
                failed += 1
                print(f"  FAIL {stem}: {status}", flush=True)
            if done % 100 == 0 or done == len(stems):
                print(
                    f"  {done}/{len(stems)} ok={ok} skip={skipped} "
                    f"missing={missing} fail={failed}",
                    flush=True,
                )
    elapsed = time.perf_counter() - t0
    print(
        f"Done in {elapsed/60:.1f} min. new={ok} cached={skipped} "
        f"no_ghcn={missing} fail={failed} new_rows={rows}",
        flush=True,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
