#!/usr/bin/env python3
"""Download NOAA ETOPO 2022 tiles from NCEI/NGDC."""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data"

PRODUCTS = {
    "15s_surface": ("15s/15s_surface_elev_gtif", "surface", 288),
    "15s_bed": ("15s/15s_bed_elev_gtif", "bed", 62),
    "15s_geoid": ("15s/15s_geoid_gtif", "geoid", 288),
    "15s_sid_surface": ("15s/15s_surface_sid_gtif", "surface_sid", 288),
    "30s_surface": ("30s/30s_surface_elev_gtif", "surface", 1),
    "30s_bed": ("30s/30s_bed_elev_gtif", "bed", 1),
    "60s_surface": ("60s/60s_surface_elev_gtif", "surface", 1),
    "60s_bed": ("60s/60s_bed_elev_gtif", "bed", 1),
}

LATS = ["N90", "N75", "N60", "N45", "N30", "N15", "N00", "S15", "S30", "S45", "S60", "S75"]
LONS = [f"W{d:03d}" for d in range(180, 0, -15)] + [f"E{d:03d}" for d in range(0, 180, 15)]


def tile_names(product: str) -> list[str]:
    path, suffix, count = PRODUCTS[product]
    del path
    if count == 1:
        return [f"ETOPO_2022_v1_{product.split('_')[0]}_N90W180_{suffix}.tif"]
    return [f"ETOPO_2022_v1_15s_{lat}{lon}_{suffix}.tif" for lat in LATS for lon in LONS]


def download_one(url: str, dest: Path) -> tuple[str, bool, str]:
    if dest.is_file() and dest.stat().st_size > 0:
        return dest.name, True, "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as fh:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
        tmp.replace(dest)
        return dest.name, True, "ok"
    except urllib.error.HTTPError as exc:
        if tmp.exists():
            tmp.unlink()
        return dest.name, False, f"HTTP {exc.code}"
    except Exception as exc:  # noqa: BLE001
        if tmp.exists():
            tmp.unlink()
        return dest.name, False, str(exc)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "product",
        choices=sorted(PRODUCTS),
        help="ETOPO 2022 layer to fetch",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Data/ETOPO2022"),
        help="Output root directory",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel downloads")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    subpath, suffix, _ = PRODUCTS[args.product]
    res = args.product.split("_")[0]
    out_dir = args.output / args.product
    names = tile_names(args.product)
    print(f"Product: {args.product} ({len(names)} files) -> {out_dir}")

    ok = skip = fail = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = []
        for name in names:
            if res in ("30s", "60s"):
                url = f"{BASE}/{subpath}/{name}"
            else:
                url = f"{BASE}/{subpath}/{name}"
            futures.append(pool.submit(download_one, url, out_dir / name))
        for fut in as_completed(futures):
            name, success, msg = fut.result()
            if msg == "skipped":
                skip += 1
            elif success:
                ok += 1
                print(f"  {name}")
            else:
                fail += 1
                print(f"  FAIL {name}: {msg}", file=sys.stderr)

    print(f"Done: {ok} downloaded, {skip} skipped, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())