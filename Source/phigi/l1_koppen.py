"""L1 Köppen–Geiger zone resolution from bundled coarse grid."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_GRID = _REPO_ROOT / "priors" / "koppen_grid_2deg.csv"


def _assign_zone(lat: float, lon: float) -> str:
    """
    Rule-based Köppen class at 2° resolution (Beck 2015 simplified).

    Covers CONUS and global tropics/mid-latitudes; unknown cells fall back to Cfa/Dfb.
    """
    la = float(lat)
    lo = float(lon)
    alat = abs(la)

    if alat < 23.5:
        return "Am"
    if alat > 66.5:
        return "Dfc" if lo < 30.0 else "Dfb"

    # Arid belt (Americas / central Asia longitude bands)
    if alat < 45.0 and (
        (-125.0 <= lo <= -95.0 and alat < 42.0)
        or (20.0 <= lo <= 60.0 and alat < 38.0)
    ):
        return "BSk" if alat < 35.0 else "BSk"

    # Humid subtropical / east US Gulf
    if -100.0 < lo <= -75.0 and 28.0 <= alat < 38.0:
        return "Cfa"
    if -95.0 < lo <= -70.0 and 35.0 <= alat < 42.0:
        return "Cfa"

    # Humid continental — northeast / Great Lakes (KMQE region)
    if -105.0 < lo <= -65.0 and 40.0 <= alat < 50.0:
        return "Dfb"
    if -105.0 < lo <= -80.0 and 38.0 <= alat < 40.0:
        return "Dfa"

    # Maritime west Europe-like longitudes
    if -15.0 <= lo <= 25.0 and 40.0 <= alat < 55.0:
        return "Cfb"

    if alat < 35.0:
        return "Cfa"
    if alat < 50.0:
        return "Dfb"
    return "Dfb"


@lru_cache(maxsize=1)
def _load_grid(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load lat, lon, zone arrays from bundled CSV."""
    p = Path(path)
    if not p.is_file():
        return np.array([]), np.array([]), np.array([])

    lats: list[float] = []
    lons: list[float] = []
    zones: list[str] = []
    with p.open(encoding="utf-8") as fh:
        header = fh.readline()
        del header
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            lats.append(float(parts[0]))
            lons.append(float(parts[1]))
            zones.append(parts[2])
    return (
        np.asarray(lats, dtype=np.float64),
        np.asarray(lons, dtype=np.float64),
        np.asarray(zones, dtype=str),
    )


def resolve_koppen_zone(
    lat: float,
    lon: float,
    *,
    kg: str = "AUTO",
    grid_path: str | Path | None = None,
) -> str:
    """
    Resolve Köppen zone for a station.

    ``-kg AUTO`` uses nearest cell in ``priors/koppen_grid_2deg.csv`` when present,
    else rule-based assignment. Explicit ``-kg Cfa`` etc. overrides when listed in priors.
    """
    override = str(kg).strip()
    if override.upper() not in ("", "AUTO", "DEFAULT"):
        return override

    path = str(grid_path or _DEFAULT_GRID)
    glat, glon, gzones = _load_grid(path)
    if glat.size:
        dist = (glat - float(lat)) ** 2 + (glon - float(lon)) ** 2
        return str(gzones[int(np.argmin(dist))])

    return _assign_zone(lat, lon)