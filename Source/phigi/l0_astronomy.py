"""L0 deterministic clear-sky solar radiation (FAO-56 style)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# WEPPCLIFF uses Langley (cal/cm²); 1 Langley = 41840 J/m².
_JOULES_PER_LANGLEY = 41840.0


def day_of_year_from_yyyymmdd(yyyymmdd: str | int) -> int:
    """Return 1-based day of year from a YYYYMMDD string."""
    s = str(yyyymmdd)
    dt = pd.Timestamp(year=int(s[:4]), month=int(s[4:6]), day=int(s[6:8]))
    return int(dt.dayofyear)


def extraterrestrial_radiation_mj_m2_day(lat_deg: float, day_of_year: int) -> float:
    """
    Extraterrestrial radiation Ra [MJ/m²/day] — FAO-56 equation 21.

    Valid for 0.5° ≤ |lat| ≤ 90°; polar night returns 0.
    """
    phi = math.radians(float(lat_deg))
    j = int(day_of_year)
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * j / 365.0)
    delta = 0.409 * math.sin(2.0 * math.pi * j / 365.0 - 1.39)
    tan_prod = math.tan(phi) * math.tan(delta)
    if tan_prod <= -1.0:
        ws = math.pi
    elif tan_prod >= 1.0:
        return 0.0
    else:
        ws = math.acos(-tan_prod)
    gsc = 0.0820  # MJ m⁻² min⁻¹
    ra = (
        (24.0 * 60.0 / math.pi)
        * gsc
        * dr
        * (
            ws * math.sin(phi) * math.sin(delta)
            + math.cos(phi) * math.cos(delta) * math.sin(ws)
        )
    )
    return max(0.0, ra)


def clear_sky_radiation_langleys(
    lat_deg: float,
    day_of_year: int,
    *,
    elev_m: float = 0.0,
) -> float:
    """
    Clear-sky daily surface radiation R^cs [Langley/day].

    Uses FAO-56 eq. 37: Rso = (0.75 + 2e-5 * elev) * Ra, then converts MJ → Langley.
    """
    ra = extraterrestrial_radiation_mj_m2_day(lat_deg, day_of_year)
    if ra <= 0.0:
        return 0.0
    rso_mj = (0.75 + 2.0e-5 * float(elev_m)) * ra
    return float(rso_mj * 1.0e6 / _JOULES_PER_LANGLEY)


def clear_sky_series_langleys(
    yyyymmdd: NDArray[np.str_] | list[str] | pd.Series,
    *,
    lat_deg: float,
    elev_m: float = 0.0,
) -> NDArray[np.float64]:
    """Vectorized clear-sky radiation for calendar days."""
    dates = np.asarray(yyyymmdd, dtype=str)
    out = np.zeros(len(dates), dtype=np.float64)
    for i, ymd in enumerate(dates):
        doy = day_of_year_from_yyyymmdd(ymd)
        out[i] = clear_sky_radiation_langleys(lat_deg, doy, elev_m=elev_m)
    return out