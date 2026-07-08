"""YYYYMMDD normalization for gap-fill paths (CSV, CLI, float dtypes)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def as_yyyymmdd_strings(values: pd.Series | NDArray | list | object) -> NDArray[np.str_]:
    """Normalize date cells (int/float/str) to ``YYYYMMDD`` strings."""
    out: list[str] = []
    for v in np.asarray(values, dtype=object).ravel():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append("")
            continue
        if isinstance(v, (int, np.integer)):
            out.append(f"{int(v):08d}")
            continue
        if isinstance(v, float):
            out.append(f"{int(v):08d}")
            continue
        s = str(v).strip()
        if s.endswith(".0") and s[:-2].isdigit():
            s = s[:-2]
        if len(s) == 8 and s.isdigit():
            out.append(s)
            continue
        parsed = pd.to_datetime(s, errors="coerce")
        if pd.isna(parsed):
            out.append(s)
        else:
            out.append(parsed.strftime("%Y%m%d"))
    return np.asarray(out, dtype=str)


def coerce_yyyymmdd_series(series: pd.Series) -> pd.Series:
    """Return a string ``YYYYMMDD`` series safe for merges and slicing."""
    return pd.Series(as_yyyymmdd_strings(series), index=series.index, dtype=str)


def month_from_yyyymmdd(dates: NDArray[np.str_]) -> NDArray[np.int_]:
    """Extract month (1–12) from normalized date strings."""
    normalized = as_yyyymmdd_strings(dates)
    months = np.ones(len(normalized), dtype=int)
    for i, d in enumerate(normalized):
        if len(d) >= 6 and d[:6].isdigit():
            months[i] = int(d[4:6])
    return months