"""Hold-out scoring for PHIGI vs legacy comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def rmse(actual: NDArray[np.float64], predicted: NDArray[np.float64], mask: NDArray[np.bool_] | None = None) -> float:
    a = np.asarray(actual, dtype=np.float64)
    p = np.asarray(predicted, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(p)
    if mask is not None:
        m &= np.asarray(mask, dtype=bool)
    if not m.any():
        return float("nan")
    diff = a[m] - p[m]
    return float(np.sqrt(np.mean(diff * diff)))


def event_brier_score(
    actual_event: NDArray[np.float64],
    predicted_prob: NDArray[np.float64],
    *,
    mask: NDArray[np.bool_] | None = None,
) -> float:
    """Brier score on binary EVENT (lower is better)."""
    y = np.asarray(actual_event, dtype=np.float64)
    p = np.clip(np.asarray(predicted_prob, dtype=np.float64), 0.0, 1.0)
    m = np.isfinite(y) & np.isfinite(p)
    if mask is not None:
        m &= np.asarray(mask, dtype=bool)
    if not m.any():
        return float("nan")
    return float(np.mean((p[m] - y[m]) ** 2))


def monthly_precip_mass_bias(
    dates: NDArray[np.str_],
    actual_precip: NDArray[np.float64],
    predicted_precip: NDArray[np.float64],
    *,
    mask: NDArray[np.bool_] | None = None,
) -> float:
    """Mean relative monthly mass bias (|bias| fraction; lower is better)."""
    ymd = np.asarray(dates, dtype=str)
    yyyymm = np.array([d[:6] for d in ymd if len(d) >= 6], dtype=str)
    if len(yyyymm) != len(ymd):
        yyyymm = np.array([d[:6] if len(d) >= 6 else "000000" for d in ymd], dtype=str)

    a = np.asarray(actual_precip, dtype=np.float64)
    p = np.asarray(predicted_precip, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(p)
    if mask is not None:
        m &= np.asarray(mask, dtype=bool)
    if not m.any():
        return float("nan")

    frame = pd.DataFrame({"YYYYMM": yyyymm[m], "actual": a[m], "pred": p[m]})
    monthly = frame.groupby("YYYYMM", as_index=False)[["actual", "pred"]].sum(min_count=1)
    actual_sum = monthly["actual"].to_numpy(dtype=np.float64)
    pred_sum = monthly["pred"].to_numpy(dtype=np.float64)
    denom = np.maximum(np.abs(actual_sum), 1e-6)
    rel = np.abs(pred_sum - actual_sum) / denom
    return float(np.nanmean(rel))


def compare_metric_table(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a side-by-side metrics frame from compare_phigi_legacy output."""
    return pd.DataFrame(rows)