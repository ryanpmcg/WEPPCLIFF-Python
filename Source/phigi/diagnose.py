"""Hold-out masking utilities for PHIGI validation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mask_sorad_holdout(
    so_rad: pd.Series,
    *,
    fraction: float = 0.15,
    seed: int = 0,
) -> tuple[pd.Series, pd.Series]:
    """
    Randomly mask a fraction of observed SO_RAD for hold-out validation.

    Returns (masked_series, boolean holdout_mask where True = held out).
    """
    rng = np.random.default_rng(seed)
    values = pd.to_numeric(so_rad, errors="coerce")
    observed = values.notna()
    idx = np.where(observed.to_numpy())[0]
    n_mask = max(1, int(len(idx) * fraction))
    chosen = rng.choice(idx, size=min(n_mask, len(idx)), replace=False)
    mask = pd.Series(False, index=so_rad.index)
    mask.iloc[chosen] = True
    masked = values.copy()
    masked.loc[mask] = np.nan
    return masked, mask


def sorad_rmse(actual: pd.Series, predicted: pd.Series, holdout: pd.Series) -> float:
    """RMSE on held-out cells only."""
    a = pd.to_numeric(actual, errors="coerce")
    p = pd.to_numeric(predicted, errors="coerce")
    m = holdout.astype(bool) & a.notna() & p.notna()
    if not m.any():
        return float("nan")
    diff = a.loc[m] - p.loc[m]
    return float(np.sqrt(np.mean(diff * diff)))