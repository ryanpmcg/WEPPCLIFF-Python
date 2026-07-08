"""L5 erosivity validation gates (I30 proxy) with K-candidate selection."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from Source.weppcliff_impute.ei_validate import (
    EIValidationResult,
    i30_proxy,
    validate_i30_against_donors,
)

__all__ = [
    "EIValidationResult",
    "i30_proxy",
    "validate_i30_against_donors",
    "select_best_i30_candidate",
]

_DEFAULT_K = 5


def select_best_i30_candidate(
    candidates: list[pd.DataFrame],
    donor_days: list[pd.DataFrame],
    *,
    min_ratio: float = 0.5,
) -> tuple[pd.DataFrame | None, EIValidationResult | None, int]:
    """
    L5.6: evaluate up to K imputed day sequences; return the first passing candidate.

    Falls back to the candidate with the highest donor I30 ratio.
    """
    if not candidates:
        return None, None, -1

    best_idx = 0
    best_ratio = -1.0
    best_result: EIValidationResult | None = None

    for idx, candidate in enumerate(candidates):
        if candidate.empty:
            continue
        result = validate_i30_against_donors(candidate, donor_days, min_ratio=min_ratio)
        if result.passed:
            return candidate, result, idx
        if result.ratio > best_ratio:
            best_ratio = result.ratio
            best_idx = idx
            best_result = result

    return candidates[best_idx], best_result, best_idx


def generate_k_candidates(
    builder: Callable[[int], pd.DataFrame | None],
    *,
    k: int = _DEFAULT_K,
) -> list[pd.DataFrame]:
    """Build K candidate day sequences from a seeded builder callback."""
    out: list[pd.DataFrame] = []
    for seed in range(k):
        frame = builder(seed)
        if frame is not None and not frame.empty:
            out.append(frame)
    return out