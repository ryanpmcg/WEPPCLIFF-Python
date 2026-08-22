"""Tests for PHIGI hardening (L3–L6) additions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.config import WeppcliffConfig  # noqa: E402
from Source.phigi.holdout_metrics import event_brier_score  # noqa: E402
from Source.phigi.l3_tau import _impute_tau  # noqa: E402
from Source.phigi.l4_precip_hurdle import _fit_logistic_wet_coeffs, _lognormal_mean  # noqa: E402
from Source.phigi.l5_ei_validate import select_best_i30_candidate  # noqa: E402
from Source.phigi.report import ImputationReport  # noqa: E402


def test_tau_ar1_uses_previous_day() -> None:
    tau = np.array([0.8, 0.5, np.nan], dtype=np.float64)
    dates = np.array(["20190601", "20190602", "20190603"], dtype=str)
    cluster = np.array([11, 11, 11], dtype=float)
    out, n_fill, _ = _impute_tau(tau, dates, cluster, lat_deg=42.0, lon_deg=-78.0, kg="Dfb")
    cluster_only = float(np.nanmean([0.8, 0.5]))
    assert n_fill == 1
    assert out[2] == pytest.approx(0.45 * 0.5 + 0.55 * cluster_only)
    assert out[2] != pytest.approx(cluster_only)


def test_logistic_coeffs_fit_on_synthetic() -> None:
    n = 120
    event = (np.linspace(0, 1, n) > 0.6).astype(float)
    tau = np.linspace(0.3, 0.9, n)
    months = np.array([(i % 12) + 1 for i in range(n)], dtype=int)
    prev = np.zeros(n, dtype=bool)
    prev[1:] = event[:-1] == 1
    coeffs = _fit_logistic_wet_coeffs(event, tau, months, prev)
    assert coeffs is not None
    assert len(coeffs) == 14


def test_lognormal_mean_positive_only() -> None:
    vals = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
    mean = _lognormal_mean(vals)
    assert mean > 0


def test_select_best_i30_candidate_prefers_passing() -> None:
    donors = [
        pd.DataFrame(
            {
                "DT_1": pd.date_range("2020-06-01", periods=3, freq="15min"),
                "DUR": [15.0, 15.0, 15.0],
                "PRECIP": [1.0, 2.0, 3.0],
            }
        )
    ]
    good = donors[0].copy()
    bad = donors[0].copy()
    bad["PRECIP"] = 0.01
    best, result, idx = select_best_i30_candidate([bad, good], donors)
    assert idx == 1
    assert result is not None
    assert result.passed


def test_report_mark_low_confidence() -> None:
    rep = ImputationReport()
    rep.mark_low_confidence("test fallback")
    assert rep.confidence_tier == "low"
    assert "test fallback" in rep.notes


def test_event_brier_score() -> None:
    actual = np.array([1.0, 0.0, 1.0])
    prob = np.array([0.9, 0.1, 0.4])
    score = event_brier_score(actual, prob)
    assert 0.0 <= score <= 1.0


@pytest.mark.slow
def test_fair_compare_holdout_random_smoke() -> None:
    from Source.phigi.fair_compare import CompareScenario, compare_engines_on_kmqe

    root = Path(__file__).resolve().parents[2]
    kmqe = root / "Input" / "ASOS_BPT_KMQE.csv"
    if not kmqe.exists():
        pytest.skip("KMQE input missing")

    result = compare_engines_on_kmqe(
        kmqe,
        CompareScenario.HOLDOUT_RANDOM,
        fraction=0.02,
        seed=0,
    )
    assert result.scenario == CompareScenario.HOLDOUT_RANDOM.value
    assert result.phigi.n_holdout_days > 0
    assert "rmse_so_rad" in result.winners
    assert result.phigi.daily_rmse["SO_RAD"] >= 0