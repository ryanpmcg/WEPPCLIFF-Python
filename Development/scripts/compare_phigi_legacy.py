#!/usr/bin/env python3
"""Side-by-side PHIGI vs legacy hold-out metrics on synthetic panels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Source.config import WeppcliffConfig  # noqa: E402
from Source.phigi.diagnose import mask_sorad_holdout, sorad_rmse  # noqa: E402
from Source.phigi.holdout_metrics import rmse  # noqa: E402
from Source.phigi.l3_daily import impute_daily_posterior  # noqa: E402
from Source.phigi.l3_tau import impute_daily_solar_radiation  # noqa: E402
from Source.phigi.l4_precip_hurdle import impute_precip_hurdle, score_precip_holdout  # noqa: E402
from Source.phigi.dates import as_yyyymmdd_strings  # noqa: E402


def _synthetic_panel(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    ymd = dates.strftime("%Y%m%d")
    cluster = rng.integers(11, 44, size=n)
    return pd.DataFrame(
        {
            "YYYYMMDD": ymd,
            "CLUSTER": cluster.astype(float),
            "SO_RAD": rng.uniform(250, 450, size=n),
            "MIN_TEMP": rng.uniform(-5, 10, size=n),
            "MAX_TEMP": rng.uniform(8, 28, size=n),
            "DP_TEMP": rng.uniform(-2, 15, size=n),
            "W_VEL": rng.uniform(1, 8, size=n),
            "W_DIR": rng.uniform(0, 360, size=n),
            "EVENT": (rng.random(size=n) < 0.18).astype(float),
            "PRECIP": np.where(rng.random(size=n) < 0.18, rng.uniform(0.2, 8, size=n), 0.0),
        }
    )


def _legacy_mean_fill(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = panel.copy()
    for col in cols:
        if col not in out.columns:
            continue
        for cluster in out["CLUSTER"].dropna().unique():
            loc = out.index[out["CLUSTER"] == cluster]
            mean = out.loc[loc, col].mean()
            out.loc[loc, col] = out.loc[loc, col].fillna(mean)
        out[col] = out[col].fillna(out[col].mean())
    return out


def compare_holdout(*, fraction: float = 0.15, seed: int = 0) -> dict[str, object]:
    panel = _synthetic_panel(seed=seed)
    truth = panel.copy()
    config = WeppcliffConfig(gf="phigi", lat=42.0, lon=-78.0, kg="Dfb", pm="f")
    config.seed = seed

    masked_sorad, holdout = mask_sorad_holdout(panel["SO_RAD"], fraction=fraction, seed=seed)
    work = panel.copy()
    work["SO_RAD"] = masked_sorad

    dly = work[["CLUSTER", "SO_RAD", "MIN_TEMP", "MAX_TEMP", "DP_TEMP", "W_VEL", "W_DIR", "EVENT"]].copy()
    dates = work["YYYYMMDD"]

    phigi_solar = impute_daily_solar_radiation(
        dly[["CLUSTER", "SO_RAD"]].copy(),
        dates,
        lat_deg=42.0,
        elev_m=200.0,
        config=config,
    )
    phigi_daily = impute_daily_posterior(
        dly.copy(),
        dates,
        lat_deg=42.0,
        elev_m=200.0,
        config=config,
    )
    phigi_daily["SO_RAD"] = phigi_solar["SO_RAD"].to_numpy()

    legacy = work.copy()
    legacy = _legacy_mean_fill(legacy, ["SO_RAD", "MIN_TEMP", "MAX_TEMP", "DP_TEMP", "W_VEL", "W_DIR"])

    mask = holdout.to_numpy(dtype=bool)
    ymd = as_yyyymmdd_strings(dates)

    metrics = {
        "sorad_rmse_phigi": sorad_rmse(truth["SO_RAD"], phigi_daily["SO_RAD"], holdout),
        "sorad_rmse_legacy": sorad_rmse(truth["SO_RAD"], legacy["SO_RAD"], holdout),
        "temp_rmse_phigi": rmse(
            truth["MAX_TEMP"].to_numpy(),
            phigi_daily["MAX_TEMP"].to_numpy(),
            mask,
        ),
        "temp_rmse_legacy": rmse(
            truth["MAX_TEMP"].to_numpy(),
            legacy["MAX_TEMP"].to_numpy(),
            mask,
        ),
    }

    precip_truth = truth.copy()
    precip_work = phigi_daily.copy()
    precip_work.loc[mask, "EVENT"] = np.nan
    precip_work.loc[mask, "PRECIP"] = np.nan
    pcp_df = pd.DataFrame({"BPS": truth["EVENT"] * 2, "ELEN": 1, "PRECIP": truth["PRECIP"]})
    var_out, pcp_out = impute_precip_hurdle(
        precip_work,
        pcp_df,
        dates,
        lat_deg=42.0,
        elev_m=200.0,
        config=config,
    )
    probs = np.clip(var_out["EVENT"].to_numpy(), 0.02, 0.98)
    metrics["precip"] = score_precip_holdout(
        dates=ymd,
        actual_event=precip_truth["EVENT"].to_numpy(),
        predicted_event=var_out["EVENT"].to_numpy(),
        predicted_prob=probs,
        actual_precip=precip_truth["PRECIP"].to_numpy(),
        predicted_precip=pcp_out["PRECIP"].to_numpy(),
        holdout_mask=mask,
    )
    metrics["winner_sorad"] = (
        "phigi"
        if metrics["sorad_rmse_phigi"] < metrics["sorad_rmse_legacy"]
        else "legacy"
    )
    metrics["winner_temp"] = (
        "phigi"
        if metrics["temp_rmse_phigi"] < metrics["temp_rmse_legacy"]
        else "legacy"
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare PHIGI vs legacy hold-out metrics.")
    parser.add_argument("--fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    metrics = compare_holdout(fraction=args.fraction, seed=args.seed)
    text = json.dumps(metrics, indent=2)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())