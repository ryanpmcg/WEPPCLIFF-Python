"""Optional PyMICE directed engine for L3 daily residuals."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from Source.phigi.l0_astronomy import clear_sky_series_langleys, day_of_year_from_yyyymmdd
from Source.phigi.dates import as_yyyymmdd_strings
from Source.phigi.l3_tau import compute_tau

if TYPE_CHECKING:
    from Source.config import WeppcliffConfig
    from Source.phigi.report import ImputationReport

_DAILY_VARS = ("MIN_TEMP", "MAX_TEMP", "DP_TEMP", "W_VEL", "W_DIR")
_MIN_ROWS = 12


def _pymice_available() -> bool:
    try:
        from Source.weppcliff_impute.adapter import _run_mice_on_frame  # noqa: F401

        return True
    except ImportError:
        return False


def pymice_directed_daily_fill(
    dly_var_df: pd.DataFrame,
    yyyymmdd: pd.Series,
    *,
    lat_deg: float,
    elev_m: float,
    config: WeppcliffConfig,
    report: ImputationReport | None = None,
    imp_method: str = "midastouch",
) -> pd.DataFrame:
    """
    Fill remaining daily gaps with PyMICE using τ and DOY as directed predictors.

    Runs per CLUSTER when ``-pm t`` and enough complete rows exist.
    """
    from Source.phigi.report import ImputationReport

    if str(config.pm).upper() != "T" or not _pymice_available():
        return dly_var_df

    from Source.weppcliff_impute.adapter import _config_from_any, _run_mice_on_frame

    rep = report or ImputationReport()
    out = dly_var_df.copy()
    dates = as_yyyymmdd_strings(yyyymmdd)
    r_cs = clear_sky_series_langleys(dates.tolist(), lat_deg=lat_deg, elev_m=elev_m)
    so_rad = pd.to_numeric(out.get("SO_RAD"), errors="coerce").to_numpy(dtype=np.float64)
    tau = compute_tau(so_rad, r_cs)
    doy = np.array([day_of_year_from_yyyymmdd(d) for d in dates], dtype=np.float64)

    value_cols = [c for c in _DAILY_VARS if c in out.columns]
    if not value_cols:
        return out

    cfg = _config_from_any(config)
    n_filled = 0

    for cluster in out["CLUSTER"].dropna().unique():
        loc = out.index[out["CLUSTER"] == cluster]
        block = out.loc[loc, value_cols].copy()
        need_any = block.isna().any(axis=1)
        if not need_any.any():
            continue

        work = block.copy()
        work["TAU"] = tau[loc.to_numpy()]
        work["DOY"] = doy[loc.to_numpy()]
        complete = work[value_cols + ["TAU", "DOY"]].notna().all(axis=1).sum()
        if complete < _MIN_ROWS:
            continue

        try:
            filled = _run_mice_on_frame(work, imp_method=imp_method, config=cfg)
        except Exception as exc:
            rep.notes.append(f"L3 PyMICE directed skipped cluster {cluster}: {exc}")
            continue

        for col in value_cols:
            was = out.loc[loc, col].isna()
            out.loc[loc, col] = filled[col].to_numpy()
            n_filled += int((was & out.loc[loc, col].notna()).sum())

    if n_filled:
        if "pymice_directed" not in rep.strategies:
            rep.strategies.append("pymice_directed")
        rep.notes.append(f"L3 PyMICE directed filled {n_filled} cells")
    return out