"""Erosion index calculations — port of R calculate_erosion_indices."""

from __future__ import annotations

import math
from typing import Callable, List

import numpy as np
import pandas as pd

from Source.config import WeppcliffConfig
from Source.data_model import WeppcliffData

EQN_FUNCS: dict[str, Callable] = {
    "AH282": lambda i: 0.119 + 0.0873 * np.log10(np.maximum(i, 1e-12)),
    "AH537": lambda i: 0.119 + 0.0873 * np.log10(np.maximum(i, 1e-12)),
    "AH703": lambda i: 0.119 + 0.0873 * np.log10(np.maximum(i, 1e-12)),
    "BF": lambda i: 0.29 * (1 - 0.72 * np.exp(-0.05 * i)),
    "MM": lambda i: 0.273 + 0.2168 * np.exp(-0.048 * i) - 0.4126 * np.exp(-0.072 * i),
    "R2": lambda i: 0.29 * (1 - 0.72 * np.exp(-0.082 * i)),
}


def _select_equations(config: WeppcliffConfig) -> List[str]:
    ee = str(config.ee).upper()
    if ee in ("ALL", "ARS"):
        return ["AH282", "AH537", "AH703", "BF", "MM", "R2"]
    return [ee]


def _user_equation(config: WeppcliffConfig) -> Callable:
    expr = config.parse_user_equation()
    if not expr:
        expr = "0.119 + 0.0873 * log10(i)"
    code = compile(expr, "<user_eq>", "eval")
    return lambda i, c=code: eval(  # noqa: S307
        c,
        {"__builtins__": {}},
        {"i": i, "log10": math.log10, "exp": math.exp, "np": np},
    )


def _storm_metrics(storms: List[pd.DataFrame]) -> dict[str, np.ndarray]:
    """Precompute per-storm summary metrics used by all erosion equations."""
    n = len(storms)
    dtimes = np.empty(n, dtype=object)
    depths = np.empty(n, dtype=np.float64)
    durs = np.empty(n, dtype=np.float64)
    pdurs = np.empty(n, dtype=np.float64)
    ips = np.empty(n, dtype=np.float64)
    itws = np.empty(n, dtype=np.float64)
    idws = np.empty(n, dtype=np.float64)
    i30s = np.empty(n, dtype=np.float64)
    tps = np.empty(n, dtype=np.float64)

    for idx, x in enumerate(storms):
        int_vals = x["INT"].astype(float).values
        dur_vals = x["DUR"].astype(float).values
        pcp_vals = x["PRECIP"].astype(float).values
        cum_vals = x["CUM"].astype(float).values

        dtimes[idx] = pd.to_datetime(x["DT_1"].iloc[0]).strftime("%Y-%m-%d %H:%M:%S")
        depths[idx] = float(pcp_vals.sum())
        dur = float(cum_vals.max() - cum_vals.min() + dur_vals[0])
        durs[idx] = dur
        pdurs[idx] = float(dur_vals.sum())
        ips[idx] = float(int_vals.max())
        itws[idx] = float(np.average(int_vals, weights=dur_vals))
        w = np.where(pcp_vals == 0, np.nan, pcp_vals)
        idws[idx] = float(np.average(int_vals, weights=w)) if np.any(~np.isnan(w)) else 0.0

        bef = x["INT_BEF_30"].max() if "INT_BEF_30" in x.columns else 0.0
        aft = x["INT_AFT_30"].max() if "INT_AFT_30" in x.columns else 0.0
        i30s[idx] = float(max(bef, aft))

        t_vec = cum_vals - cum_vals.min() + dur_vals[0]
        tp_idx = int_vals == int_vals.max()
        tps[idx] = float(np.median(t_vec[tp_idx]) / t_vec.max()) if t_vec.max() else 0.0

    i30_adj = i30s.copy()
    i30_adj[durs < 30] = 2 * depths[durs < 30]

    return {
        "dtimes": dtimes,
        "depths": depths,
        "durs": durs,
        "pdurs": pdurs,
        "ips": ips,
        "itws": itws,
        "idws": idws,
        "i30s": i30_adj,
        "tps": tps,
    }


def calculate_erosion_indices(
    data: WeppcliffData,
    storm_list: int,
    storm_df: int,
    config: WeppcliffConfig,
) -> WeppcliffData:
    config.verbose("Calculating erosion indices...")
    storms: List[pd.DataFrame] = data[storm_list]
    eqn = _select_equations(config)
    eqn_funcs = dict(EQN_FUNCS)
    if "USER" in eqn or str(config.ee).upper() == "USER":
        eqn_funcs["USER"] = _user_equation(config)

    for name in eqn:
        func = eqn_funcs.get(name, eqn_funcs["R2"])
        cap = 76 if name in ("AH537", "AH703") else None
        updated: List[pd.DataFrame] = []
        for sdf in storms:
            i = sdf["INT"].astype(float).values
            if cap is not None:
                i = np.minimum(i, cap)
            ed = func(i)
            ke = ed * sdf["PRECIP"].astype(float).values
            updated.append(sdf.assign(**{f"{name}_ED": ed, f"{name}_KE": ke}))
        storms = updated
    data[storm_list] = storms

    metrics = _storm_metrics(storms)
    depths_arr = metrics["depths"]
    durs_arr = metrics["durs"]
    i30_arr = metrics["i30s"]

    sp = pd.DataFrame({
        "DT": metrics["dtimes"],
        "DEPTH": depths_arr,
        "DUR": durs_arr,
        "PDUR": metrics["pdurs"],
        "PRATIO": metrics["pdurs"] / np.where(durs_arr == 0, 1, durs_arr),
        "PEAKINT": metrics["ips"],
        "PEAKTIME": metrics["tps"],
        "TIME_WM_INT": metrics["itws"],
        "DEPTH_WM_INT": metrics["idws"],
        "I30": i30_arr,
    })
    sp["ANT"] = pd.to_datetime(sp["DT"]).diff().dt.total_seconds() / 60

    ke_cols: dict[str, list] = {}
    ei_cols: dict[str, list] = {}
    for name in eqn:
        ke_name = f"{name}_KE"
        ke_vals = np.array([float(s[ke_name].sum()) for s in storms])
        ke_cols[ke_name] = ke_vals
        max_i30 = i30_arr.copy()
        if name == "AH537":
            max_i30[max_i30 > 64] = 64
        ei_cols[f"{name}_EI"] = (max_i30 * ke_vals).tolist()

    out = pd.concat([sp, pd.DataFrame(ke_cols), pd.DataFrame(ei_cols)], axis=1)
    for col in out.columns[1:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    data[storm_df] = out
    return data