#!/usr/bin/env python3
"""Stage-level diagnosis of Python vs R missing-event fill on USC00010008."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

from Source.config import WeppcliffConfig, set_config  # noqa: E402
from Source.impute import (  # noqa: E402
    attach_event_dates,
    binarize_filled_event,
    empty_breakpoint_frame,
    find_missing_precip,
)
from Source.phigi.dates import coerce_yyyymmdd_series  # noqa: E402
from Source.phigi.fair_compare import build_pre_gapfill_data  # noqa: E402
from Source.weppcliff_impute.setup import build_mice_plan  # noqa: E402
from Source.weppcliff_impute.cluster import select_cluster_granularity  # noqa: E402
import Source.impute as impute_mod  # noqa: E402


def _cfg():
    inp = Path("/Users/home/Software/WEPPCLIFF/NCDC-Update/Inputs/Complete/WEPP_USC00010008.csv")
    cfg = WeppcliffConfig(
        pm="t",
        gf="legacy",
        st="f",
        qi="t",
        verb="f",
        io=100,
        d=str(inp.parent),
        f=inp.name,
        id="t",
        qc="t",
        ei="t",
    )
    set_config(cfg)
    cfg.apply_runtime_flags()
    return inp, cfg


def summarize(name: str, s: pd.Series) -> str:
    x = pd.to_numeric(s, errors="coerce")
    x = x.dropna()
    if x.empty:
        return f"{name}: empty"
    qs = x.quantile([0.1, 0.5, 0.9]).to_list()
    return (
        f"{name}: n={len(x)} mean={float(x.mean()):.3f} "
        f"p10={qs[0]:.3f} p50={qs[1]:.3f} p90={qs[2]:.3f} max={float(x.max()):.3f}"
    )


def main() -> None:
    inp, cfg = _cfg()
    rec: dict = {}

    orig_find = impute_mod.find_missing_precip

    def wrap_find(df):
        out = orig_find(df)
        rec["missing_event_na"] = int(out["EVENT"].isna().sum())
        rec["missing_event_0"] = int((pd.to_numeric(out["EVENT"], errors="coerce") == 0).sum())
        rec["missing_event_1"] = int((pd.to_numeric(out["EVENT"], errors="coerce") == 1).sum())
        rec["n_clusters_daily"] = int(pd.to_numeric(out["CLUSTER"], errors="coerce").nunique())
        rec["cluster_ids"] = sorted(pd.to_numeric(out["CLUSTER"], errors="coerce").dropna().unique().tolist())
        return out

    impute_mod.find_missing_precip = wrap_find

    orig_bin = impute_mod.binarize_filled_event

    def wrap_bin(filled, observed):
        rec["event_raw"] = pd.to_numeric(filled, errors="coerce").copy()
        rec["event_obs"] = pd.to_numeric(observed, errors="coerce").copy()
        out = orig_bin(filled, observed)
        rec["event_bin"] = out.copy()
        return out

    impute_mod.binarize_filled_event = wrap_bin

    orig_attach = impute_mod.attach_event_dates

    def wrap_attach(imp_df, event_dates):
        out = orig_attach(imp_df, event_dates)
        rec["dly_imp"] = out.copy()
        return out

    impute_mod.attach_event_dates = wrap_attach

    orig_empty = impute_mod.empty_breakpoint_frame
    empty_nrows: list[int] = []
    empty_ymd: list[str] = []

    def wrap_empty(template, n_rows, *, yyyymmdd, cluster):
        empty_nrows.append(int(n_rows))
        empty_ymd.append(str(yyyymmdd))
        return orig_empty(template, n_rows, yyyymmdd=yyyymmdd, cluster=cluster)

    impute_mod.empty_breakpoint_frame = wrap_empty

    print("build_pre_gapfill...", flush=True)
    data = build_pre_gapfill_data(inp, cfg)
    dly_pre = data.slot_df(10).copy()
    pcp_pre = data.slot_df(5).copy()
    pre_p = pd.to_numeric(dly_pre["PRECIP"], errors="coerce")
    pre_wet_dates = set(coerce_yyyymmdd_series(dly_pre.loc[pre_p > 0, "YYYYMMDD"]))
    print(
        f"pre wet={int((pre_p > 0).sum())} mass={float(pre_p.sum()):.3f} "
        f"bp={len(pcp_pre)} gran={getattr(cfg, 'impute_granularity', None)}",
        flush=True,
    )

    print("fill_data...", flush=True)
    from Source.impute import fill_data

    filled = fill_data(data, 5, 10, 18, 15)
    dly = filled.slot_df(18)
    pcp = filled.slot_df(15)
    post_p = pd.to_numeric(dly["PRECIP"], errors="coerce")
    post_bps = pd.to_numeric(dly["BPS"], errors="coerce").fillna(0)
    bp_p = pd.to_numeric(pcp["PRECIP"], errors="coerce")

    r_daily = pd.read_csv(
        "/Users/home/Software/WEPPCLIFF/NCDC-Update/Results/Daily/USC00010008 DAILY DATA.csv"
    )
    r_p = pd.to_numeric(r_daily["PRECIP"], errors="coerce")
    r_bps = pd.to_numeric(r_daily["BPS"], errors="coerce")
    r_ymd = coerce_yyyymmdd_series(r_daily["YYYYMMDD"])
    py_ymd = coerce_yyyymmdd_series(dly["YYYYMMDD"])

    print("\n=== STAGE: missing EVENT scan ===")
    print(
        f"EVENT NA={rec.get('missing_event_na')} 0={rec.get('missing_event_0')} "
        f"1={rec.get('missing_event_1')} n_cluster={rec.get('n_clusters_daily')} "
        f"gran={getattr(cfg, 'impute_granularity', None)}"
    )

    raw = rec["event_raw"]
    obs = rec["event_obs"]
    binned = rec["event_bin"]
    na_mask = obs.isna()
    raw_na = raw[na_mask]
    print("\n=== STAGE: EVENT cart (NA days only) ===")
    print(f"NA days {int(na_mask.sum())}")
    print(summarize("raw EVENT on NA days", raw_na))
    print(
        f"raw unique (rounded) {sorted(set(np.round(raw_na.dropna(), 3)))[:20]} "
        f"n_unique={raw_na.nunique()}"
    )
    print(
        f"binarize 1s on NA days={int((binned[na_mask] == 1).sum())} "
        f"0s={int((binned[na_mask] == 0).sum())} "
        f"frac1={float((binned[na_mask] == 1).mean()):.3f}"
    )
    print(f"total EVENT=1 after bin={int((binned == 1).sum())} observed 1s={int((obs == 1).sum())}")

    imp = rec["dly_imp"]
    new_mask = ~imp["YYYYMMDD"].astype(str).isin(pre_wet_dates)
    new_imp = imp.loc[new_mask]
    obs_imp = imp.loc[~new_mask]
    print("\n=== STAGE: event-characteristic impute (daily BPS/ELEN/PRECIP) ===")
    print(f"event table rows={len(imp)} new={int(new_mask.sum())} observed_wet_kept={int((~new_mask).sum())}")
    print(summarize("new BPS (pre-round)", new_imp["BPS"]))
    print(summarize("new ELEN", new_imp["ELEN"]))
    print(summarize("new daily PRECIP (event-level, before i_pcp overwrite)", new_imp["PRECIP"]))
    print(summarize("observed-wet BPS", obs_imp["BPS"]))
    print(summarize("observed-wet PRECIP (event-level)", obs_imp["PRECIP"]))

    rec["empty_nrows"] = empty_nrows
    rec["empty_ymd"] = empty_ymd
    print("\n=== STAGE: empty breakpoint frames ===")
    print(f"empty frames={len(empty_nrows)} nrows unique={sorted(set(empty_nrows))[:10]} sum_nrows={sum(empty_nrows)}")
    if empty_nrows:
        print(summarize("empty frame nrows", pd.Series(empty_nrows, dtype=float)))
    new_bps_round = pd.to_numeric(new_imp["BPS"], errors="coerce").clip(lower=0).round()
    print(
        f"new-event rounded BPS sum={float(new_bps_round.sum())} "
        f"mean={float(new_bps_round.mean()) if len(new_bps_round) else float('nan'):.2f} "
        f"frac_round_to_0={float((new_bps_round == 0).mean()) if len(new_bps_round) else float('nan'):.3f}"
    )
    print(
        f"BPS used for empty frames is PRE-impute (NA->1). "
        f"If R-style post-impute BPS were used, extra bp rows would be "
        f"{float(new_bps_round.clip(lower=1).sum()) - len(empty_nrows):.0f}"
    )

    print("\n=== STAGE: daily BPS vs breakpoint rows (i_pcp alignment) ===")
    print(
        f"sum daily BPS={float(post_bps.sum())} bp_rows={len(pcp)} "
        f"delta={float(post_bps.sum()) - len(pcp)}"
    )
    print(f"bp PRECIP na={int(bp_p.isna().sum())} >0={int((bp_p > 0).sum())} sum={float(bp_p.sum()):.3f}")
    print(f"daily PRECIP (i_pcp) sum={float(post_p.sum()):.3f} wet={int((post_p > 0).sum())}")

    merged_imp = pd.DataFrame(
        {
            "YYYYMMDD": coerce_yyyymmdd_series(imp["YYYYMMDD"]),
            "event_P": pd.to_numeric(imp["PRECIP"], errors="coerce"),
            "event_BPS": pd.to_numeric(imp["BPS"], errors="coerce").clip(lower=0).round(),
        }
    ).merge(
        pd.DataFrame(
            {
                "YYYYMMDD": py_ymd,
                "ipc_P": post_p,
                "ipc_BPS": post_bps,
            }
        ),
        on="YYYYMMDD",
        how="left",
    )
    new_m = merged_imp["YYYYMMDD"].astype(str).isin([str(x) for x in empty_ymd])
    if not new_m.any():
        new_m = ~merged_imp["YYYYMMDD"].astype(str).isin(pre_wet_dates)
    print(summarize("new days event-level P", merged_imp.loc[new_m, "event_P"]))
    print(summarize("new days i_pcp P", merged_imp.loc[new_m, "ipc_P"]))
    print(
        f"new days event-level mass={float(merged_imp.loc[new_m, 'event_P'].sum()):.3f} "
        f"i_pcp mass={float(merged_imp.loc[new_m, 'ipc_P'].sum()):.3f}"
    )
    print(
        f"new days i_pcp==0 {int((merged_imp.loc[new_m, 'ipc_P'] <= 0).sum())} "
        f"event_P>0 {int((merged_imp.loc[new_m, 'event_P'] > 0).sum())}"
    )

    overlap = pd.DataFrame({"YYYYMMDD": coerce_yyyymmdd_series(dly_pre["YYYYMMDD"]), "pre": pre_p}).merge(
        pd.DataFrame({"YYYYMMDD": py_ymd, "post": post_p}),
        on="YYYYMMDD",
        how="inner",
    )
    wet_pre = overlap["pre"] > 0
    print("\n=== STAGE: observed wet-day mass after fill ===")
    print(
        f"overlap wet-pre mass pre={float(overlap.loc[wet_pre, 'pre'].sum()):.3f} "
        f"post={float(overlap.loc[wet_pre, 'post'].sum()):.3f} "
        f"ratio={float(overlap.loc[wet_pre, 'post'].sum()) / float(overlap.loc[wet_pre, 'pre'].sum()):.4f}"
    )
    zeroed = int(((overlap["pre"] > 0) & (overlap["post"].fillna(0) <= 0)).sum())
    print(f"observed wet days zeroed={zeroed}")

    print("\n=== vs R daily ===")
    both = pd.DataFrame({"YYYYMMDD": r_ymd, "r_p": r_p, "r_bps": r_bps}).merge(
        pd.DataFrame({"YYYYMMDD": py_ymd, "py_p": post_p, "py_bps": post_bps}),
        on="YYYYMMDD",
        how="inner",
    )
    r_wet = both["r_p"] > 0
    py_wet = both["py_p"] > 0
    pre_wet = both["YYYYMMDD"].astype(str).isin(pre_wet_dates)
    print(f"calendar overlap days={len(both)}")
    print(f"R wet={int(r_wet.sum())} Python wet={int(py_wet.sum())} pre wet={int(pre_wet.sum())}")
    print(f"R-only wet (filled by R not Python)={int((r_wet & ~py_wet).sum())}")
    print(f"Python-only wet={int((py_wet & ~r_wet).sum())}")
    print(f"both wet={int((r_wet & py_wet).sum())}")
    print(summarize("R P on R-only wet days", both.loc[r_wet & ~py_wet, "r_p"]))
    print(summarize("R BPS on R-only wet days", both.loc[r_wet & ~py_wet, "r_bps"]))
    print(summarize("R P on both-wet", both.loc[r_wet & py_wet, "r_p"]))
    print(summarize("Python P on both-wet", both.loc[r_wet & py_wet, "py_p"]))
    print(f"R mass={float(r_p.sum()):.3f} Python mass={float(post_p.sum()):.3f}")
    print(
        f"R extra mass vs pre={float(r_p.sum()) - float(pre_p.sum()):.3f} "
        f"Python extra vs pre={float(post_p.sum()) - float(pre_p.sum()):.3f}"
    )

    # mice plan for EVENT-like frame
    print("\n=== mice plan (EVENT cart, scatter/quickpred/qi) ===")
    sample = pd.DataFrame(
        {
            "CLUSTER": np.ones(20),
            "SO_RAD": np.linspace(10, 30, 20),
            "MAX_TEMP": np.linspace(5, 25, 20),
            "MIN_TEMP": np.linspace(-5, 15, 20),
            "DP_TEMP": np.linspace(-8, 12, 20),
            "W_VEL": np.linspace(1, 8, 20),
            "W_DIR": np.linspace(0, 350, 20),
            "EVENT": [1, 0, 1, 0, np.nan] * 4,
        }
    )
    from Source.weppcliff_impute.adapter import _config_from_any, coerce_numeric_impute_frame

    frame = coerce_numeric_impute_frame(sample.drop(columns=["CLUSTER"]))
    plan = build_mice_plan(
        frame,
        imp_method="cart",
        io_cap=100,
        quick_impute=True,
        partition="scatter",
    )
    print("methods", plan["method"])
    print("blocks", {k: v for k, v in plan["blocks"].items()})
    names = plan["column_names"]
    pred = plan["predictor_matrix"]
    if "EVENT" in names:
        j = names.index("EVENT")
        print("EVENT pred row", dict(zip(names, pred[j].tolist())))
        print("EVENT spec", plan["variable_specs"][j])
    print("partition scatter (R uses collect); qi cart maxit<=10 (R midastouch unless -qi)")
    print("R EVENT is factor; Python EVENT is float 0/1")

    print("\n=== clustering ===")
    print(f"config.impute_granularity={getattr(cfg, 'impute_granularity', None)}")
    print(f"n CLUSTER labels on daily scan={rec.get('n_clusters_daily')}")
    print("R create_cluster_col is calendar only (month/half/quarter); Python -pm uses select_cluster_granularity")


if __name__ == "__main__":
    main()
