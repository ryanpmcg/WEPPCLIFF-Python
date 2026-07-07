# PyMICE Integration Plan — WEPPCLIFF Gap Filling

**Version:** 1.0  
**Date:** 26 June 2026  
**Status:** Phase 6d complete; Phase 7a (JOMO) wired via `-jm`
**PyMICE:** [github.com/ryanpmcg/PyMICE](https://github.com/ryanpmcg/PyMICE) (`pymice[ml] @ git+https://github.com/ryanpmcg/PyMICE.git`)  
**WEPPCLIFF:** `/Users/home/Software/Grok/WEPPCLIFF/`

**Future work:** `FUTURE_IMPROVEMENTS_ROADMAP.md`

---

## 1. Problem

WEPPCLIFF gap-filling targets **cluster-correlated multivariate weather**. Relationships shift through the year; they should be learned from observed data (`quickpred` drops weak predictors). The Python port currently uses **cluster/global means** despite `imp_method` arguments.

---

## 2. Statistical design

### 2.1 Cluster granularities (coarse → fine)

| Tier | Clusters/year | Notes |
|------|---------------|-------|
| **Seasonal** | 4 | Lat-aware DJF/MAM/JJA/SON; coarsest allowed |
| **Month** | 12 | Calendar month |
| **Half-month** | 24 | WEPPCLIFF `hmn` |
| **Quarter-month** | 48 | WEPPCLIFF `qmn` |

**Removed:** global (1). **Fallback:** adjacent-season pooling or Köppen prior (6b), not global cluster.

**Selection:** `select_cluster_granularity()` picks the **finest** tier where every cluster meets `n_k ≥ max(30, 5p)` and bootstrap `quickpred` stability ≥ 0.6. High within-cluster noise → coarsen locally.

### 2.2 Daily imputation

- **FCS + `quickpred`** — one daily pass (not staged solar→temp→wind hack)
- **Hybrid:** per-cluster FCS when clusters are well-populated; unified **`2l.norm`** when any cluster is sparse
- **Optional (`-jm t`):** multivariate **`jomoImpute`** / **`panImpute`** block for coupled daily weather vector
- Methods: continuous → `midastouch` (or `-im` override); `EVENT` → `cart` when `-qi t`

### 2.3 Sub-daily (6a vs 6c)

- **6a:** row-wise FCS on `DUR`, `PRECIP`, time components (better than means; EI weakness remains)
- **6c:** storm-template donor imputation for `I30` / sustained intensity fidelity

### 2.4 Constraints

- `MIN_TEMP ≤ MAX_TEMP` via re-impute loop + `post_squeeze`
- `PRECIP ≥ 0`, `DUR ≥ 0`
- Dry-day masks via `where`

---

## 3. Architecture

```
MICE/src/pymice/integrations/weppcliff/
├── __init__.py
├── cluster.py       # seasonal/month/hmn/qmn + select_cluster_granularity
├── setup.py         # DataFrame → VariableSpec, quickpred, maxit
├── adapter.py       # impute_by_cluster, impute_by_df
├── constraints.py   # temperature post hooks
├── missingness.py   # NA pattern classifier
├── resilience.py    # adaptive fallback chain
├── nonstationary.py # year-block predictors (-ns)
├── diagnostics.py   # verbose imputation report
├── validation.py    # ampute scenarios + hold-out metrics (6d)
└── priors/          # Köppen JSON store + blending

WEPPCLIFF/Source/impute.py
  -pm t  → adapter (default)
  -pm f  → mean-fill (legacy)
```

---

## 4. Phases

### Phase 0 — Documentation ✅

- Sync `MICE/Documentation/IMPLEMENTATION_PLAN.md` with `agent.md`
- This plan + `FUTURE_IMPROVEMENTS_ROADMAP.md`
- Update `WEPPCLIFF/Documentation/Planning/agent.md` and `PLAN.md` §5

### Phase 6a — Daily FCS adapter ✅

- [x] `pymice.integrations.weppcliff` module
- [x] Covariance-based `create_cluster_col` (seasonal floor)
- [x] `-pm` feature flag; mean-fill fallback
- [x] Unit tests + smoke on synthetic weather panel

### Phase 6b — Resilience + priors ✅

- [x] Köppen prior DB (`priors/default_priors.json`) + blending
- [x] `scripts/calibrate_koppen_priors.py` for global data calibration
- [x] Missingness classifier + resilience fallback chain
- [x] Nonstationary (`-ns`) year-block predictors
- [x] `-kg` Köppen zone override; lat/lon from station metadata
- [x] Imputation diagnostics (`-verb t` / `-iv t`)

### Phase 6c — Sub-daily / EI ✅

- [x] Storm-template donor imputation (`storm_template.py`, `subdaily.py`)
- [x] Interval overlap resolution (`overlap.py`) replaces duplicate-merge loop
- [x] I30 proxy validation gates (`ei_validate.py`); warnings when `-verb t`
- [x] `-st` flag (default `t`) toggles storm-template vs row-wise FCS

### Phase 6d — Validation ✅

- [x] `validation.py` — synthetic weather panel, scenario masks, hold-out RMSE/MAE
- [x] `ampute` MCAR scenario + short/long/seasonal/variable-absent/mixed patterns
- [x] `tests/unit/test_weppcliff_6d.py`
- [x] `Development/tests/test_gapfill_autotutorial.py` (AutoTutorial.sh lines 26–27)
- [x] `Development/scripts/run_gapfill_validation.py`

---

## 5. CLI

| Flag | Default | Purpose |
|------|---------|---------|
| `-pm` | `t` | Use PyMICE FCS (`f` = mean-fill) |
| `-im` | `DEFAULT` | Method override (`midastouch`, `pmm`, …) |
| `-io` | `100` | Max MICE iterations cap |
| `-qi` | `f` | Quick impute → `cart`, 10 iterations |
| `-iv` | `f` | Print MICE progress |
| `-ns` | `f` | Year-block nonstationary predictors |
| `-kg` | `AUTO` | Köppen zone for regional priors |
| `-st` | `t` | Storm-template sub-daily imputation |
| `-jm` | `f` | Daily multivariate JOMO/PAN block (`t` = `jomoImpute`/`panImpute`) |

---

## 6. PyMICE readiness

**Shipped (6a–6d + 7a):** `mice`, `quickpred`, `midastouch`, `cart` `[ml]`, `2l.norm`, `2l.pan`, `jomoImpute`, `panImpute`, `post`/`squeeze`, `blocks`, Köppen priors, `ampute` validation harness.

**Not ported:** `mice.mids`, `futuremice`, `rf`, `jomo2*` level-2 target models, Fortran bit-parity.