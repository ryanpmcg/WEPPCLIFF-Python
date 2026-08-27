# Ampute–impute experiment matrix

**Status:** Implemented 2026-07-31  
**Code:** `Source/weppcliff_impute/experiment_matrix/`  
**CLI:** `Development/scripts/run_experiment_matrix.py`  
**Tests:** `Development/tests/test_experiment_matrix.py`  
**Outputs:** `Autotest/Output/ExperimentMatrix/`

This document defines the assessment matrix that completes the 2026-07 multiseed
study: dual **whole vs hole** climate statistics, powered scenarios, multi-engine
ablations, storm/breakpoint metrics, and optional full-workflow PHIGI compare.

---

## 1. Motivation (findings that drove the design)

| Issue | Legacy multiseed (N=30, midastouch only) | Matrix response |
| --- | --- | --- |
| Monthly mass bias looked great under short/long gaps | Dilution by intact months | Dual **scope=whole** and **scope=hole** |
| Short gaps under-powered (~40 days) | Noisy KS / mass | **B:** powered short gaps (~10% of days) |
| Long gap fixed at rows 120–300 | One season only | **C:** DJF/MAM/JJA/SON rotation |
| Storm EMAX/EMIN explode on long gap | Not in paper table | Storm metrics in multiseed + gates |
| SO_RAD sensor-absent fails hard | Single-seed only | **D:** variable-absent multiseed |
| Breakpoint ANT/BTW bad; `-st` untested | Single-seed MCAR | **E:** breakpoint engines |
| PHIGI not in paper multiseed | Feature table only | **F:** full `fill_workflow` fair compare |
| Clusters labeled on complete data | Optimistic | **midastouch_post_cluster** engine |
| MNAR / method ablation missing | MCAR only | **G:** MNAR cold + JOMO |

---

## 2. Dual-scope scoring

Every mass/mean climate statistic is emitted with:

| Field | Values | Meaning |
| --- | --- | --- |
| `scope` | `whole` | Reconstructed **full series** vs truth |
| `scope` | `hole` | Only calendar days with any introduced NA |
| `scope` | `holdout_cells` | Amputed cells with finite truth+impute (occurrence, KS, RMSE) |
| `period` | `daily` | Sum/RMSE over selected days |
| `period` | `monthly` | Per-YYYYMM aggregates then mean rel. bias or total ratio |
| `period` | `seasonal` | DJF/MAM/JJA/SON mass or mean ratios |
| `period` | `annual` | Per-year aggregates |
| `period` | `sample` | Distributional sample (KS, sector TVD) |

**Interpretation rule**

- Use **hole** metrics to judge reconstruction skill.
- Use **whole** metrics to judge whether the export still looks like climate for WEPP/USLE.
- Never treat whole monthly bias alone as proof that the hole was filled well.

Primary code: `experiment_matrix/scoring.py` → `score_dual_panel()`.

---

## 3. Experiment catalog (A–G)

| ID | Title | Scenarios | Engines | Panel | Seeds (smoke / default / paper) |
| --- | --- | --- | --- | --- | --- |
| **A** | MCAR baseline | `mcar_ampute` | midastouch, mean, cart | daily | 1 / 10 / 30 |
| **B** | Powered short gaps | `short_gaps_powered` | midastouch, midastouch_post_cluster, mean | daily | 1 / 10 / 30 |
| **C** | Long gap × seasons | `long_gap_{DJF,MAM,JJA,SON}` | midastouch, midastouch_post_cluster, mean | daily | 1 / 5 / 15 |
| **D** | Variable-absent | `variable_absent:SO_RAD`, `…:MIN_TEMP` | midastouch, mean, midastouch_post_cluster | daily | 1 / 5 / 10 |
| **E** | Breakpoint MCAR | `breakpoint_mcar` | midastouch, storm_template_fcs, mean | breakpoint | 1 / 5 / 15 |
| **F** | Full workflow | fair_compare holdouts | workflow_legacy, workflow_phigi | workflow | 0 / 2 / 5 |
| **G** | MNAR + JOMO | `mnar_cold_sensor`, `mcar_ampute` | midastouch, jomo, mean | daily | 1 / 5 / 10 |

Catalog source of truth: `experiment_matrix/catalog.py`.

---

## 4. Engines

| Name | Behavior |
| --- | --- |
| `mean` | Cluster mean fill (`pm=f`) |
| `midastouch` | PyMICE midastouch; **pre-ampute** clusters (paper baseline) |
| `cart` | PyMICE cart ablation |
| `midastouch_post_cluster` | midastouch; clusters assigned **after** amputation |
| `jomo` | Daily JOMO/PAN block (`jm=t`) |
| `storm_template_fcs` | Sub-daily path preferring storm templates (`st=t`) |
| `workflow_legacy` | Full `fill_workflow` `-gf legacy` |
| `workflow_phigi` | Full `fill_workflow` `-gf phigi` |

---

## 5. How to run

```bash
cd /Users/home/Software/Grok/WEPPCLIFF
. .venv/bin/activate

# List catalog
python Development/scripts/run_experiment_matrix.py --list

# CI / developer smoke (fast; mean-capable synthetic tests also in pytest)
python Development/scripts/run_experiment_matrix.py --profile smoke --serial \
  --experiments A,B --n-seeds 1 --io 8

# Recommended local assessment
python Development/scripts/run_experiment_matrix.py --profile default --workers 4

# Paper multiseed (long)
python Development/scripts/run_experiment_matrix.py --profile paper --workers 4

# Full workflow PHIGI vs legacy (slow; needs Input/ASOS_BPT_KMQE.csv)
python Development/scripts/run_experiment_matrix.py --experiments F --include-workflow \
  --profile default --serial
```

### Prerequisites

- **Daily panel:** `Autotest/Output/AutoTest.1-Daily-Time-Series.csv`  
  (from `weppcliff.py … -fn AutoTest.1 -ed 6` or Autotest run)
- **Breakpoint panel (E):** `Autotest/Output/AutoTest.1-Precipitation.csv`
- **Workflow (F):** `Input/ASOS_BPT_KMQE.csv`

### Outputs

| File | Content |
| --- | --- |
| `replicates_long.csv` | Every seed × metric (long form) |
| `aggregate_mean_std.csv` | mean/std/median/p05/p95 by factors |
| `gates.csv` | CI / assessment gate results |
| `REPORT.md` | Human summary with whole vs hole tables |
| `TABLE_experiment_A_dual_scope.md` | Paper-facing A snapshot |

Paper profile also copies aggregates into `JOSS-PAPER/results/matrix_*.csv`.

---

## 6. Gates (automated)

| Gate | Rule |
| --- | --- |
| `A1_mcar_wet_f1` | midastouch MCAR wet F1 ≥ 0.95 |
| `A2_mcar_hole_mass` | hole daily PRECIP mass_ratio ∈ [0.75, 1.25] |
| `C_emax_*` | EMAX mean_ratio &lt; 5 per season (tracks known failure mode) |
| `D1_sorad_absent_runs` | SO_RAD variable-absent RMSE finite |
| `Z_job_success_rate` | ≥ 95% jobs status=ok |

Gates are **assessment instrumentation**, not all hard product requirements. EMAX
gates document residual weakness of joint continuous MICE on storm peaks.

---

## 7. Relationship to older scripts

| Artifact | Role after this work |
| --- | --- |
| `autotest1_ampute_impute_analysis.py` | Single-seed plots / SUMMARY still useful |
| `autotest1_ampute_impute_multiseed.py` | Superseded for new work; keep for paper freeze reproducibility |
| `experiment_matrix/*` + `run_experiment_matrix.py` | **Canonical** multi-experiment assessment |
| `compare_phigi_legacy.py` | Used inside experiment **F** |

---

## 8. Recommended reading order for results

1. `REPORT.md` — whole vs hole contrast tables  
2. Filter `aggregate_mean_std.csv` where `scope=='hole'` and `variable=='PRECIP'`  
3. Storm block: `variable in (EMAX,EMIN,BPS,ELEN)`  
4. Experiment D SO_RAD RMSE (holdout_cells)  
5. Experiment E ANT/BTW mean_ratio and I30 mean_rel_bias  
6. Experiment F workflow I30/EI columns if run  

---

## 9. Implementation map

```
Source/weppcliff_impute/experiment_matrix/
  __init__.py       # public exports
  catalog.py        # ExperimentSpec A–G
  scenarios.py      # powered gaps, seasons, MNAR, breakpoint MCAR
  engines.py        # panel + workflow engines
  scoring.py        # dual whole/hole temporal metrics
  runner.py         # job graph, parallel execute, gates
  report.py         # REPORT.md writers

Development/scripts/run_experiment_matrix.py
Development/tests/test_experiment_matrix.py
Documentation/Planning/AmputeImputeExperimentMatrix.md  # this file
```

---

## 10. Document history

| Date | Change |
| --- | --- |
| 2026-07-31 | Initial matrix implementation: dual-scope scoring, experiments A–G, CLI, tests, gates |
