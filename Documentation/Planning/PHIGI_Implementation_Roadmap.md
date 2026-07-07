# PHIGI Implementation Roadmap

**Physics-Informed Hierarchical Generative Imputation**

**Version:** 1.2  
**Date:** 6 July 2026  
**Status:** In progress — L0/L3 solar shipped; L1 provisional; **L2 partial** (local mixtures + CLUSTER wired for τ and MICE)
**Supersedes (eventually):** `PYMICE_INTEGRATION_PLAN.md` orchestration model; PyMICE remains a subroutine in L3  
**Companion:** `FUTURE_IMPROVEMENTS_ROADMAP.md`, `agent.md`

---

## 1. Executive summary

WEPPCLIFF gap filling will be rebuilt as a **hierarchical generative pipeline** (PHIGI), not an extension of staged `impute_by_cluster` + MICE. The redesign addresses the root failure mode observed in production: **bad solar radiation corrupts the entire daily and sub-daily record** because SO_RAD is imputed first as a generic continuous variable and downstream variables treat it as truth.

**Core architectural shifts:**

| Old (WEPPCLIFF / PyMICE adapter) | New (PHIGI) |
|----------------------------------|-------------|
| Impute raw SO_RAD with MICE | Compute clear-sky \(R^{cs}\); impute cloud transmittance \(\tau\) only |
| Calendar month / hmn / qmn strata | Köppen zone × season × **weather-type mixture** |
| Staged symmetric FCS blocks | Directed generative layers L0–L5 |
| Köppen univariate late fallback | KG joint prior \(\Sigma\) + mixture weights at L1 |
| Row-wise sub-daily MICE | Storm generator with EI/I30 validation |
| Mean fill last resort | Explicit confidence flags; no silent means |

**Phased delivery:** six phases over ~16–24 weeks of focused work (estimates assume one primary developer + periodic review). Each phase ships independently testable value; WEPPCLIFF remains runnable with `-pm f` (legacy mean fill) until Phase 4 gates pass.

---

## 2. Design principles (non-negotiable)

1. **Impute stochastic residuals, not deterministic physics.** SO_RAD, day length, and astronomical geometry are computed, not drawn.
2. **Köppen is the spatial backbone**, not a fallback bucket. Zone identifies priors, seasons, and donor pools.
3. **Directed causality** in daily imputation: \(\tau \rightarrow T \rightarrow\) moisture; regime \(\rightarrow\) event \(\rightarrow\) sub-daily structure.
4. **Validate on erosivity**, not NA count. Hold-out metrics: I30, KE, EI bias, monthly precip mass, T order violations.
5. **Point output for WEPP compatibility** (`m = 1` export), but internal multi-draw search where needed; imputation report documents uncertainty tier.
6. **Legacy path preserved** until Phase 4: `-pm f` and `-gf legacy` (proposed) keep staged MICE available for regression comparison.

---

## 3. Target architecture

```
QC + gap segmentation + missingness diagnosis
        │
        ▼
L0  Deterministic skeleton (R_cs, DOY, lat/lon/elev)
        │
        ▼
L1  Köppen zone + calibrated regime priors (Σ, mixtures)
        │
        ▼
L2  Weather-type mixture fit (local + pooled + prior)
        │
        ▼
L3  Daily posterior (τ, Tmax, Tmin, wind, dew) — PyMICE optional engine
        │
        ▼
L4  Hurdle precipitation (occurrence → daily amount → BPS/ELEN)
        │
        ▼
L5  Storm generator (templates → breakpoints → overlap resolve)
        │
        ▼
Export .cli + ImputationReport (confidence tier, methods used)
```

### Proposed package layout

```
Source/phigi/
├── __init__.py
├── config.py              # PHIGIConfig (flags, thresholds)
├── diagnose.py            # gap segmentation, missingness profiles
├── l0_astronomy.py        # clear-sky radiation, day length
├── l1_koppen.py           # raster lookup, zone resolution
├── l1_priors.py           # load/calibrate Σ, mixture weights
├── l2_weather_types.py    # mixture fit, assignment, pooling
├── l3_daily.py            # daily posterior orchestration
├── l3_tau.py              # transmittance imputation + QC
├── l3_temperature.py      # Tmax/Tmin conditional models
├── l3_moisture_wind.py    # dew point, wind (circular handling)
├── l4_precip_hurdle.py    # occurrence + amount
├── l5_storm_generator.py  # templates, scaling, scheduling
├── l5_ei_validate.py      # I30/KE/EI gates (may wrap existing)
├── report.py              # ImputationReport dataclass + export
└── pipeline.py            # fill_data_phigi() entry

Source/impute.py             # dispatches: legacy | phigi via -gf
priors/                      # versioned KG prior DB (repo root or subdir)
  koppen_priors_v1.json
  storm_templates_v1/        # optional serialized template libraries
```

---

## 4. Phase overview

| Phase | Name | Duration (est.) | Outcome |
|-------|------|---------------|---------|
| **0** | Foundations & harness | 2 weeks | Test infrastructure, baselines, feature flag |
| **1** | L0 SO_RAD decomposition | 2–3 weeks | τ-based radiation; SO_RAD QC; cascade fix |
| **2** | L1 Köppen priors | 3–4 weeks | Raster zone + calibrated joint priors |
| **3** | L2 weather types + L3 daily | 4–5 weeks | Replace calendar clusters; directed daily imputation |
| **4** | L4 hurdle precip | 2–3 weeks | EVENT + daily rain model |
| **5** | L5 storm generator | 3–4 weeks | EI-faithful sub-daily; overlap scheduling |
| **6** | Integration & deprecation | 2 weeks | Default PHIGI; legacy gated; docs |

**Total:** ~16–23 weeks.

---

## 5. Phase 0 — Foundations & harness

**Goal:** Measure current failure modes; enable parallel development without breaking production.

### Deliverables

| ID | Work | Files |
|----|------|-------|
| 0.1 | Add `-gf` flag: `legacy` (default until Phase 6) \| `phigi` | `args.py`, `config.py`, `Manual.md` |
| 0.2 | `ImputationReport` schema (methods, tier, per-variable confidence) | `Source/phigi/report.py` |
| 0.3 | Hold-out validation harness: mask → impute → score | `Development/tests/test_phigi_holdout.py`, `Source/phigi/diagnose.py` |
| 0.4 | Baseline benchmarks: EI/I30 bias, SO_RAD τ residual, T violations on AutoTutorial + 3 stations | `Development/reports/PHIGI_Baseline_*.txt` |
| 0.5 | Scenario fixtures: short gap, long gap, seasonal, variable-absent, SO_RAD-corrupt | `Development/fixtures/phigi/` |

### Verification gate

```bash
.venv/bin/python -m pytest Development/tests/test_phigi_holdout.py -v  # scaffold tests pass
# Baseline report committed for regression comparison
```

### Risks

- None blocking. Pure scaffolding.

---

## 6. Phase 1 — L0: SO_RAD decomposition & physical QC

**Goal:** Fix the observed root failure — bad SO_RAD poisons everything downstream.

### Theory

\[
SO\_RAD_d = R^{cs}_d \cdot \tau_d, \quad \tau_d \in (0, 1]
\]

- \(R^{cs}_d\): Angstrom / Bird clear-sky model from lat, elev, DOY (Langleys/day output to match WEPPCLIFF units).
- \(\tau_d\): sole stochastic daily radiation variable.

### Deliverables

| ID | Work | Notes |
|----|------|-------|
| 1.1 | `l0_astronomy.py`: \(R^{cs}\) per calendar day | Validate vs published tables; unit tests |
| 1.2 | `l3_tau.py`: compute \(\tau = SO\_RAD / R^{cs}\) for observed days; flag outliers | Residual threshold by season |
| 1.3 | SO_RAD QC integration: streak/spike/range on \(\tau\), not raw SO_RAD | Hook into `quality.py` when `-gf phigi` |
| 1.4 | τ imputation: cluster×season mean → Köppen L1 prior fallback | Wired in Phase 2 (L1) |
| 1.5 | Reconstruct SO_RAD from \(\hat\tau \cdot R^{cs}\) before temperature step | Temperature models take \(R^{cs}\), \(\tau\) |

### Verification gate

| Metric | Target |
|--------|--------|
| τ RMSE on hold-out (masked SO_RAD) | < 0.15 mean absolute τ error (tune per climate) |
| SO_RAD RMSE | ≥ 30% better than legacy MICE on same mask |
| T order violations after imputation | No increase vs legacy |
| SO_RAD corruption stress test | Inject 50% bias in summer SO_RAD → τ QC flags ≥ 80% of affected days |

### CLI

| Flag | Purpose |
|------|---------|
| `-gf phigi` | Enable L0/L1 stub path (Phase 1: radiation only) |
| `-tauqc` | τ residual QC threshold override (optional) |

### Dependencies

- Phase 0 harness

### Risks

| Risk | Mitigation |
|------|------------|
| Unit mismatch (W/m² vs Langley) | Golden tests against R `convert_units` SO_RAD path |
| Polar night / extreme lat | Clamp \(R^{cs}\); τ undefined → skip τ QC, use KG prior only |

---

## 7. Phase 2 — L1: Köppen-Geiger prior system (**provisional — under review**)

**Goal:** Replace lat-band heuristics and univariate JSON with a **calibrated joint prior** per zone × season.

> **Architecture review (Jul 2026):** Köppen–Geiger as the spatial backbone is **questioned**. KG provides a climate *label* raster, not τ/T/wind *priors*; the current `koppen_grid_2deg.csv` is rule-generated (not Beck); and L2 **local weather-type mixtures** may identify regimes more faithfully at the station than zone lookup. **L2 does not depend on L1** for its core path (local fit → assign). L1 remains wired as a τ fallback when local donors are absent; spatial pooling (Beck raster, GSHHG offshore mask, ETOPO relief tiers) is deferred pending a decision on whether KG or **station-first mixtures + optional geographic pooling** is the long-term design.

### Deliverables

| ID | Work | Notes |
|----|------|-------|
| 2.1 | `l1_koppen.py`: raster lookup (Beck 2015 or bundled GeoTIFF/CSV grid) | `-kg AUTO` uses raster; manual override kept |
| 2.2 | Prior schema v2: zone × season → `{tau, tmax, tmin, w_vel, w_dir, dp}` means, stds, **correlation matrix** | `priors/koppen_priors_v2.json` |
| 2.3 | `scripts/calibrate_koppen_priors.py`: fit from GHCN-Daily / GSOD / DAYMET subset | Document provenance per zone |
| 2.4 | `l1_priors.py`: sample multivariate normal from Σ; draw τ, T, W, DP jointly | Used when local n ≈ 0 |
| 2.5 | KG-informed seasons: monsoon / dry season splits for Am, Aw, etc. | Extends 4-season lat model |

### Verification gate

| Metric | Target |
|--------|--------|
| Zone lookup | ≥ 95% match manual classification on 50 test stations |
| Prior coverage | All KG zones in CONUS + common global zones have entries |
| Multivariate draw sanity | Simulated panels pass Tmin ≤ Tmax in ≥ 99.5% of draws |
| Marginal calibration | KS distance vs held-out network data < legacy Köppen v1 |

### Dependencies

- Phase 1 (τ variable defined)
- External: observational network extract (one-time download); Beck 2018 KG raster (optional — not integrated)

### Spatial data assets on disk (Jul 2026)

| Asset | Path | Role |
|-------|------|------|
| ETOPO 2022 15s surface | `Data/ETOPO2022/15s_surface/` (288 tiles, ~4.9 GB) | Relief / ruggedness mask candidate |
| GSHHG 2.3.7 shapefiles | `Data/GSHHG/GSHHS_shp/`, `WDBII_shp/` | Land/ocean, coastlines |
| Distance to shore | `Data/GSHHG/dist_to_GSHHG_v2.3.7_1m.nc` | Offshore ≥1° mask (km; + water / − land) |
| Beck KG raster | *not downloaded* | Would supply zone labels only if L1 kept |

### Risks

| Risk | Mitigation |
|------|------------|
| Large prior DB | Ship CONUS-first; lazy download for global |
| Station elevation ≠ grid elev | Lapse-rate adjustment on T priors |

---

## 8. Phase 3 — L2 weather types + L3 daily posterior

**Goal:** Replace calendar clustering and staged MICE with **weather-type mixtures** and **directed daily imputation**.

### L2: Weather-type mixture

| ID | Work | Status |
|----|------|--------|
| 3.1 | Fit 1–4 component Gaussian mixture on \((\tau, T_{max}, T_{min}, W_{vel}, DP)\) per **meteorological season** (BIC selection; min *n* = 30) | **Done** — `l2_weather_types.py`; station-first (no Köppen) |
| 3.2 | Pooling: local → same-KG neighbor stations (optional `-pool`) → L1 prior | **Deferred** — `-pool` not wired |
| 3.3 | Assign each day to MAP component; encode as CLUSTER (`season×10 + component`) for donor matching | **Done** — `assign_weather_types`, `weather_type_cluster_labels` |
| 3.7 | Replace calendar `create_cluster_col` when `-gf phigi` | **Partial** — daily panel, event, and sub-daily paths use `merge_phigi_clusters` when LUT present; legacy calendar path retained for `-gf legacy` |

### L3: Directed daily imputation

| Step | Variables | Model |
|------|-----------|-------|
| 3a | \(\tau\) | Mixture component + persistence AR(1) on \(\tau\) residuals |
| 3b | \(T_{max}, T_{min}\) | Regress on \(R^{cs}\), \(\tau\), DOY, component; PyMICE/JOMO optional for coupled draw |
| 3c | \(W_{vel}, W_{dir}\) | Von Mises + Rician or sin/cos Gaussian; weak coupling to τ |
| 3d | \(DP_{temp}\) | Magnus-type or regress on \(T_{min}, T_{max}\), component |

| ID | Work | Status |
|----|------|--------|
| 3.4 | `l2_weather_types.py` + `l3_daily.py` orchestrator | **Partial** — L2 + `pipeline.py` hooks shipped; `l3_daily.py` not started |
| 3.5 | PyMICE as **directed** engine: custom predictor matrix (not `collect` block) | **Deferred** |
| 3.6 | Constraint loop: Tmin ≤ Tmax (retain R logic, fewer iterations expected) | Legacy loop retained |
| 3.8 | Wire `resilient_impute` concepts into L3 fallback order | **Deferred** |

### Verification gate

```bash
.venv/bin/python -m pytest Development/tests/test_phigi_holdout.py -v
# All scenarios: short_gaps, long_gap_all, persistent_seasonal, variable_absent, mixed
```

| Metric | Target vs legacy `-pm t` |
|--------|---------------------------|
| Daily RMSE (T, τ) | ≥ 15% improvement on hold-out |
| Dry-season false wet days | ≤ legacy rate |
| Imputation report | 100% of runs document tier + methods |

### Dependencies

- Phase 1 (τ defined) — **required**
- Phase 2 (L1 Köppen) — **optional**; pooling fallback (3.2) deferred; local mixture path does not need KG
- PyMICE from GitHub (`ryanpmcg/PyMICE`)

### Risks

| Risk | Mitigation |
|------|------------|
| Mixture identifiability with small n | Minimum n for fitting; fall back to L1 only |
| Wind circular stats | Start with sin/cos; defer von Mises to Phase 6 polish |

---

## 9. Phase 4 — L4: Hurdle precipitation

**Goal:** Replace `cart` EVENT imputation and decoupled PRECIP/BPS/ELEN fills.

### Deliverables

| ID | Work |
|----|------|
| 4.1 | `l4_precip_hurdle.py`: logistic occurrence model P(wet \| KG, season, τ, T, persistence) |
| 4.2 | Integrate `find_missing_precip` dry-month logic as **likelihood constraint**, not post-hoc patch |
| 4.3 | Wet-day amount: gamma / lognormal mixture by zone × season |
| 4.4 | BPS / ELEN: derive from L5 storm generator when possible; else regress on amount + component |
| 4.5 | Dry-day enforcement: PRECIP = 0, BPS = 0, no breakpoint expansion |

### Verification gate

| Metric | Target |
|--------|--------|
| EVENT Brier score | ≥ 10% better than legacy on hold-out |
| Monthly precip mass balance | Mean bias < 5% on hold-out months |
| Dry cluster false events | ≤ legacy `find_missing_precip` corrected rate |

### Dependencies

- Phase 3 (daily τ, T available)

---

## 10. Phase 5 — L5: Storm generator & erosivity validation

**Goal:** Sub-daily imputation that preserves **I30 / KE / EI** fidelity.

### Deliverables

| ID | Work |
|----|------|
| 5.1 | Template library: extract from observed wet days per weather-type component |
| 5.2 | KG-pooled templates when local donors < 5 |
| 5.3 | Parametric fallback: 1–2 pulse model calibrated per zone (convective vs stratiform) |
| 5.4 | Scale template depths to daily PRECIP; preserve timing fractions |
| 5.5 | Interval scheduling overlap resolver (extend `overlap.py`) |
| 5.6 | EI/I30/KE hold-out gate: resample up to K candidates; reject implausible |
| 5.7 | Replace `impute_subdaily_timeseries` row-FCS when `-gf phigi` |

### Verification gate

| Metric | Target |
|--------|--------|
| I30 hold-out bias | < 10% (legacy ~10%+ erosion loss documented) |
| EI hold-out bias | < 5% mean relative error |
| KE bias | < 8% |
| AutoTutorial gap-fill CLI | Structural validation 0 errors |

### Dependencies

- Phases 3–4
- Existing `erosion.py`, `storm_template.py` (refactor into L5)

---

## 11. Phase 6 — Integration, default switch, deprecation

**Goal:** PHIGI becomes default; legacy path archived.

### Deliverables

| ID | Work |
|----|------|
| 6.1 | `fill_data_phigi()` replaces `fill_data()` when `-gf phigi` (default) |
| 6.2 | `-gf legacy` runs old staged path for one release cycle |
| 6.3 | `ImputationReport` written to Export when `-verb t` or `-ir t` (new flag) |
| 6.4 | Update `Manual.md`, `agent.md`, `PYMICE_INTEGRATION_PLAN.md` (mark superseded) |
| 6.5 | Remove mean-fill as silent fallback in PHIGI path; emit `confidence=low` + fill tier |
| 6.6 | Tutorial + Autotest: add PHIGI smoke; gap-fill autotest uses `-gf phigi` |

### Verification gate

```bash
.venv/bin/python -m pytest Development/tests/ -v
bash Tutorial/AutoTutorial.sh
bash Autotest/AutoTest.sh   # non-gap tests unchanged
.venv/bin/python Development/scripts/run_phigi_validation.py --full
```

### Default change

| Release | `-gf` default |
|---------|---------------|
| v2.1.0 | `legacy` |
| v2.2.0 | `phigi` |

---

## 12. CLI flag plan (cumulative)

| Flag | Default | Purpose |
|------|---------|---------|
| `-gf` | `legacy` → `phigi` | Gap-fill engine: `legacy` \| `phigi` |
| `-kg` | `AUTO` | Köppen zone; raster lookup when PHIGI |
| `-pool` | `f` | Pool neighbor stations in same KG zone (future multi-station) |
| `-tauqc` | auto | τ residual QC threshold |
| `-ir` | `f` | Write imputation report JSON to Export |
| `-pm` | `t` | PyMICE engine within L3 (when applicable) |
| `-jm` | `f` | JOMO block for coupled daily draw |
| `-st` | `t` | Storm-template (L5); ignored when `-gf legacy` uses old `-st` |

Existing flags `-id`, `-io`, `-qi`, `-iv`, `-ns` retain meaning within their respective engines.

---

## 13. Testing strategy

### Per-phase tests

| Layer | Test type |
|-------|-----------|
| L0 | Unit: \(R^{cs}\) vs reference; τ bounds |
| L1 | Unit: zone lookup; MVN draw constraints |
| L2 | Unit: mixture fit/assign; integration: pooled prior |
| L3 | Hold-out RMSE per scenario; T order; SO_RAD corrupt stress |
| L4 | Brier; monthly mass; dry-season |
| L5 | I30/EI/KE hold-out; CLI structure |
| End-to-end | AutoTutorial gap lines; West-Illinois re-run comparison |

### Regression policy

- Legacy `-gf legacy` outputs frozen at Phase 0 baseline tags for comparison.
- PHIGI does not target byte-identical `.cli` with R; targets **structural validity + erosivity metrics**.

### Required scripts

```
Development/scripts/run_phigi_validation.py    # full hold-out suite
Development/scripts/calibrate_koppen_priors.py # prior calibration (Phase 2)
Development/scripts/compare_phigi_legacy.py      # side-by-side report
```

---

## 14. Migration map (old → new)

| Legacy module | Disposition |
|---------------|-------------|
| `Source/impute.py` `fill_data` | Becomes dispatcher; legacy body → `fill_data_legacy()` |
| `Source/weppcliff_impute/adapter.py` | L3 engine only (directed FCS); not orchestrator |
| `Source/weppcliff_impute/resilience.py` | Concepts absorbed into L2/L3 fallback |
| `Source/weppcliff_impute/cluster.py` | Calendar mode for `-gf legacy`; PHIGI uses `l2_weather_types.py` |
| `Source/weppcliff_impute/priors/` | Migrate to `priors/` + `l1_priors.py` |
| `Source/weppcliff_impute/storm_template.py` | Absorbed into `l5_storm_generator.py` |
| `Source/weppcliff_impute/validation.py` | Extend for PHIGI scenarios |

---

## 15. Success criteria (program level)

| Criterion | Measurement |
|-----------|-------------|
| SO_RAD cascade fixed | Corrupt-SO_RAD stress test does not degrade T RMSE > 2× τ-only path |
| EI fidelity | Hold-out EI bias < 5% (vs ~10% legacy erosion loss) |
| I30 fidelity | Hold-out I30 bias < 10% |
| Sparse record robustness | 30-year / 40% missing record completes with `confidence` tier documented |
| No silent mean fill | PHIGI path zero mean-fill without `confidence=low` flag |
| WEPP compatibility | `.cli` passes `validate_cli_file` rules 1–5 |
| User transparency | Imputation report lists layers used per gap segment |

---

## 16. Open research items (defer past Phase 6)

| Item | Notes |
|------|-------|
| Hierarchical Bayesian station-in-zone | Full HB via PyMICE 8.x or Stan |
| Multi-station spatial borrow (`-pool t`) | Requires station network index |
| Zero-inflated hurdle with tail copulas | Precip extremes |
| ERA5-Land assimilation for τ on cloudy days | External data license |
| Nonstationary climate trend term | Decade blocks in L2 mixtures |

---

## 17. Recommended start order

1. **Phase 0** (1 week) — flag + harness + baseline report  
2. **Phase 1** (2–3 weeks) — proves SO_RAD thesis; highest user-visible win  
3. **Phase 2** (parallel start after 1.1) — priors calibration can run while τ QC is built  
4. **Phase 3** — critical path  
5. **Phases 4–5** — sequential  
6. **Phase 6** — release  

**First PR suggestion:** Phase 0 + Phase 1.1–1.3 (clear-sky + τ QC only, no full imputation switch).

**Status (6 Jul 2026, v1.2):**

**Shipped**
- `-gf legacy|phigi`, `-tauqc`, `-ir` flags
- **L0** clear-sky + **L3 τ** solar stage in `fill_data` when `-gf phigi`
- **L1 (provisional):** rule-based `koppen_grid_2deg.csv`, `koppen_priors_v2.json`, τ Köppen fallback — **under architecture review**
- **L2 (partial):** `Source/phigi/l2_weather_types.py` + `pipeline.py` hooks
  - Local 1–4 component GMM per meteorological season on `(τ, Tmin, Tmax, W_vel, DP)`; BIC model selection; min *n* = 30
  - **Pipeline order:** `phigi_fit_weather_types` → `phigi_wire_weather_type_clusters(context="solar")` → `phigi_impute_solar_stage` (τ uses weather-type CLUSTER donors) → `phigi_refresh_weather_type_labels` → legacy PyMICE on temp/wind/precip using weather-type CLUSTER
  - `config.impute_granularity = "weather_type"`; `phigi_cluster_lut` merged into event and sub-daily frames via `merge_phigi_clusters` (bypasses calendar `create_cluster_col` on PHIGI path)
  - Report strategies: `local_weather_mixture`, `weather_type_tau_donors`, `weather_type_cluster`, plus existing `tau_qc_invalidate` / `koppen_tau_impute`
- Spatial data: ETOPO 2022 15s surface, GSHHG shapefiles + `dist_to_GSHHG` grid (see §7)
- `Development/scripts/download_etopo2022.py`, `calibrate_koppen_priors.py` (scaffold)
- KMQE (`ASOS_BPT_KMQE.csv`): PHIGI fixes 195-day SO_RAD gap-fill (~400 Langley/day vs legacy zeros); **16 weather-type strata** (4 seasons × 4 BIC-selected components); precip sub-daily unchanged vs legacy
- **28** unit tests in `Development/tests/test_phigi_*.py` (incl. L2 wire + τ cluster donors)

**Deferred / open**
- L1: Beck raster import, tiered resolution grid (2° ocean / 0.1° land / 0.01° rugged), prior calibration — **pending KG vs station-first decision**
- L2: spatial pooling (`-pool`); BIC cap / min-*n* tuning (4 components all seasons on KMQE may overfit); full hold-out gates vs legacy
- L3: `l3_daily.py` directed daily posterior (τ → T → moisture); replace staged MICE blocks (3.5–3.8)
- L4–L5: hurdle precip, storm generator
- Gap-fill string-dtype bug on `-id t` CLI path

---

## 18. Document history

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 6 Jul 2026 | Initial roadmap from PHIGI design review |
| 1.1 | 6 Jul 2026 | L1 provisional; ETOPO/GSHHG assets; L2 local mixtures started; KG route under review |
| 1.2 | 6 Jul 2026 | L2 partial ship: CLUSTER wired for solar τ + MICE; event/sub-daily `merge_phigi_clusters`; 28 tests; KMQE 16 strata |