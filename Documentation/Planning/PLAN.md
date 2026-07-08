# WEPPCLIFF R → Python Conversion Plan

**Version:** 1.0  
**Date:** 25 June 2026  
**Source:** `/Users/home/Software/Antigravity/WEPPCLIFF/WEPPCLIFF.R` (v1.7.1, ~2,891 lines, 89 functions)  
**Target:** `/Users/home/Software/Grok/WEPPCLIFF/`

## 1. Objective

Convert WEPPCLIFF from a monolithic R script into a modular Python package that:

- Preserves **all CLI flags, defaults, and behaviors** documented in `Documentation/Manual.md`
- Provides **one Python function per R function** (may split complex functions into helpers)
- Supports **installation**, **automated tutorial**, and **automated testing** like the original
- Uses **concurrent processing** where the R version used `foreach`/`doParallel`
- Renames **ALL-CAPS directories** to standard capitalization
- Accepts **minor numeric/rounding differences** vs R benchmarks

## 2. Directory Mapping

| R (Antigravity) | Python (Grok) | Notes |
|-----------------|---------------|-------|
| `INPUT/` | `Input/` | Sample CSV inputs |
| `OUTPUT/` | `Output/` | `.cli` files |
| `EXPORT/` | `Export/` | `.rds` → `.pkl` or `.json`; CSV exports unchanged |
| `PLOTS/` | `Plots/` | PNG plots via matplotlib |
| `LIBRARY/` | *(removed)* | Python uses `pip` / `venv` instead of local R library |
| `INSTALLER/` | `Installer/` | `Install.sh` / `Install.bat` |
| `AUTOTEST/` | `Autotest/` | Benchmark comparison scripts |
| `TUTORIAL/` | `Tutorial/` | `AutoTutorial.sh` / `.bat` |
| `ARCHIVE/` | `Archive/` | Historical R versions (reference only) |
| `WEPPCLIFF.R` | `weppcliff.py` + `Source/` | Entry point at repo root |

## 3. Python Package Architecture

```
WEPPCLIFF/
├── weppcliff.py              # CLI entry point (replaces Rscript WEPPCLIFF.R)
├── pyproject.toml            # Package metadata + dependencies
├── requirements.txt          # Pinned deps for Installer
├── Source/
│   ├── __init__.py
│   ├── config.py             # WeppcliffConfig dataclass (replaces R globals)
│   ├── branding.py           # print_weppcliff_logo, license functions
│   ├── paths.py              # find_this_file, set_wds, create_directory
│   ├── args.py               # parse/validate/assign CLI arguments
│   ├── io.py                 # load_input_file, load_input_directory, export
│   ├── preprocess.py         # preprocess_file, store_data, convert_data
│   ├── utils.py              # vec/df find_replace, convert_df_type
│   ├── duration.py           # create_duration_data, modify_midnight_breakpoints
│   ├── units.py              # convert_units
│   ├── quality.py            # QC pipeline (12 functions)
│   ├── storms.py             # break_storms, identify_storms, smart_window_sum
│   ├── time_bounds.py        # time_bounds, start/end_of_year/day
│   ├── erosion.py            # calculate_erosion_indices
│   ├── daily.py              # process_daily_data, process_alt_data, dew_point
│   ├── precip.py             # process_precip_data, aggregate_precip_data
│   ├── impute.py             # fill_data + cluster imputation (MICE placeholder)
│   ├── plots.py              # graph_data and all plot helpers
│   ├── cli_output.py         # generate_header/export/create_cli_file
│   ├── workflow.py           # core_workflow, fill_workflow, parallel/series
│   └── main.py               # main(), orchestration
├── Documentation/
│   ├── Manual.md

│   └── Planning/             # PLAN.md, agent.md, design notes
├── Development/
│   ├── tests/                # pytest suite
│   ├── scripts/              # analysis / validation helpers
│   └── Commands/             # macOS .command launchers
├── Tutorial/                 # AutoTutorial.sh, Tutorial.md
├── Input/  Output/  Export/  Plots/  Installer/  Autotest/
└── weppcliff.py
```

### 3.1 Design Principles

1. **`WeppcliffConfig`** — Single config object passed through the pipeline instead of R's `.GlobalEnv` mutations (`<<-`). Thread/process workers receive a serialized copy.
2. **Pandas DataFrames** — Replace R `data.frame`; the `data` list structure (indices 1–22+) maps to a `WeppcliffData` class with named slots.
3. **NumPy vectorization** — Replace R loops in `smart_window_sum`, QC tests, and aggregation where safe.
4. **`concurrent.futures.ProcessPoolExecutor`** — Replace `foreach`/`doParallel` for multi-station runs.
5. **Explicit typing** — Use type hints on all public functions for agent maintainability.

## 4. R → Python Dependency Mapping

| R Package | Python Equivalent | Usage |
|-----------|-------------------|-------|
| `readr` / `vroom` | `pandas` | CSV/TSV/TXT loading |
| `data.table` | `pandas` | `rbindlist`, fast merges |
| `ggplot2` | `matplotlib` | Time-series plots |
| `mice` | **PyMICE** (`pymice.integrations.weppcliff`) | Gap filling via `-pm t` |
| `EnvStats` | `scipy.stats` | Return-period / distribution (future) |
| `foreach` / `doParallel` | `concurrent.futures` | Parallel station processing |
| `jsonlite` | `json` (stdlib) | JSON export if enabled |
| `profvis` / `Rprof` | `cProfile` (optional) | Developer profiling (`-prof`) |
| `tzdb` | `zoneinfo` (stdlib) | Timezone (`-tz`) |

## 5. MICE / Gap-Filling Strategy

**Phase 6a–6d (complete):** PyMICE FCS via `pymice.integrations.weppcliff` when `-pm t` (default).

- `impute_by_cluster` / `impute_by_df` → PyMICE `mice()` with `quickpred`, per-cluster FCS or unified `2l.norm`
- **6b:** Köppen–Geiger priors, nonstationary (`-ns`), resilience fallback chain, diagnostics
- **6c:** Storm-template sub-daily (`-st`), overlap resolution, I30/EI validation gates
- **6d:** Synthetic weather validation harness, AutoTutorial gap-fill smoke tests
- `-pm f` → legacy cluster/global mean fill
- Cluster tiers: **seasonal (4)**, month, half-month, quarter-month (covariance-based selection; no global cluster)
- Planning docs: `PYMICE_INTEGRATION_PLAN.md`, `FUTURE_IMPROVEMENTS_ROADMAP.md`

**Phase 7a (PyMICE):** Full `jomoImpute` / `panImpute` in PyMICE core. WEPPCLIFF daily multivariate block via **`-jm t`** (optional; default `f` keeps scatter FCS).

Functions affected: `fill_data`, `impute_by_cluster`, `impute_by_df`, `find_missing_precip`, `create_cluster_col`, `fill_workflow`

## 6. Suspected CLI Output Bug

User reports a possible bug in CLI generation. Investigation targets:

| Area | R Functions | Suspected Issue |
|------|-------------|-----------------|
| Breakpoint counting | `generate_export_data` | `daily_bps_adj` midnight adjustments; `updated_bps` off-by-one when `clock_resets < 0` |
| Breakpoint writing | `create_cli_file` | `last.pcp = current.pcp + daily_df$BPS[i] - 2` indexing; `new.day` flag logic for non-precip breaks |
| Hour=24 convention | `create_cli_file` | Midnight rows use `HOUR == "24"` but `generate_export_data` uses `pcp_times == 0 → 86400` |
| Single-storm mode | `create_cli_file` | `sid > 1 \| sm == 2` condition uses bitwise `\|` (works in R but easy to mistranslate) |
| Missing-day guard | `create_cli_file` | `nrow(daily_df) < ceiling(daysCount)` may false-positive after imputation |

**Validation approach:**

1. Run R `AutoTest.sh` → capture 8 benchmark `.cli` MD5 sums
2. Run Python autotest with **tight** comparison on tests 1–8
3. Binary-diff any failing `.cli` files line-by-line
4. Fix Python; if R output is internally inconsistent, document and match R behavior (compatibility over correctness)

## 7. Complete Function Mapping (89 functions)

### 7.1 Branding & Setup (11)

| R Function | Python Module | Python Function |
|------------|---------------|-----------------|
| `print_weppcliff_logo` | `branding.py` | `print_weppcliff_logo()` |
| `print_license_agreement` | `branding.py` | `print_license_agreement()` |
| `prompt_license_agreement` | `branding.py` | `prompt_license_agreement()` |
| `create_directory` | `paths.py` | `create_directory()` |
| `create_base_directory` | `paths.py` | `create_base_directory()` |
| `find_this_file` | `paths.py` | `find_this_file()` |
| `set_wds` | `paths.py` | `set_wds()` |
| `install_dependencies` | `main.py` | `install_dependencies()` |
| `load_packages` | `main.py` | `load_packages()` *(no-op in Python)* |
| `parse_arg_locations` | `args.py` | `parse_arg_locations()` |
| `early_license_agreement` | `args.py` | `early_license_agreement()` |

### 7.2 Arguments (4)

| R Function | Python Module |
|------------|---------------|
| `assign_args_to_vars` | `args.py` → `assign_args_to_vars()` |
| `assign_empty_args` | `args.py` → `assign_empty_args()` |
| `check_args` | `args.py` → `check_args()` |
| `assign_final_args` | `args.py` → `assign_final_args()` |

### 7.3 I/O & Preprocessing (10)

| R Function | Python Module |
|------------|---------------|
| `load_input_file` | `io.py` |
| `check_input_format` | `io.py` |
| `load_input_directory` | `io.py` |
| `preprocess_file` | `preprocess.py` |
| `store_data` | `preprocess.py` |
| `convert_data` | `preprocess.py` |
| `remove_missing_datetime_rows` | `preprocess.py` |
| `preserve_missing_data` | `preprocess.py` |
| `convert_df_type` | `utils.py` |
| `vec_find_replace` / `df_find_replace` | `utils.py` |

### 7.4 Duration & Aggregation (3)

| R Function | Python Module |
|------------|---------------|
| `create_duration_data` | `duration.py` |
| `modify_midnight_breakpoints` | `duration.py` |
| `aggregate_precip_data` | `precip.py` |

### 7.5 Units & QC (14)

| R Function | Python Module |
|------------|---------------|
| `convert_units` | `units.py` |
| `initialize_qc_data` | `quality.py` |
| `quality_check_inputs` | `quality.py` |
| `create_rate_qc_data` | `quality.py` |
| `run_nonprecip_checks` | `quality.py` |
| `extreme_limit_test` | `quality.py` |
| `rle_streak_test` | `quality.py` |
| `snr_spike_test` | `quality.py` |
| `rle_stick_test` | `quality.py` |
| `run_precip_checks` | `quality.py` |
| `return_period_test` | `quality.py` |
| `remove_qc_fails` | `quality.py` |
| `quality_check_outputs` | `quality.py` |
| `impose_daily_value_limits` / `extreme_limit_replace` | `quality.py` |

### 7.6 Storms & Time (11)

| R Function | Python Module |
|------------|---------------|
| `trim_data` | `storms.py` |
| `break_storms` | `storms.py` |
| `cum_to_inc` | `storms.py` |
| `identify_storms` | `storms.py` |
| `evaluate_connectivity` | `storms.py` |
| `int_30m` | `storms.py` |
| `smart_window_sum` | `storms.py` |
| `time_bounds` | `time_bounds.py` |
| `start_of_year` / `end_of_year` | `time_bounds.py` |
| `start_of_day` / `end_of_day` | `time_bounds.py` |

### 7.7 Erosion, Daily, Precip (9)

| R Function | Python Module |
|------------|---------------|
| `calculate_erosion_indices` | `erosion.py` |
| `process_daily_data` | `daily.py` |
| `process_alt_data` | `daily.py` |
| `dew_point_temperature` | `daily.py` |
| `use_alt_data` | `daily.py` |
| `process_precip_data` | `precip.py` |
| `fill_agg_df` | `precip.py` |
| `annual_summary` | `daily.py` |

### 7.8 Imputation (5)

| R Function | Python Module | Notes |
|------------|---------------|-------|
| `fill_data` | `impute.py` | Mean-fill placeholder |
| `create_cluster_col` | `impute.py` | |
| `impute_by_cluster` | `impute.py` | Mean-fill placeholder |
| `impute_by_df` | `impute.py` | Mean-fill placeholder |
| `find_missing_precip` | `impute.py` | |

### 7.9 Plots (8)

| R Function | Python Module |
|------------|---------------|
| `graph_data` | `plots.py` |
| `create_input_plots` | `plots.py` |
| `annual_plots` | `plots.py` |
| `monthly_plots` | `plots.py` |
| `daily_plots` | `plots.py` |
| `storm_plots` | `plots.py` |
| `create_other_plots` | `plots.py` |

### 7.10 Export & CLI (6)

| R Function | Python Module |
|------------|---------------|
| `export_data` | `io.py` |
| `export_to_csv` | `io.py` |
| `export_pcp_bps` | `io.py` |
| `generate_header_data` | `cli_output.py` |
| `generate_export_data` | `cli_output.py` |
| `create_cli_file` | `cli_output.py` |

### 7.11 Workflow & Execution (9)

| R Function | Python Module |
|------------|---------------|
| `create_cluster` | `workflow.py` |
| `kill_cluster` | `workflow.py` |
| `fill_workflow` | `workflow.py` |
| `core_workflow` | `workflow.py` |
| `parallel_execution` | `workflow.py` |
| `series_execution` | `workflow.py` |
| `main` | `main.py` |
| `view_data` | `utils.py` |
| `stop_quietly` | `utils.py` |

## 8. Data Structure Mapping

R stores everything in a numbered `data` list. Python `WeppcliffData` slots:

| Index | Name | Content |
|-------|------|---------|
| 1–3 | `precip_ts`, `alt_ts`, `daily_ts` | Input time series |
| 4 | `metadata` | dict: STATION, LAT, LON, ELEV, OBS_YRS, B_YR, YRS_SIM |
| 5–7 | `precip_qc`, `alt_qc`, `daily_qc` | QC copies |
| 8 | `storm_list` | list of storm DataFrames |
| 9 | `storm_df` | storm summary DataFrame |
| 10 | `daily_out` | processed daily |
| 11–14 | `monthly_ts`, `annual_ts`, `monthly_mean`, `annual_summary` | aggregates |
| 15+ | post-imputation mirrors | same roles after `fill_workflow` |

## 9. CLI Compatibility

Invocation change:

```bash
# R (original)
Rscript --vanilla WEPPCLIFF.R --args -f file.csv -la y

# Python (target)
python weppcliff.py -f file.csv -la y
# or after install:
weppcliff -f file.csv -la y
```

All 52 flags from `flags` vector in R (lines 2748–2758) must parse identically. Argument printing format in `assign_args_to_vars` should be preserved for user familiarity.

## 10. Testing Strategy

### 10.1 Automated Test (Autotest/)

Port `AutoTest.sh` → `Autotest/AutoTest.sh`:

1. Run 8 WEPPCLIFF commands (same flags as R version)
2. MD5-compare output `.cli` files against `Autotest/Benchmarks/`
3. Optionally compare export artifacts (`.pkl` vs `.rds` with structural equivalence)

**Tolerance:** Exact MD5 for tests 1–8 CLI files is the goal. If mean-imputation prevents exact match on tests using `-id t`, document and compare only non-imputation tests strictly.

### 10.2 Unit Tests (`tests/`)

- Per-function tests with minimal synthetic DataFrames
- Golden-file tests for `generate_export_data` and `create_cli_file`
- Regression test for suspected CLI bug cases (midnight breakpoints, cumulative precip)

### 10.3 Cross-Repository Validation

After each major phase:

```bash
# R baseline
cd /Users/home/Software/Antigravity/WEPPCLIFF && bash AUTOTEST/AutoTest.sh

# Python port
cd /Users/home/Software/Grok/WEPPCLIFF && bash Autotest/AutoTest.sh
```

## 11. Installation

`Installer/Install.sh`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python weppcliff.py -fr t -la y   # first-run equivalent
```

Windows: `Installer/Install.bat` with analogous steps.

## 12. Implementation Phases

### Phase 1 — Scaffold (Day 1)
- [x] Create directory structure with renamed folders
- [x] `pyproject.toml`, `requirements.txt`, `weppcliff.py` entry point
- [x] `WeppcliffConfig`, `args.py` with all flags
- [x] `Installer/`, `Autotest/`, `Tutorial/` shell scripts

### Phase 2 — Data Pipeline (Day 1–2)
- [x] I/O, preprocessing, duration, units
- [x] QC module
- [x] Storm separation + time bounds
- [x] Daily/precip/erosion processing

### Phase 3 — Output (Day 2–3)
- [x] CLI generation (`cli_output.py`, `cli_validate.py`, `cli_input.py`)
- [x] Export (CSV, pickle, weather summaries via `export_weather.py`)
- [x] Plots

### Phase 4 — Imputation & Workflow (Day 3)
- [x] PyMICE FCS via `Source/weppcliff_impute/` (`-pm t`); mean-fill fallback (`-pm f`)
- [x] `core_workflow`, `fill_workflow`
- [x] Parallel execution

### Phase 5 — Validation (Day 3–4)
- [x] Port autotest; AutoTest 2–8 byte-identical to R; AutoTest.1 structurally valid (MD5 parity deferred)
- [x] Unit test coverage (101 tests in `Development/tests/`)
- [x] Gap-fill CLI-input workflow (`YYYYMMDD` float dtype normalized via `Source/phigi/dates.py`)

## 13. Concurrency Plan

| R Behavior | Python Implementation |
|------------|----------------------|
| `makeCluster(min(s, max.cores))` | `ProcessPoolExecutor(max_workers=min(stations, cores))` |
| `foreach %dopar%` | `executor.map(core_workflow, files)` |
| Export `.export.list` to workers | Pass `WeppcliffConfig` + file path per task |
| `registerDoParallel` | Context manager around executor |
| Single station | Sequential path (no pool overhead) |

Use `os.cpu_count()` with same 90% / minus-4 logic as R `create_cluster`.

## 14. Export Format Change

| R | Python | Notes |
|---|--------|-------|
| `.rds` | `.pkl` (pickle) or `.json` | Autotest RDS comparisons become structural checks |
| `.csv` | `.csv` | Same names, same columns |
| `.cli` | `.cli` | Must remain byte-identical for autotest |

## 15. Resolved Decisions

1. **Export binary:** `.pkl` (pickle) replaces `.rds` everywhere for efficient read/write.
2. **Autotest:** Exclude gap-filled (`-id t`) comparisons; compare `.cli` MD5 to current R output when available.
3. **Python version:** `>=3.10` with standard `venv` (cross-platform).
4. **LIBRARY folder:** Removed; dependencies via `Installer/Install.sh` + `pip`.
5. **PyMICE dependency (6 July 2026):** Install from GitHub (`pymice[ml] @ git+https://github.com/ryanpmcg/PyMICE.git`), not PyPI — an unrelated package also named `pymice` exists on PyPI. Do not use a local editable path in production install files.

## 16. Success Criteria

- [x] `python weppcliff.py` runs with all CLI flags
- [x] `bash Installer/Install.sh` succeeds on macOS (with network for PyMICE)
- [x] `bash Tutorial/AutoTutorial.sh` completes without error
- [x] `bash Autotest/AutoTest.sh` passes AutoTest 2–8; AutoTest.1 MD5 parity intentionally deferred
- [x] All 89 R functions have Python equivalents callable from tests
- [x] `agent.md` enables a fresh agent to continue work without re-reading R source