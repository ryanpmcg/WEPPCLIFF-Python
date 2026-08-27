# Agent Guide — WEPPCLIFF Python Port

This document directs AI agents working on the R → Python conversion of WEPPCLIFF.

## Quick Reference

| Item | Path |
|------|------|
| **Python target** | `/Users/home/Software/Grok/WEPPCLIFF/` |
| **R source (read-only reference)** | `/Users/home/Software/Antigravity/WEPPCLIFF/WEPPCLIFF.R` |
| **Implementation plan** | `Documentation/Planning/PLAN.md` |
| **Gap-fill redesign (PHIGI)** | `Documentation/Planning/PHIGI_Implementation_Roadmap.md` |
| **Ampute–impute experiment matrix (A–G)** | `Documentation/Planning/AmputeImputeExperimentMatrix.md` |
| **PHIGI modules** | `Source/phigi/` (L0–L6 pipeline; opt-in via `-gf phigi`) |
| **PHIGI validation** | `run_phigi_validation.py`, `compare_phigi_legacy.py` (fair KMQE hold-out) |
| **CLI manual** | `Documentation/Manual.md` |
| **Entry point** | `weppcliff.py` |
| **Source modules** | `Source/` |

## Before You Change Anything

1. Read `Documentation/Planning/PLAN.md` for architecture and phase status.
2. Read the corresponding R function in `WEPPCLIFF.R` before writing Python.
3. **Do not modify** the Antigravity (R) repository unless explicitly asked.
4. Match **function names** to R (snake_case, identical spelling).
5. Preserve **CLI flag names and defaults** from `Documentation/Manual.md`.

## Project Conventions

### Directory Names (renamed from R)

Use standard capitalization everywhere:

- `Input/` not `INPUT/`
- `Output/` not `OUTPUT/`
- `Export/`, `Plots/`, `Installer/`, `Autotest/`, `Documentation/`, `Development/`, `Source/`

### Config vs Globals

R uses `<<-` global assignment. Python uses:

```python
from Source.config import WeppcliffConfig

config = WeppcliffConfig.from_args(args)
# Pass config to every function that reads flags in R
```

Never introduce module-level mutable state for user arguments.

### Data Container

R uses a numbered `data` list. Python uses `WeppcliffData` (see `Documentation/Planning/PLAN.md` §8). Index numbers in R comments (e.g., `data[[5]]`) must map to named attributes.

### Dependencies

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
scipy>=1.10
pymice[ml] @ git+https://github.com/ryanpmcg/PyMICE.git
# optional: pyarrow>=15  (only for -ef parquet)
```

Install via `Installer/Install.sh` or `pip install -e .`. **Do not** `pip install pymice` from PyPI alone — that resolves to a different project. PyMICE must come from [ryanpmcg/PyMICE](https://github.com/ryanpmcg/PyMICE). Parquet binary export: `pip install 'weppcliff[parquet]'`.

## Function Conversion Rules

1. **One R function → one Python function** (minimum). You may extract helpers if it improves clarity.
2. **Same inputs and outputs** — DataFrame column names must match R exactly (`DT_1`, `PRECIP`, `YYYYMMDD`, etc.).
3. **Datetime format** — Default `%Y-%m-%d %H:%M:%S`; respect `-dtf1`, `-dtf2`, `-dtf3`.
4. **Missing data sentinel** — `-99999` → `NaN` when `-pmd` is absent (default).
5. **Boolean flags** — Pure on/off CLI flags use **presence = true, absence = false** (`BOOLEAN_FLAGS` in `Source/config.py`). Config still stores `"t"` / `"f"`; use `config.is_true("qc")`. Not for multi-value flags (`-mp`, `-pi`, `-ai`, `-cv`, `-ed`, `-gf`, …).

## Known Divergences (Intentional)

### MICE Imputation (PyMICE Phase 6a–6d + 7a)

R uses the `mice` package inside `impute_by_cluster` / `impute_by_df`. Python uses **`pymice.integrations.weppcliff`** when **`-pm`** is present (omit for mean fill).

| Flag | Behavior |
|------|----------|
| `-pm` | FCS via PyMICE + resilience fallback chain (omit for mean fill) |
| `-ns` | Year-block predictors (nonstationary relationships) |
| `-kg` | Köppen zone for regional priors (`AUTO` infers from lat) |
| `-st` | Storm-template sub-daily imputation (with `-pm`; omit for row-wise FCS) |
| `-jm` | Daily **multivariate JOMO/PAN block** (`jomoImpute` / `panImpute`) instead of scatter FCS |

**Fallback order (6b daily):** local FCS (or JOMO when `-jm t`) → adjacent-cluster pool → `2l.norm` / unified JOMO → Köppen prior → mean fill.

**Sub-daily (6c):** storm-template donors preserve spacing; `resolve_overlaps()` replaces the R duplicate-merge loop.

**Validation (6d):** `test_gapfill_autotutorial.py`, `run_phigi_validation.py`, `compare_phigi_legacy.py` (full `fill_workflow` on KMQE; scenarios: `holdout_random`, `sorad_corrupt_summer`, `seasonal_gap`, `sparse_donors`).

**Experiment matrix (A–G, dual whole/hole scoring):** Canonical assessment is
`Development/scripts/run_experiment_matrix.py` → `Source/weppcliff_impute/experiment_matrix/`.
Reports daily/monthly/seasonal/annual stats for **whole series** and **the hole**.
See `Documentation/Planning/AmputeImputeExperimentMatrix.md`. Older single-path
multiseed (`autotest1_ampute_impute_multiseed.py`) remains for the 2026-07-26 paper freeze.

**Cluster granularity:** seasonal (4) → month → half-month → quarter-month. See `PYMICE_INTEGRATION_PLAN.md` and `FUTURE_IMPROVEMENTS_ROADMAP.md`.

### PHIGI gap-fill engine (`-gf phigi`)

Directed hierarchical imputation replaces staged symmetric MICE on the PHIGI path:

```
L2 weather-type fit → L3 solar (τ) → L3 daily → L4 precip hurdle → L5 sub-daily → L6 report
```

| Flag | Behavior |
|------|----------|
| `-gf phigi` | PHIGI pipeline (`fill_data_phigi`) |
| `-gf legacy` | Staged PyMICE / mean-fill (default) |
| `-tauqc` | τ QC bounds for solar step (`auto` = 0.05–1.15) |
| `-ir` | Write `Export/Imputation-Report.json` |
| `-pm` | PyMICE directed residual fill within L3 (optional) |

**L3 fallbacks:** τ-weighted cluster donors → adjacent weather-type components → L2 mixture → L1 Köppen prior → tiered low-confidence global mean (reported, not silent).

**L4/L5:** logistic EVENT hurdle, lognormal wet amounts, K-candidate I30 selection with KG-season template pooling.

Run PHIGI tests: `.venv/bin/python Development/scripts/run_phigi_validation.py`

### Export Binary Format

R writes `.rds`. Python writes compressed `.npz` for `-ed 1` / `-ed 2` (default `-ef npz`). `-ef parquet` writes a zip of per-slot parquet tables (PyArrow). Autotest does **not** compare binary exports.

### Autotest policy

- Compare `.cli` MD5 to R reference at `/Users/home/Software/Antigravity/WEPPCLIFF/AUTOTEST/OUTPUT/` when present; else `Autotest/Benchmarks/`.
- **Exclude** gap-filled runs (`-id t`) from autotest.
- `pytest Development/tests/test_autotest.py` enforces byte-identical `.cli` parity with R (skips if R reference missing).

### Rounding

Minor float differences vs old benchmarks are acceptable. Target parity with **current R output** on the same machine.

## Priority: CLI Output Bug

The user suspects a bug in CLI generation. Focus investigation on:

1. `generate_export_data` — breakpoint count adjustments around midnight
2. `create_cli_file` — breakpoint line writing loop, `last.pcp` indexing

When translating, **match R output byte-for-byte** even if the R logic looks wrong — compatibility first. Add a comment referencing the suspected issue.

Key R lines:

- `generate_export_data`: ~2317–2427
- `create_cli_file`: ~2431–2542

Validate with:

```bash
bash Autotest/AutoTest.sh
diff Autotest/Output/AutoTest.2.cli Autotest/Benchmarks/AutoTest.2.cli
```

## Running the Project

### Install

```bash
cd /Users/home/Software/Grok/WEPPCLIFF
bash Installer/Install.sh
```

### Single command (equivalent to R)

```bash
python weppcliff.py -f Input/ASOS_BPT_KMQE.csv -la y -verb
```

### Automated tutorial

```bash
bash Tutorial/AutoTutorial.sh
```

### Automated test

```bash
bash Autotest/AutoTest.sh
```

### Cross-validate against R

```bash
cd /Users/home/Software/Antigravity/WEPPCLIFF && bash AUTOTEST/AutoTest.sh
cd /Users/home/Software/Grok/WEPPCLIFF && bash Autotest/AutoTest.sh
```

## Module Ownership

| Module | R Functions | Status |
|--------|-------------|--------|
| `branding.py` | logo, license | Done |
| `paths.py` | directories, find_this_file | Done |
| `args.py` | all argument parsing | Done |
| `io.py` | load/export | Done |
| `cli_input.py` | *(Python extension)* CLIGEN `.cli` input | Done |
| `export_weather.py` | *(Python extension)* weather/KE/EI CSV exports | Done |
| `preprocess.py` | store, convert, preserve | Done |
| `utils.py` | find_replace, convert_df_type | Done |
| `duration.py` | duration, midnight breakpoints | Done |
| `units.py` | convert_units | Done |
| `quality.py` | full QC pipeline | Done |
| `storms.py` | storms, smart_window_sum | Done |
| `time_bounds.py` | time_bounds, start/end helpers | Done |
| `erosion.py` | calculate_erosion_indices | Done |
| `daily.py` | daily, alt, dew point | Done |
| `precip.py` | precip processing, aggregate | Done |
| `impute.py` + `weppcliff_impute/` | fill_data, PyMICE FCS, PHIGI dispatcher | Done |
| `phigi/` | L0–L6 hierarchical gap fill (`-gf phigi`) | In progress (L3–L6 shipped; hold-out gates) |
| `plots.py` | all graph_* functions | Done |
| `cli_output.py` | header, export, create_cli | Done (AutoTest.1 MD5 parity deferred) |
| `cli_validate.py` | *(Python extension)* structural CLI rules | Done |
| `workflow.py` | core, fill, parallel | Done |
| `main.py` | main, install | Done |

## Testing Requirements

Before marking a module done:

1. Add unit tests in `Development/tests/test_<module>.py`
2. Run `pytest tests/ -v`
3. If the module affects `.cli` output, run `bash Autotest/AutoTest.sh`

## Concurrency

- Use `concurrent.futures.ProcessPoolExecutor` for multi-station runs (`-mp t` with multiple files).
- Each worker runs `core_workflow(file, config)` with an isolated config copy.
- Disable parallel path when `mp=f` or only one station.

## Common Pitfalls

| Pitfall | R Behavior | Python Fix |
|---------|------------|------------|
| POSIX datetime | `as.POSIXct`, `difftime` | `pandas.to_datetime`, `.diff()` |
| `rle` | `rle(x)` | `(pd.Series(x) != pd.Series(x).shift()).cumsum()` or `itertools` |
| `aggregate` | `aggregate(..., by=list(...), FUN=)` | `df.groupby(...).agg(...)` |
| `merge(all.x=T)` | SQL left join | `pd.merge(..., how="left")` |
| `rbindlist` | data.table | `pd.concat` |
| `formatC` width padding | fixed-width CLI | Python f-strings with width spec |
| `sink` / `cat` | file write | `open(path, "w")` or `path.write_text` |
| Factor columns | `as.factor` | `astype("category")` or strings |
| `%/%` integer div | floor division | `//` |

## When Stuck

1. Run the equivalent R command with `-verb t` and compare intermediate exports (`-ed 1`).
2. Check `Autotest/Benchmarks/` for expected output.
3. Ask the user — per project instructions, **ask questions** rather than guessing on ambiguous behavior.

## Commit / PR Guidance

- One module or phase per logical commit.
- Message format: `feat(precip): port process_precip_data from R`
- Never commit `.venv/`, `Output/`, `Export/`, or generated plots.

## Version Target

Port of **WEPPCLIFF v1.7.1** (R). Python package version starts at **2.0.0** to indicate the language migration.