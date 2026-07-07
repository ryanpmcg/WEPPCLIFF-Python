# Agent Guide — WEPPCLIFF Python Port

This document directs AI agents working on the R → Python conversion of WEPPCLIFF.

## Quick Reference

| Item | Path |
|------|------|
| **Python target** | `/Users/home/Software/Grok/WEPPCLIFF/` |
| **R source (read-only reference)** | `/Users/home/Software/Antigravity/WEPPCLIFF/WEPPCLIFF.R` |
| **Implementation plan** | `Documentation/Planning/PLAN.md` |
| **Gap-fill redesign (PHIGI)** | `Documentation/Planning/PHIGI_Implementation_Roadmap.md` |
| **PHIGI modules** | `Source/phigi/` (L0 astronomy + L3 τ solar step; `-gf phigi`) |
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
```

Install via `Installer/Install.sh` or `pip install -e .`. **Do not** `pip install pymice` from PyPI alone — that resolves to a different project. PyMICE must come from [ryanpmcg/PyMICE](https://github.com/ryanpmcg/PyMICE).

## Function Conversion Rules

1. **One R function → one Python function** (minimum). You may extract helpers if it improves clarity.
2. **Same inputs and outputs** — DataFrame column names must match R exactly (`DT_1`, `PRECIP`, `YYYYMMDD`, etc.).
3. **Datetime format** — Default `%Y-%m-%d %H:%M:%S`; respect `-dtf1`, `-dtf2`, `-dtf3`.
4. **Missing data sentinel** — `-99999` → `NaN` when `-pmd f` (default).
5. **Boolean flags** — R uses `"t"` / `"f"` strings; keep the same convention in config, compare with `.upper() == "T"`.

## Known Divergences (Intentional)

### MICE Imputation (PyMICE Phase 6a–6d + 7a)

R uses the `mice` package inside `impute_by_cluster` / `impute_by_df`. Python uses **`pymice.integrations.weppcliff`** when **`-pm t`** (default).

| Flag | Behavior |
|------|----------|
| `-pm t` | FCS via PyMICE + resilience fallback chain |
| `-pm f` | Legacy cluster/global **mean fill** |
| `-ns t` | Year-block predictors (nonstationary relationships) |
| `-kg` | Köppen zone for regional priors (`AUTO` infers from lat) |
| `-st t` | Storm-template sub-daily imputation (default with `-pm t`) |
| `-st f` | Row-wise FCS for sub-daily breakpoints |
| `-jm t` | Daily **multivariate JOMO/PAN block** (`jomoImpute` / `panImpute`) instead of scatter FCS |

**Fallback order (6b daily):** local FCS (or JOMO when `-jm t`) → adjacent-cluster pool → `2l.norm` / unified JOMO → Köppen prior → mean fill.

**Sub-daily (6c):** storm-template donors preserve spacing; `resolve_overlaps()` replaces the R duplicate-merge loop.

**Validation (6d):** `test_gapfill_autotutorial.py`, `run_gapfill_validation.py`, PyMICE `validation.py` harness.

**Cluster granularity:** seasonal (4) → month → half-month → quarter-month. See `PYMICE_INTEGRATION_PLAN.md` and `FUTURE_IMPROVEMENTS_ROADMAP.md`.

### Export Binary Format

R writes `.rds`. Python writes `.pkl` (pickle) for `-ed 1` / `-ed 2`. Autotest does **not** compare binary exports.

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
python weppcliff.py -f Input/ASOS_BPT_KMQE.csv -la y -verb t
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
| `impute.py` + `weppcliff_impute/` | fill_data, PyMICE FCS | Done (gap-fill CLI-input path has open bug) |
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