# WEPPCLIFF v2 Refactor Plan

**Version:** 1.0  
**Date:** 25 June 2026  
**Status:** Planning only — no code changes in this document  
**Prerequisite:** Python port validated for accuracy (structural CLI rules pass; midnight model verified on AutoTest.1)

## 1. Objective

Refactor the Python codebase for **speed** and **simplicity** while preserving **identical outputs** for all non-gap-fill workflows. This plan is a systematic, per-module review of `Source/` (~4,900 lines, 24 modules, ~90 functions).

### In scope

- Performance: vectorization, fewer DataFrame copies, compiled hot loops, startup/IO optimizations
- Simplicity: remove R-port artifacts, deduplicate logic, table-driven configs, explicit APIs
- Output parity: `.cli` byte identity (where currently targeted), QC flags, storm IDs, erosion indices, CSV exports

### Out of scope (deferred to MICE / gap-fill phase)

All gap-filling and imputation paths are **excluded** from refactor work until a Python MICE workflow is designed:

| Module / function | Reason |
|-------------------|--------|
| `impute.py` — entire module | Mean-fill placeholder; MICE parity pending |
| `workflow.fill_workflow` | Orchestrates `fill_data` |
| `workflow.core_workflow` branch `if config.is_true("id")` | Dispatches to gap-fill path |
| `precip.fill_agg_df` | Inserts missing observation rows for gap-filled series |
| `cli_breakpoint_time.gap_breakpoint_time` | Gap-line timing only |

Brief note on deferred `impute.py`: highest future wins are `fill_data` vectorization (replace `iterrows`), `create_cluster_col` counting (`value_counts` vs O(n²) min-length), and a real MICE implementation. **Parity risk: High** — require golden gap-fill fixtures before any change.

---

## 2. Verification strategy (before any PR)

Every refactor PR must pass:

```bash
.venv/bin/python -m pytest Development/tests/ -v
bash Autotest/AutoTest.sh
bash Tutorial/AutoTutorial.sh   # smoke
```

Additional gates by touch area:

| Area touched | Extra verification |
|--------------|-------------------|
| `smart_window_sum`, storm ID | Storm SID snapshots; connectivity flags on known breakpoint file |
| `generate_export_data`, `create_cli_file` | `.cli` byte diff vs R reference; `validate_cli_file` = 0 errors |
| `process_precip_data`, `process_daily_data` | ELEN/EMAX/EMIN, BPS, monthly/annual aggregates vs pickled slot exports |
| `calculate_erosion_indices` | Per-storm KE/EI/I30 vs reference (AH282/AH537/AH703 caps) |
| `quality.py` | QC flag vectors on known inputs; especially `snr_spike_test` boundary rows |
| `preserve_missing_data`, sentinel handling | `-99999` → NaN parity on sample frames |
| `plots.py` | Filename/path correctness after `chdir` removal (optional pixel snapshot) |

**Profiling baseline:** Run `cProfile` on AutoTest.1 and a large breakpoint file before Phase 1 to quantify wins.

---

## 3. Phased implementation plan

### Phase 0 — Infrastructure (low risk, enables later work)

| PR | Work | Files | Impact |
|----|------|-------|--------|
| 0.1 | Parse `argv` once; thread `locations` dict through `assign_args_to_vars` | `args.py`, `main.py` | High startup |
| 0.2 | Normalize all T/F flags to bool in `assign_final_args`; add `config.cv_off`, `config.do_qc`, etc. | `args.py`, `config.py`, `workflow.py` | Medium hot-path |
| 0.3 | DRY `_load_and_preprocess()` shared by `series_execution` and `_process_station` | `workflow.py` | Maintainability |
| 0.4 | Remove dead code: `if not ai: pass`, duplicate `parse_args`, redundant license branch | `workflow.py`, `main.py` | Clarity |
| 0.5 | Move `import pandas` to module top in `workflow.py` | `workflow.py` | Style |

### Phase 1 — Hot-path performance (highest ROI)

| PR | Work | Files | Impact | Parity risk |
|----|------|-------|--------|-------------|
| 1.1 | **Numba or sliding-window rewrite** of `smart_window_sum` | `storms.py` | **Very high** (10×+ plausible) | Medium |
| 1.2 | Vectorize `generate_export_data`: single datetime parse, grouped BPS/TPK, drop `iterrows` | `cli_output.py` | High | Medium |
| 1.3 | Single `groupby` in `process_precip_data`; replace `apply(lambda x: x.unique())` | `precip.py` | High | Medium |
| 1.4 | Pre-extract numpy arrays in `create_cli_file`; unify emit + simulate | `cli_output.py`, `cli_validate.py` | Medium–high | Medium–high |
| 1.5 | One `groupby.agg` in `process_daily_data` / `process_alt_data` (replace 6× `_aggregate`) | `daily.py` | Medium | Low |
| 1.6 | Parse `DT_1` once in `aggregate_precip_data` | `precip.py` | Medium | Medium |
| 1.7 | In-place / masked sentinel replacement in `preserve_missing_data` | `preprocess.py`, `utils.py` | High (large series) | Low |
| 1.8 | Vectorize `calculate_erosion_indices`: single-pass storm metrics, preallocated arrays | `erosion.py` | High (many storms) | Medium |

### Phase 2 — CLI layer consolidation

| PR | Work | Files | Impact | Parity risk |
|----|------|-------|--------|-------------|
| 2.1 | Shared `emit_day_breakpoints()` used by `create_cli_file` and `simulate_day_lines` | `cli_output.py`, `cli_validate.py` | Medium (eliminates double traversal) | Medium |
| 2.2 | Validate numerics pre-`_fmt_num` in `generate_export_data` | `cli_validate.py`, `cli_output.py` | Medium | Low |
| 2.3 | Vectorized `_fmt_num` for export columns | `cli_output.py` | Medium–high | Low |
| 2.4 | Remove `os.chdir` from CLI write; use `Path(config.o) / filename` | `cli_output.py` | Robustness | Low |
| 2.5 | Single file read in `validate_cli_file` | `cli_validate.py` | Low | None |

### Phase 3 — Pipeline and I/O cleanup

| PR | Work | Files | Impact | Parity risk |
|----|------|-------|--------|-------------|
| 3.1 | Slim multiprocessing worker payload (subset of config, not full pickle) | `workflow.py`, `config.py` | Medium (MP memory) | Medium |
| 3.2 | Optimize `export_pcp_bps`: `set` for DT dedup, fewer concat/sort passes | `io.py` | High (large exports) | Low |
| 3.3 | C-engine `read_csv` for delimiter-separated `.txt` where safe | `io.py` | Medium (large inputs) | Low |
| 3.4 | Table-driven `convert_units` | `units.py` | Clarity | Medium |
| 3.5 | Slot map helper / named constants for `5 if not id else 15` pattern | `workflow.py` | Readability | None |
| 3.6 | `WeppcliffData.copy()` semantics: document shallow vs deep; fix if pipelines branch | `data_model.py` | Correctness | Medium |

### Phase 4 — QC, plots, and polish

| PR | Work | Files | Impact | Parity risk |
|----|------|-------|--------|-------------|
| 4.1 | `snr_spike_test`: bitwise masks vs nested `intersect1d`; bounds on `highs+1` | `quality.py` | Medium | Medium |
| 4.2 | Batch QC updates per slot DataFrame in `run_nonprecip_checks` | `quality.py` | Medium | Low |
| 4.3 | Deduplicate `extreme_limit_test` / `extreme_limit_replace` | `quality.py` | Low | Low |
| 4.4 | Remove `os.chdir` from all plot functions; absolute `savefig` paths | `plots.py` | Robustness | Low |
| 4.5 | Consolidate `annual_plots` / `monthly_plots`; fix `graph_data` storm-plot control flow | `plots.py` | Maintainability | Low |
| 4.6 | Table-driven `check_args` with pre-compiled regexes | `args.py` | Low startup | None |

---

## 4. Cross-cutting patterns

| Pattern | Locations | Issue | Recommendation |
|---------|-----------|-------|----------------|
| Full DataFrame `.copy()` before in-place ops | `preprocess`, `utils.df_find_replace`, `storms`, `duration`, `cli_output` | Memory pressure on long breakpoint series | In-place path when caller owns frame; copy-on-write where pandas allows |
| R 1-based column indices | `preprocess.convert_data`, `preserve_missing_data`, `utils._resolve_cols` | Fragile after column reorder | Prefer column names; keep `_resolve_cols` documented as R-compat |
| `str(flag).upper() == "T"` scattered | `workflow`, `io`, `units` | Repeated string work | Bool flags set once in `assign_final_args` |
| Magic slot numbers `5`, `15`, `18`… | `workflow`, exports | Hard to read | `SlotIndex` enum or config properties `pcp_slot`, `dly_slot` |
| `get_config()` singleton | `cli_output.generate_header_data`, global reads | Testability, implicit deps | Pass `config` explicitly (keep singleton for transition) |
| `os.chdir` for output | `cli_output`, `plots` | Thread-unsafe, side effects | Always join `config` dir paths |
| Repeated `convert_df_type` + `df_find_replace` blocks | `precip`, `daily` | Boilerplate | `_finalize_ts_df(df, key_col, config)` helper |
| Multiple `groupby(dates)` on same Series | `precip`, `daily` | Redundant passes | Factor `dates` once; single `.agg()` |
| `iterrows()` in hot paths | `cli_output`, `cli_validate`, `quality` (indirect) | Slow Python loops | Vectorized groupby / numpy indexing |
| Duplicated emission logic | `create_cli_file` vs `simulate_day_lines` | Drift risk, double CPU | Single source of truth for breakpoint emission |

---

## 5. Per-module recommendations

### `storms.py` — **highest priority**

| Function | Current approach | Issues | Recommendation | Impact | Parity |
|----------|------------------|--------|----------------|--------|--------|
| `smart_window_sum` | Python `for j in range(n)`; per-iteration `np.arange` window slice | **Primary bottleneck** for storm ID and I30; O(n × window) pure Python | Numba `@njit` with preallocated buffers, or two-pointer sliding window on monotonic `CUM` | **Very high** | Medium |
| `evaluate_connectivity` | Calls `smart_window_sum` twice; copies df | Dominated by window sum | Drop copy; share `rows` computation with `int_30m` when both run | Small | Low |
| `int_30m` | Duplicate `rows`/`min_dur` vs connectivity | Redundant work | Hoist shared `rows` calc | Small | Low |
| `cum_to_inc` | Full copy; index scan for zero precip | Extra allocation | Boolean mask `df.loc[df["PRECIP"] != 0]` | Medium | Low |
| `identify_storms` | `out.copy()` | Unnecessary | `df.assign(...)` | Low | Low |
| `break_storms` | List of per-storm DataFrames | Memory | Keep if API requires; else single frame + SID | Memory | Depends |
| `trim_data` | 4× `to_datetime` per df in min/max lists | Redundant parsing | One parse per slot DataFrame | Low | Low |

### `cli_output.py` — **second priority**

| Function | Current approach | Issues | Recommendation | Impact | Parity |
|----------|------------------|--------|----------------|--------|--------|
| `_fmt_num` | Scalar format; called via `.map(lambda)` thousands of times | Callback overhead | Vectorized string format for export columns | Medium–high | Low |
| `generate_header_data` | `get_config()` inside; 12× `_fmt_num` | Global coupling | Pass `config` arg | Low | None |
| `generate_export_data` | Multiple `.copy()`; per-day loops for BPS/gaps; `iterrows` for TPK; 4× `groupby`; `.map(_fmt_num)` on all cols | **Major hot path** | Parse `DT_1` once; vectorized BPS adj; `groupby` + `idxmax` for TPK; single `.agg()` for AMOUNT/DURATION/TP/IP | **High** | Medium |
| `create_cli_file` | `pcp_df.copy()`; string `DT_1`; per-row `iloc` in breakpoint loop; `os.chdir`; duplicate sim in validator | Inner loop bound by breakpoint days | Pre-extract arrays; shared emitter; remove `chdir` | Medium–high | Medium–high |

### `precip.py`

| Function | Current approach | Issues | Recommendation | Impact | Parity |
|----------|------------------|--------|----------------|--------|--------|
| `aggregate_precip_data` | 6× `pd.to_datetime` on `DT_1`; Python `set` union of timestamps | Repeated parsing | Parse once; `Index.union` | Medium–high | Medium |
| `fill_agg_df` | **EXCLUDED** — gap-fill row insertion | — | Defer | — | — |
| `process_precip_data` | 4× `groupby(dates)`; `apply(lambda x: x.unique())` for events; repeated type/replace blocks | SID stats slow per day | Single `groupby.agg`; shared `_finalize_ts_df` | **High** | Medium |

### `daily.py`

| Function | Current approach | Issues | Recommendation | Impact | Parity |
|----------|------------------|--------|----------------|--------|--------|
| `_aggregate` | Temp DataFrame per call; called 6× on same index | Allocations | Fold into one `groupby.agg` in caller | Medium | Low |
| `process_daily_data` | Six aggregates; triple monthly groupby; duplicate finalize boilerplate | Same as precip pattern | Parse date once; shared helpers | Medium | Low–medium |
| `process_alt_data` | Five `_aggregate` calls after dew point | Same | One groupby after `dp_temp` column | Medium | Low |
| `dew_point_temperature` | `rh.copy()`; `rh[rh==0]=1` | Minor copy | `np.where(rel_hum == 0, 1, rel_hum)` | Small | Low |
| `use_alt_data` | Positional `iloc[:, [0,7,8,...]]` | Fragile | Select/rename by column name | Clarity | Low |
| `annual_summary` | `describe()` to slot | Fine | No change | — | — |

### `cli_validate.py`

| Function | Current approach | Issues | Recommendation | Impact | Parity |
|----------|------------------|--------|----------------|--------|--------|
| `validate_raw_bps` | `pcp_df.copy()`; `daily_df.iterrows()` | Slow on many days | Vectorized merge/compare on `YYYYMMDD` | Medium | Low |
| `validate_export_bps` | `iloc` loop; parses formatted strings | Validates too late | Move to numeric pre-format in export | Medium | Low |
| `simulate_day_lines` | Duplicates `create_cli_file` loop logic | Double CPU today | Merge with shared emitter (Phase 2) | Medium | Medium |
| `validate_breakpoint_sequence` | Four passes over breakpoint list | Fine for small n | Optional single-pass | Low | Low |
| `parse_cli_days` / `validate_cli_file` | File read twice | Redundant IO | Single read | Low | None |
| `ymd_to_date_key`, helpers | Scalar string ops | OK for CLI parse | Vectorized variant for DataFrame paths | Low | Low |

### `cli_breakpoint_time.py`

| Function | Current approach | Issues | Recommendation | Impact | Parity |
|----------|------------------|--------|----------------|--------|--------|
| `interval_crosses_midnight` | `int(hour)`, `int(min)` per call | String parse in hot loop | Numeric minute-of-day from caller arrays | Low–medium | Low |
| `storm_start_time_cligen` | String → float each call | Same | Numeric API at array boundary | Low | Low |
| `first_breakpoint_time` | Branch on midnight cross | Minimal | Optional fused numeric version | Low | Low |
| `gap_breakpoint_time` | **EXCLUDED** | — | Defer | — | — |
| `should_cap_last_breakpoint_at_midnight` | Early exits; clear logic | Re-parses strings | Pass precomputed `crosses_midnight` | Negligible | Low |

### `duration.py`

| Function | Current approach | Issues | Recommendation | Impact | Parity |
|----------|------------------|--------|----------------|--------|--------|
| `create_duration_data` | Per-slot copy; `set(dt_parsed)` for gap fill; triple index union for zero precip | Python set of Timestamps | Anti-join / boolean mask for zero rows; parse datetime once per slot | Medium | Medium |
| `modify_midnight_breakpoints` | Copy; fractional split; `concat` new rows | CLIGEN-sensitive math | Build new rows array in one concat; **do not change math** | Low–medium | **High** |

### `erosion.py`

| Function | Current approach | Issues | Recommendation | Impact | Parity |
|----------|------------------|--------|----------------|--------|--------|
| `_select_equations` | Map `ee` to equation list | Fine | Cache on config | Negligible | None |
| `_user_equation` | `eval()` per call; mutates global `EQN_FUNCS` | Security (expected for USER) | `compile()` once; local dict not global | Small | Low–medium |
| `calculate_erosion_indices` | Nested loops: equations × storms; repeated `assign`; list → DataFrame | O(eq × storms) allocations | Precompute storm metrics once; vectorize per equation; unify AH* caps | **High** | Medium |

### `quality.py`

| Function | Current approach | Issues | Recommendation | Impact | Parity |
|----------|------------------|--------|----------------|--------|--------|
| Module constants | Parallel `NM`, `R_LOC`, `I_LOC` arrays | `R_LOC == I_LOC`; index alignment fragile | `NamedTuple` / dataclass records per variable | Clarity | Low |
| `create_rate_qc_data` | Per-var loop; dead length guard | Minor | Batch slot 5–7 fetches | Small | Low |
| `run_nonprecip_checks` | Per-var `slot_df` + full writeback | Repeated fetches | Hoist thresholds; batch per slot | Medium | Low |
| `extreme_limit_test` / `extreme_limit_replace` | Near-duplicate masking | DRY opportunity | Shared `_out_of_limits_mask` | Low | Low |
| `rle_streak_test` | Python loop over streak indices | Slow on long streaks | Vectorize via run boundaries | Medium | Low |
| `snr_spike_test` | 3× `nanquantile`; nested `intersect1d`; `highs+1` boundary | Expensive; boundary edge cases | Bitwise `&`; explicit bounds trim | Medium–high | **Medium** |
| `rle_stick_test` | Loop over short PASS runs | Hard to vectorize | Keep; document R parity | Low | Medium if wrong |
| `run_precip_checks` | Full copy for no-op `return_period_test` | Unnecessary copy | In-place until RP implemented | Small | Low |
| `return_period_test` | Stub | — | No change | — | — |
| `remove_qc_fails` | Series copy + boolean index | Minor | `df.loc` single assign | Small | Low |
| `impose_daily_value_limits` | Rebuilds limit dicts each call | Trivial waste | Module-level `DAILY_LIMITS` dict | Negligible | Low |

### `workflow.py`

| Function | Current approach | Issues | Recommendation | Impact | Parity |
|----------|------------------|--------|----------------|--------|--------|
| `create_cluster` | `multiprocessing` import inside | Minor | Top-level import | Low | None |
| `kill_cluster` | Print only | Misleading R name | Rename or remove | None | None |
| `fill_workflow` | **EXCLUDED** | — | Defer | — | — |
| `core_workflow` | Long linear pipeline; dead `ai` branch; repeated flag strings; bare `except` on plots | Readability | Precomputed bools; slot map; log plot errors | Low–medium | Low |
| `_process_station` | Full config pickle; `chdir` | Large MP payload | Slim worker context | Medium | Medium |
| `parallel_execution` | `ProcessPoolExecutor`; log file opened `"w"` each completion | Log may overwrite | Append or buffer log lines | Low | Low |
| `series_execution` | Duplicates load/preprocess with `_process_station` | DRY | Shared helper (Phase 0) | Low | None |

### `preprocess.py`

| Function | Current approach | Issues | Recommendation | Impact | Parity |
|----------|------------------|--------|----------------|--------|--------|
| `preprocess_file` | Full `.copy()` to add columns | Memory | `reindex` columns or in-place add | Medium | Low |
| `store_data` | 12× `_series`; 3 new DataFrames from frame | Column copies | `frame[["DT_1","PRECIP"]].copy()` per slot; metadata from `iloc[0]` | Medium | Low |
| `convert_data` | `convert_df_type` copies each slot | Double copy | In-place column conversion option | Medium | Low |
| `remove_missing_datetime_rows` | `.loc` + `reset_index` × 3 slots | Allocations | Skip reset if index not relied upon | Low–medium | Low |
| `preserve_missing_data` | `df_find_replace` full copy per slot | **Hot path** on load | Masked `numpy` / `pd.replace` in place | **High** | Low |

### `io.py`

| Function | Current approach | Issues | Recommendation | Impact | Parity |
|----------|------------------|--------|----------------|--------|--------|
| `load_input_file` | Substring extension check; `engine="python"` for txt | Slow C-engine path | `path.suffix`; try C engine first | Medium | Low |
| `check_input_format` | `pd.DataFrame(file)` redundant wrap | Extra alloc | Use frame directly | Low–medium | None |
| `load_input_directory` | Substring ext; `set` dedup loses order | Edge cases | `suffix`; `sorted(unique)` | Low | Low |
| `export_data` | Repeated setup in children | Minor | Hoist dir/name resolution | Low | None |
| `export_to_csv` | Repeated path resolution | Minor | Accept pre-resolved paths | Low | None |
| `export_pcp_bps` | Concat + sort; `isin` O(n²) | Large breakpoint exports | `set` for DT membership; fewer passes | **High** | Low |

### `utils.py`

| Function | Current approach | Issues | Recommendation | Impact | Parity |
|----------|------------------|--------|----------------|--------|--------|
| `vec_find_replace` | Full `pd.to_numeric` even when only `find` | Multi-pass | Branch: coerce only when `inf`/`nan` flags | Medium | Medium |
| `_resolve_cols` | R 1-based indexing | Intentional | Document; do not “fix” to 0-based | — | **High** if changed |
| `df_find_replace` | Always `df.copy()` | Memory | Optional in-place | Medium | Low |
| `rle` | Numpy adjacent compare | Already efficient | No change | — | — |
| `convert_df_type` | Per-column astype + to_numeric | Full copy | In-place on owned copy | Small | Low |
| `view_data`, `stop_quietly` | Dev/CLI utilities | Not hot path | No change | — | — |

### `units.py`

| Function | Current approach | Issues | Recommendation | Impact | Parity |
|----------|------------------|--------|----------------|--------|--------|
| `convert_units` | Sequential `if` per variable; regex for SO_RAD exception | Repetitive R port | Table-driven `CONVERTERS` dict; explicit SO_RAD check | Clarity | Medium |

### `time_bounds.py`

| Function | Current approach | Issues | Recommendation | Impact | Parity |
|----------|------------------|--------|----------------|--------|--------|
| `_to_timestamp` | Redundant type branches | Verbose | `pd.Timestamp(value)` | Negligible | Low |
| `start_of_year` / `end_of_year` | Repeated microsecond truncation | Duplication | Shared `_truncate` helper | Clarity | Medium |
| `start_of_day` / `end_of_day` | `config` unused (R signature) | Noise | `normalize()` / `Timedelta` | Negligible | Low |
| `time_bounds` | Duplicated validation blocks | Verbose | `_parse_bounds()` helper | Clarity | Low |

### `args.py`

| Function | Current approach | Issues | Recommendation | Impact | Parity |
|----------|------------------|--------|----------------|--------|--------|
| `parse_args` | Two argv scans (parse + locations) | Startup cost | Merge into one pass | Medium | None |
| `parse_arg_locations` | Called 52× from `_display_value` | **High startup cost** | Parse once (Phase 0) | **High** | None |
| `_display_value` | Re-parses locations every flag | Same | Accept `locations` param | **High** | None |
| `assign_args_to_vars` | 52-flag print loop | R `specified` count hack | Precomputed locations | Medium | Low |
| `assign_empty_args` | ~80 sequential `if not` defaults | Verbose | Defaults dict / `__post_init__` | Maintainability | Low |
| `check_args` | Recompiles regex each run | Minor | Module-level compiled patterns | Low | None |
| `assign_final_args` | Only `pmd` → bool | Hot-path string ops remain | All T/F → bool | Medium | Low |

### `config.py`

| Function / member | Current approach | Issues | Recommendation | Impact | Parity |
|-------------------|------------------|--------|----------------|--------|--------|
| `get_config` / `set_config` | Module singleton | Implicit deps | Long-term: explicit threading; short-term keep | Architectural | Medium |
| `is_true` | `str.upper()` each call | Hot path | Cached bools after assign_final | Low–medium | Low |
| `copy` | `deepcopy` | Heavy for MP | Slim worker payload | Medium | Medium |
| `ensure_datetime_formats` | Defaults dtf2/dtf3 | Duplicated in `assign_empty_args` | Single source | Low | None |
| `parse_unit_vars` | Rebuilds dash positions twice | Minor | Parse once at CLI time | Low | Low |
| `from_args` | Re-parses argv | Duplicate work | Accept pre-parsed config | Low | None |

### `data_model.py`

| Function | Current approach | Issues | Recommendation | Impact | Parity |
|----------|------------------|--------|----------------|--------|--------|
| `WeppcliffData` slots | 1-based R-style dict | Fine | Document iteration order | — | — |
| `copy` | Shallow dict copy | DataFrames shared | Rename `shallow_copy` or deep-copy slots when branching | Correctness | Medium |
| `get_metadata` / `update_metadata` | Lazy dict; extra copies in update | Minor | Normalize at `store_data`; in-place update | Low | Low |
| `slot_df` | Returns live reference | Good for speed | Optional `copy=True` for safe reads | Low | None |

### `main.py`

| Function | Current approach | Issues | Recommendation | Impact | Parity |
|----------|------------------|--------|----------------|--------|--------|
| `install_dependencies` | Two pip subprocesses | Rare path | Single invocation | Low | None |
| `load_packages` | Fake “0.00 seconds” | Called twice sometimes | Call once | None | Low (console) |
| `main` | May double-parse args via `cli_entry` | Startup | Single parse path | Low | None |
| `cli_entry` | `main(parse_args())` | Same | Pass config through | Low | None |

### `plots.py`

| Function | Current approach | Issues | Recommendation | Impact | Parity |
|----------|------------------|--------|----------------|--------|--------|
| `create_input_plots` | `os.chdir` + savefig | Thread-unsafe | Full path in `savefig` | Robustness | Low |
| `annual_plots` / `monthly_plots` | Near-duplicates + chdir | DRY | Shared bar-compare helper | Maintainability | Low |
| `daily_plots` | chdir; manual NaN mask | Minor | `np.where`; full path | Small | Low |
| `storm_plots` | `isin` per column in loop | O(columns × n) | Precompute `dup_mask` once | Medium | Low |
| `create_other_plots` | chdir | Same | Full path | Small | Low |
| `graph_data` | Nested loops; bare `except`; storm plots inside var loop | Silent failures | Registry table; storm plots once | Reliability | Low |

### `paths.py` / `branding.py`

| Function | Current approach | Issues | Recommendation | Impact | Parity |
|----------|------------------|--------|----------------|--------|--------|
| `find_this_file` | argv / `__file__` resolution | Fine | `@lru_cache` if hot | Negligible | Low |
| `set_wds`, `create_directory`, `create_base_directory` | Straightforward | None | No change | — | — |
| `print_*`, `prompt_license_agreement` | Print / stdin | Not hot path | No change | — | — |

### `impute.py` — **EXCLUDED (summary only)**

| Function | Deferred action |
|----------|-----------------|
| `_max_iter` | Revisit with MICE iteration policy |
| `impute_by_cluster` / `impute_by_df` | Replace mean-fill with MICE or sklearn iterative imputer |
| `create_cluster_col` | `value_counts` instead of O(n²) min-length |
| `find_missing_precip` | Vectorize event/month streak logic |
| `fill_data` | Remove `iterrows`; batch imputation; reduce concat/merge |

---

## 6. Priority matrix (non-gap-fill)

| Rank | Target | Expected gain | Effort | Parity risk |
|------|--------|---------------|--------|-------------|
| 1 | `storms.smart_window_sum` | Very high | Medium | Medium |
| 2 | `cli_output.generate_export_data` | High | High | Medium |
| 3 | `precip.process_precip_data` | High | Medium | Medium |
| 4 | `preprocess.preserve_missing_data` + `utils` | High | Medium | Low |
| 5 | `cli_output.create_cli_file` + validator unification | Medium–high | High | Medium–high |
| 6 | `erosion.calculate_erosion_indices` | High (EI runs) | Medium | Medium |
| 7 | `args.parse_arg_locations` dedup | High (startup only) | Low | None |
| 8 | `daily` single groupby | Medium | Low | Low |
| 9 | `io.export_pcp_bps` | High (exports) | Medium | Low |
| 10 | `quality.snr_spike_test` | Medium | Medium | Medium |
| 11 | `plots` chdir removal | Robustness | Low | Low |
| 12 | `workflow` MP payload | Medium (multi-station) | Medium | Medium |

---

## 7. Architectural directions (v2, post-refactor)

These are **not** required for the performance refactor but align with a cleaner v2 codebase:

1. **Explicit config threading** — Reduce `get_config()` reads; pass `WeppcliffConfig` through pipeline functions.
2. **Typed slot access** — Replace magic indices with named properties on `WeppcliffData` (e.g. `data.precip_qc`, `data.daily_ts`).
3. **Pipeline stages as callables** — `core_workflow` becomes a ordered list of `(name, fn)` for easier testing and profiling per stage.
4. **Separate validation from emission** — `cli_validate` consumes shared `DayEmissionPlan` structs built once in `cli_output`.
5. **Optional dependencies** — `numba` as optional extra for `smart_window_sum` acceleration.

---

## 8. Success criteria

Refactor is complete when:

- [ ] All Phase 0–4 PRs merged or explicitly deferred with rationale
- [ ] `pytest Development/tests/` — same pass/xfail counts as pre-refactor
- [ ] `Autotest/AutoTest.sh` — structural validation passes; R MD5 parity policy unchanged
- [ ] `cProfile` shows ≥30% wall-time reduction on AutoTest.1 (target; measure baseline first)
- [ ] No regression in `.cli` validation (0 errors on AutoTest.1)
- [ ] Gap-fill paths untouched; documented in `agent.md` as deferred
- [ ] `Documentation/Planning/agent.md` updated with any new conventions introduced during refactor

---

## 9. References

| Document | Path |
|----------|------|
| Original conversion plan | `Documentation/Planning/PLAN.md` |
| Agent guide | `Documentation/Planning/agent.md` |
| CLI breakpoint design | `Documentation/Planning/CLI_Breakpoint_Enforcement_Plan.md` |
| Tests | `Development/tests/` |
| R reference | `/Users/home/Software/Antigravity/WEPPCLIFF/WEPPCLIFF.R` |

---

*Planning document only. Implementation proceeds PR-by-PR per Phase order above.*