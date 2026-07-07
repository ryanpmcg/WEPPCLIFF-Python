# CLI Breakpoint Enforcement Plan

Investigation-only deliverable for AutoTest.1 (continuous / breakpoint mode, `bf=1`).
No production code was changed during this analysis.

## Executive summary

| Question | Answer |
|----------|--------|
| Does incongruence start upstream of `.cli`? | **No.** Slot 5 (precip QC) and slot 10 (daily `BPS`) are **byte-for-byte equivalent** between R `.rds` and Python `.pkl` on all 6,940 daily rows. |
| Where do R vs Python diverge? | **`generate_export_data`** (breakpoint accounting adjustments) and **`create_cli_file`** (loop indexing, time/depth emission). |
| Does Python violate the stated rules? | **No**, when the `.cli` is parsed with an NA-aware daily-line regex. Python has **0** days where declared breaks ≠ actual lines. R has **113** such days. |
| Were the 8 “Python violations” in the first diff report real? | **Mostly no.** Three were false positives caused by the analysis parser not recognizing daily rows with `NA` weather fields; re-validation with a corrected parser eliminated all Python rule failures. |

Supporting artifacts:

- `Development/reports/AutoTest.1_Breakpoint_Diff_Report.txt` — R vs Python `.cli` comparison (parser uses strict numeric daily regex; see §5)
- `Development/reports/AutoTest.1_Upstream_BPS_Report.txt` — slot 5/10 upstream comparison
- `Development/scripts/analyze_upstream_bps.py` — upstream comparison script
- `Development/scripts/analyze_test1_breakpoints.py` — `.cli` diff / validation script

---

## Validation rules to enforce

These are the invariants the user specified. They should be enforced on **emitted `.cli` content** (and on in-memory structures immediately before write).

| # | Rule | Notes |
|---|------|-------|
| 1 | Daily `Breaks` column == number of breakpoint lines that follow | Hard invariant. Python already satisfies this; R does not on 113 days. |
| 2 | First breakpoint line depth == `0.00` | Start-of-storm marker. |
| 3 | Within a day: breakpoint times non-decreasing | Allow `24.000 → 0.xxx` only if explicitly treated as midnight wrap (see §4). |
| 4 | Within a day: cumulative depths non-decreasing | Flat plateaus (same depth, later time) = zero increment; allowed. |
| 5 | No breakpoint time > `24.000` hours | Upper bound. |
| 6 | *(Recommended addition)* First breakpoint time ≥ `0.000` | Python emits **243** negative start times (e.g. `-0.033`) when `DUR/60 > HOUR+MIN/60`. R does similarly. Not in original rules; decide whether to clamp or carry to previous day. |

---

## Pipeline map and checkpoint placement

```
Input CSV
  → preprocess / storms / duration        [slot 1 → slot 5 precip]
  → process_precip_data                   [slot 10 daily BPS = row count/day]
  → generate_export_data                  [midnight adj, +1, rain-gap adjbrkpt]
  → create_cli_file                       [sequential pcp_df walk → lines]
  → AutoTest.1.cli
```

### Checkpoint A — post `process_precip_data` (slot 10)

**Validate:**

- `BPS` for each `YYYYMMDD` equals `groupby(date).size()` on slot 5 precip.
- `BPS == 0` or `NaN` on no-precip days; non-negative integers otherwise.

**AutoTest.1 result:** R and Python identical (0 differing days). No enforcement needed here for R/Py parity; keep as a regression guard.

### Checkpoint B — post `generate_export_data`

**Validate (per day):**

| Field | Role | Source in code |
|-------|------|----------------|
| `raw_bps` | Precip row count from slot 10 | `daily_df["BPS"]` before adjustments |
| `loop_bps` | Drives `create_cli_file` iteration | `daily_df["BPS"]` after midnight adj and `+1` (stored as `nbrkpt_vals` in Python) |
| `header_bps` | Written as `adjbrkpt` on daily line | `updated_bps` after rain-gap (`BTW`) adjustments |

**Invariants:**

- `loop_bps == 0` ⇒ no breakpoint block for that day.
- `header_bps >= loop_bps` (gap days add extra declared breaks; both R and Py show **1,627** days with `header_bps != loop_bps`).
- `header_bps == 0` ⇒ `loop_bps == 0`.

**AutoTest.1 result:** R and Python `loop_bps` and `header_bps` match on every day. Divergence is **not** in adjustment arithmetic at this stage.

**Known design tension:** The daily line prints `adjbrkpt` (`header_bps`) but the write loop bounds use `loop_bps`. On gap-heavy days, declared breaks can exceed lines written unless gap lines are also emitted. R exhibits this mismatch; Python matches R’s algorithm and matches declared-to-actual when parsed correctly.

### Checkpoint C — inside `create_cli_file` (pre-write)

**Validate per day before appending lines:**

1. `current_pcp` points at the first slot-5 row for this calendar day (or the correct carry-over row for midnight-spanning storms).
2. `last_pcp = current_pcp + loop_bps - 2` satisfies `1 <= current_pcp <= last_pcp + 1 <= nrow(pcp_df)`.
3. Simulated line count from the loop matches `header_bps` **or** document intentional gap accounting delta.
4. Emitted times/depths satisfy rules 2–5 (and 6 if adopted).

**AutoTest.1 result:** This is where **all R vs Python `.cli` divergence** manifests:

- **1,028** days: declared break count differs by −1 (514), +1 (477), or +2 (37).
- **103** days: same declared count but different breakpoint times/depths (rounding or gap-line timing).
- **5,912** days: identical declared count and matching breakpoints.

### Checkpoint D — post-write `.cli` parse (Autotest gate)

Re-parse written file with an **NA-aware** daily regex (all six weather fields may be `NA`):

```python
NUM = r"(-?\d+\.\d+|NA)"
DAILY_RE = re.compile(
    rf"^\s*(\d+)\s+(\d+)\s+(\d{{4}})\s+(\d+)\s+"
    rf"{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s*$"
)
```

Reject file if any day fails rules 1–5.

---

## Root-cause analysis by symptom class

### 1. Upstream data (eliminated)

`analyze_upstream_bps.py` compared R `.rds` vs Python `.pkl`:

| Stage | Differing days |
|-------|----------------|
| A. Raw slot 10 `BPS` | 0 |
| B. Slot 5 row count / day | 0 |
| C. `loop_bps` | 0 |
| D. `header_bps` | 0 |

**Conclusion:** Fix effort should **not** target `process_precip_data`, storms, or duration for AutoTest.1 parity.

### 2. ±1 declared break count (1,028 days)

Symmetric skew (514 at −1, 477 at +1) indicates **off-by-one in midnight / gap / `+1` accounting** between implementations, not random data corruption. First divergent day is `2000-01-04` (R=93, Py=94).

**Likely loci (in priority order):**

1. `generate_export_data` — `decrease_days` detection (`00:00:00` rows), `daily_bps_adj`, rain-gap `BTW` day assignment, `rle` aggregation of gap increases (R uses `rle(increase_days)`; Python uses `Counter` — verify equivalence on multi-gap days).
2. `create_cli_file` — `current_pcp` / `last_pcp` 1-based indexing; gap-line emission when `BTW > 0` and `new.day == F`.

### 3. Same count, different values (103 days)

Typically ±0.01 depth or ±0.001 time. Check:

- `_fmt_num` rounding vs R `format(round(...))`.
- Midnight `HOUR=24` substitution scope.
- Order of gap-line vs precip-line emission.

### 4. Negative first breakpoint times (243 Python lines)

Formula: `bp_time = HOUR + MIN/60 - DUR/60`. When storm starts just after midnight with long first duration, time goes negative.

**Enforcement option:** clamp to `0.000`, or assign to previous calendar day (R behavior may differ). Requires a product decision.

### 5. False “catastrophic” violations in the original diff report

The original `analyze_test1_breakpoints.py` daily regex required numeric `T-Max`/`T-Min` only. Daily rows with `NA` in **any** weather column were skipped as day boundaries, causing breakpoint lines to “bleed” across months (e.g. `2018-02-22` reported as 921 lines; actual file has **14**).

**Action for plan:** Fix the Autotest parser before using violation counts as CI gates.

---

## Proposed enforcement architecture

### Module: `Source/cli_validate.py` (new)

```text
validate_raw_bps(daily_df, pcp_df) -> list[Violation]
validate_export_bps(export, pcp_df) -> list[Violation]
simulate_day_lines(day_idx, export, pcp_df) -> SimResult
validate_day_emission(sim, header_bps, lines) -> list[Violation]
validate_cli_file(path) -> list[Violation]
```

`Violation` dataclass: `date`, `stage`, `rule`, `message`, `severity` (`error` | `warn`).

### Integration points

| Location | Behavior |
|----------|----------|
| `workflow.py` after `process_precip_data` | Run `validate_raw_bps`; warn on failure (data QC issue). |
| `cli_output.generate_export_data` return | Run `validate_export_bps`; **error** if `header_bps < loop_bps` or negative values. |
| `cli_output.create_cli_file` daily loop | Run `simulate_day_lines` + `validate_day_emission` **before** appending; accumulate violations. |
| `create_cli_file` end | If any `error` severity: raise `CLIValidationError` with day list; do **not** write file (or write to `.cli.invalid` when `verb=t`). |
| `Autotest/AutoTest.sh` | After run, call `validate_cli_file` with NA-aware parser; fail test on errors. |

### Single source of truth for break count

**Problem:** `adjbrkpt` (header) and `loop_bps` (iterator) diverge by design on gap days.

**Recommended approach:**

1. Introduce explicit fields on export: `loop_bps`, `header_bps`, `expected_lines`.
2. Define `expected_lines` as the **exact** number of lines the write loop will emit (including gap lines and the depth-0 starter).
3. Set daily `Breaks` column from `expected_lines`, not from `header_bps` alone.
4. Until that refactor, enforcement should validate **emitted lines == parsed breaks column**, not `header_bps == loop_bps`.

This preserves CLIGEN readability while making rule 1 mechanically enforceable.

---

## Implementation PR plan (ordered)

### PR 1 — Autotest parser fix (no production change)

- Update `analyze_test1_breakpoints.py` daily regex to accept `NA` in all six weather fields.
- Regenerate `AutoTest.1_Breakpoint_Diff_Report.txt`.
- Add test: known NA daily row (`2007-03-01`) must parse as boundary.

### PR 2 — `cli_validate.py` + post-write gate

- Implement rules 1–5 and NA-aware parser.
- Wire into `Autotest/AutoTest.sh`.
- Confirm Python AutoTest.1 passes with 0 violations.

### PR 3 — Pre-write simulation guard in `create_cli_file`

- Port loop simulation from `analyze_upstream_bps.simulate_cli_loop_bounds` into production validator.
- Assert `current_pcp` monotonic and in-bounds.
- Fail fast on overflow (`last_pcp >= len(pcp_df)`).

### PR 4 — R parity: ±1 break count

- Binary-search divergent days (start `2000-01-04`) comparing per-day `loop_bps` / `header_bps` / emitted lines between R and Python.
- Focus on `BTW` gap day bucketing and `rle` vs `Counter` aggregation.
- Add parameterized unit tests from extracted precip slices.

### PR 5 — Negative start times (optional policy)

- Document chosen policy (clamp vs carry).
- Implement in time emission; add Autotest assertions.

### PR 6 — R reference behavior (informational)

- R fails rule 1 on 113 days under NA-aware parser — likely the same `adjbrkpt` vs lines-written gap.
- Decide whether Python should **match R exactly** (including R’s inconsistencies) or **enforce stricter rules** than R.

---

## Decision matrix: match R vs enforce rules

| Strategy | Pros | Cons |
|----------|------|------|
| **Match R output** | Autotest MD5 parity | Perpetuates R’s declared≠actual days |
| **Enforce rules on Python** | CLIGEN-correct files | AutoTest.1 will not be byte-identical to R until R is fixed or test tolerances updated |
| **Hybrid** | Parity tests for values; separate structural validation | Two test surfaces |

**Recommendation:** Enforce rules 1–5 on Python output (Python already passes). Maintain a **structural** Autotest (declared=actual, monotonic depths) plus a **separate** R parity test with tolerances for ±1 break counts until PR 4 closes the gap.

---

## Verification commands (analysis only)

```bash
# Upstream comparison
python3 Development/scripts/analyze_upstream_bps.py

# Breakpoint diff (after PR 1 parser fix)
python3 Development/scripts/analyze_test1_breakpoints.py

# Quick Python self-consistency (NA-aware)
python3 -c "..."  # see investigation notebook in session; fold into PR 2
```

---

## Appendix: key code locations

| File | Functions | Responsibility |
|------|-----------|----------------|
| `Source/precip.py` | `process_precip_data` | `BPS = groupby(date).size()` |
| `Source/cli_output.py` | `generate_export_data` | Midnight adj, `+1`, gap `updated_bps`, `adjbrkpt` |
| `Source/cli_output.py` | `create_cli_file` | 1-based `current_pcp` walk, gap lines, formatting |
| `Source/workflow.py` | `core_workflow` | Calls export + CLI write |
| R `WEPPCLIFF.R` ~2317–2542 | same names | Reference implementation |

---

*Generated: 2026-06-25. Investigation only; no production source files were modified.*