# Development

Internal tooling, analysis scripts, and tests for the Python port. Not required for end-user runs of WEPPCLIFF.

Documentation (manual, plans) is in [`Documentation/`](../Documentation/). The tutorial lives in [`Tutorial/`](../Tutorial/).

| Path | Purpose |
|------|---------|
| `Commands/` | Double-click `.command` launchers (pytest, autotest, tutorial, analysis) |
| `reports/` | Generated analysis reports (breakpoint diffs, upstream BPS) |
| `scripts/` | One-off analysis and validation helpers |
| `scripts/run_experiment_matrix.py` | **Canonical** ampute–impute matrix A–G (dual whole/hole scoring) |
| `tests/` | `pytest` suite |
| `tests/test_experiment_matrix.py` | Unit + smoke for experiment matrix |

Package implementation: `Source/weppcliff_impute/experiment_matrix/`.  
Design doc: `Documentation/Planning/AmputeImputeExperimentMatrix.md`.

## Common commands

From the repository root:

```bash
.venv/bin/python -m pytest Development/tests/
bash Tutorial/AutoTutorial.sh
bash Autotest/AutoTest.sh

# Ampute–impute experiment matrix (A–G)
.venv/bin/python Development/scripts/run_experiment_matrix.py --list
.venv/bin/python Development/scripts/run_experiment_matrix.py --profile default --workers 4
.venv/bin/python -m pytest Development/tests/test_experiment_matrix.py -q
```