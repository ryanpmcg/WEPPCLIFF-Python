# Development

Internal tooling, analysis scripts, and tests for the Python port. Not required for end-user runs of WEPPCLIFF.

Documentation (manual, plans) is in [`Documentation/`](../Documentation/). The tutorial lives in [`Tutorial/`](../Tutorial/).

| Path | Purpose |
|------|---------|
| `Commands/` | Double-click `.command` launchers (pytest, autotest, tutorial, analysis) |
| `reports/` | Generated analysis reports (breakpoint diffs, upstream BPS) |
| `scripts/` | One-off analysis and validation helpers |
| `tests/` | `pytest` suite |

## Common commands

From the repository root:

```bash
.venv/bin/python -m pytest Development/tests/
bash Tutorial/AutoTutorial.sh
bash Autotest/AutoTest.sh
```