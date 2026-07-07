# WEPPCLIFF (Python)

Python port of [WEPPCLIFF](https://github.com/ryanpmcg/WEPPCLIFF) — a command-line tool for preparing climate inputs for WEPP, quality checking, gap filling, and erosion-index calculations (USLE-family models). The original R software was published in the Journal of Open Source Software [![DOI](https://joss.theoj.org/papers/10.21105/joss.02029/status.svg)](https://doi.org/10.21105/joss.02029).

## Documentation

User and developer documentation lives in [`Documentation/`](Documentation/):

| Document | Description |
|----------|-------------|
| [Manual.md](Documentation/Manual.md) | CLI flags, arguments, and usage |
| [Tutorial/](Tutorial/) | Step-by-step tutorial and `AutoTutorial.sh` |
| [Planning/](Documentation/Planning/) | Implementation plans and agent guide |

## Install

From the repository root:

```bash
bash Installer/Install.sh
```

Or manually:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt   # pulls PyMICE from github.com/ryanpmcg/PyMICE
pip install -e .
```

Requires Python 3.10+.

## Quick start

```bash
.venv/bin/python weppcliff.py -f ASOS_BPT_KMQE.csv -la y
```

## Tutorial, autotest, and tests

```bash
bash Tutorial/AutoTutorial.sh
bash Autotest/AutoTest.sh
.venv/bin/python -m pytest Development/tests/
```

Development tooling (scripts, reports, launchers) is described in [Development/README.md](Development/README.md).

## Repository layout

```
Source/                              # production code
Input/  Export/  Output/  Plots/   # runtime directories
Tutorial/                            # step-by-step tutorial and AutoTutorial.sh
Documentation/                       # manual and planning docs
Development/                         # tests, scripts, reports
Installer/  Autotest/
weppcliff.py                         # CLI entry point
```

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).