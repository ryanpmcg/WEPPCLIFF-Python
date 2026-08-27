# JOSS-style paper: WEPPCLIFF Python gap filling

Draft paper package (modelled on the original R WEPPCLIFF JOSS submission).

**Results freeze:** 2026-07-26 — multiseed N=30 (90/90 jobs ok). Brain hub: `Research/Projects/WEPPCLIFF.md` (vault).

| Path | Description |
| --- | --- |
| `paper.md` | Manuscript (Pandoc/JOSS front matter) |
| `paper.bib` | Bibliography |
| `figures/` | Climate-aware evaluation figures |
| `results/TABLE_multiseed_summary.md` | \(N=30\) mean±std table |
| `results/TABLE_feature_comparison.md` | New vs legacy feature matrix |
| `results/multiseed_*.csv` | Full replicate and aggregate CSVs |

## Regenerate results

From the repository root (requires AutoTest.1 daily export):

```bash
python Development/scripts/autotest1_ampute_impute_analysis.py
python Development/scripts/autotest1_ampute_impute_multiseed.py --n 30 --workers 4 --io 20
python Development/scripts/inject_joss_multiseed_table.py
```

### Experiment matrix (dual whole/hole scoring, A–G)

Canonical assessment beyond the 2026-07-26 freeze table:

```bash
python Development/scripts/run_experiment_matrix.py --profile paper --workers 4
# outputs → Autotest/Output/ExperimentMatrix/ and results/matrix_*.csv
```

Design and metric definitions: `Documentation/Planning/AmputeImputeExperimentMatrix.md`.

Single-seed diagnostic plots are written under `Autotest/Output/AmputeImpute/` and key figures are copied into `figures/`.
