# AutoTest.1 multiseed ampute/impute (N=30)

Seeds 0–29; PyMICE midastouch; io=20; climate-aware scoring for PRECIP/wind.
Values are mean ± sample standard deviation across 30 independent seeds.

| Metric | mcar_ampute | short_gaps | long_gap_all |
| --- | --- | --- | --- |
| Pointwise RMSE (mean of 4 vars) | 47.182 ± 3.168 | 86.588 ± 8.499 | 84.928 ± 3.340 |
| MIN_TEMP RMSE (°C) | 4.423 ± 0.530 | 12.220 ± 1.393 | 11.162 ± 0.439 |
| MAX_TEMP RMSE (°C) | 4.956 ± 0.796 | 13.313 ± 1.533 | 12.187 ± 0.585 |
| DP_TEMP RMSE (°C) | 5.719 ± 0.615 | 13.604 ± 1.726 | 13.029 ± 0.516 |
| SO_RAD RMSE (Ly) | 173.630 ± 12.568 | 307.214 ± 32.000 | 303.333 ± 12.377 |
| PRECIP wet/dry accuracy | 0.985 ± 0.008 | 0.983 ± 0.033 | 0.985 ± 0.013 |
| PRECIP wet-day F1 | 0.992 ± 0.004 | 0.991 ± 0.017 | 0.992 ± 0.007 |
| PRECIP monthly mass bias | 0.049 ± 0.008 | 0.011 ± 0.006 | 0.020 ± 0.005 |
| PRECIP holdout mass ratio | 0.928 ± 0.045 | 0.571 ± 0.327 | 0.701 ± 0.122 |
| PRECIP wet-day intensity KS | 0.056 ± 0.014 | 0.335 ± 0.102 | 0.126 ± 0.040 |
| W_VEL speed KS | 0.095 ± 0.025 | 0.199 ± 0.062 | 0.110 ± 0.025 |
| W_VEL mean ratio | 0.975 ± 0.029 | 0.994 ± 0.086 | 1.013 ± 0.033 |
| W_VEL monthly mean bias | 0.006 ± 0.001 | 0.002 ± 0.001 | 0.002 ± 0.001 |
| W_DIR sector TVD | 0.111 ± 0.040 | 0.226 ± 0.082 | 0.102 ± 0.028 |
| W_DIR monthly circ. mean err (°) | 1.585 ± 0.239 | 0.309 ± 0.100 | 0.503 ± 0.102 |

Full replicates: `Autotest/Output/AmputeImpute/multiseed/replicates.csv`
