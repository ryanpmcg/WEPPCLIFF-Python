# New versus legacy gap-filling features

Comparison of the original R WEPPCLIFF / Python **legacy** path (`-gf legacy`, optional cluster means when `-pm` is omitted) with the Python **modern** stack: PyMICE FCS (`-pm`), storm templates (`-st`), and the physics-informed hierarchical engine PHIGI (`-gf phigi`).

| Capability | Legacy (R WEPPCLIFF / `-gf legacy`) | New (Python PyMICE + PHIGI) |
| --- | --- | --- |
| **Language / runtime** | R + CRAN `mice` | Python 3.10+ with `pymice-fcs` (PyMICE) |
| **Default gap-fill engine** | Staged multivariate MICE within calendar / half-month clusters | Same staged path available; optional **PHIGI** directed pipeline (`-gf phigi`) |
| **Imputation backend** | R `mice` (PMM / default methods) | PyMICE FCS (`midastouch`, `cart`, optional JOMO/PAN block via `-jm`) |
| **Cluster-mean fallback** | Used when MICE unavailable or as simple fill | Explicit cluster-wise mean when `-pm` is omitted; never the preferred production path |
| **Solar radiation** | Treated as a generic continuous MICE variable | **PHIGI L0/L3:** clear-sky \(R^{cs}\) from astronomy; impute cloud transmittance \(\tau\), then reconstruct SO_RAD |
| **Climate stratification** | Month / half-month / quarter-month calendar strata | Köppen–Geiger zone (`-kg`) + meteorological season + **weather-type mixtures** (PHIGI L1–L2) |
| **Regional priors** | Limited / ad hoc | Calibrated Köppen × season prior store; sparse-donor resilience |
| **Daily temperature order** | Soft / post-hoc corrections | Hard post-impute constraints (`MIN_TEMP ≤ MAX_TEMP`) and directed \(T\) stage in PHIGI L3 |
| **Precipitation model** | Joint continuous fill of daily PRECIP (and related fields) with other weather | **Hurdle model (PHIGI L4):** wet/dry occurrence → wet-day amount → BPS/ELEN; storm metrics only on wet days |
| **Sub-daily precipitation** | Row-wise / breakpoint FCS (spacing can collapse) | **Storm-template donor resampling** (`-st`) preserving inter-breakpoint timing for EI/I30 |
| **Nonstationarity** | Not supported | Year-block predictors (`-ns`) so multivariate relationships may drift across long records |
| **Missingness diagnosis** | Implicit (MICE on NA cells) | Explicit gap-kind profiles (short gaps, long gaps, variable-absent, seasonal) in validation harness |
| **Erosivity-aware validation** | EI computed after fill; not used as imputation objective | Hold-out scoring on monthly precip mass, wet-day occurrence, I30/EI relative bias (PHIGI L5 + fair compare) |
| **Imputation report** | Console / plots only | Structured `Imputation-Report.json` (`-ir` with PHIGI): layers used, strategy, confidence tier |
| **Quality control** | Physical limits + spike/streak/stick (R) | Ported QC stack; PHIGI adds \(\tau\) transmittance QC bounds (`-tauqc`) |
| **Evaluation philosophy (this paper)** | Often day-matched residual error for all variables | **Variable-appropriate metrics:** pointwise RMSE only for smooth continuous fields; precip/wind scored on occurrence, mass, and climatological distributions |
| **Opt-in flags** | `-id`, `-im`, `-io`, `-qi`, `-iv` | Adds `-pm`, `-st`, `-ns`, `-kg`, `-jm`, `-gf {legacy\|phigi}`, `-tauqc`, `-ir` |
| **Regression safety** | R reference for CLI / EI | Python Autotest + byte-level `.cli` parity tests; `-gf legacy` preserved for side-by-side comparison |

## Design intent

Legacy WEPPCLIFF proved that multivariate MICE can gap-fill weather for WEPP and USLE-family erosivity workflows [@McGehee:2020; @vanBuuren:2011]. The Python redesign keeps that capability while fixing structural failure modes:

1. **Do not invent physics** — clear-sky radiation and day length are computed, not drawn.
2. **Do not treat dry days as zero-storm rows in the same model as wet days** — hurdle precipitation separates occurrence from intensity.
3. **Do not destroy breakpoint spacing** when filling sub-daily precip — storm templates protect EI-relevant structure.
4. **Do not score stochastic weather like a continuous regression target** — precip and wind are judged on climate fidelity, not exact day identity.

## Flag cheat-sheet

| Goal | Recommended flags |
| --- | --- |
| Closest to original R behaviour (Python) | `-id` without `-pm` (cluster means) or `-id -pm -gf legacy` |
| Production multivariate FCS (staged) | `-id -pm` (optional `-st` for storms, `-qi` for speed) |
| Physics-informed hierarchical fill | `-id -pm -gf phigi` (optional `-ir`, `-kg AUTO`) |
| Fair hold-out comparison | `Development/scripts/compare_phigi_legacy.py` and multiseed ampute harness |
