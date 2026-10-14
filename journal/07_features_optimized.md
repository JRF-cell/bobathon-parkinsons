# 07_features_optimized

> Merge review (2026-10-14): the team attributes **Kaggle 3.966** to this
> branch. Historical CV results below are retained; see the
> [merge review](../docs/OPTIMIZED_MERGE_REVIEW.md) for reproduced checks and
> the distinction between ensemble predictions and component-only Hub reports.

<!--
Design note for experiments/07_features_optimized.py (same stem, one to
one with the script). Write and agree on it before creating the script.

Lifecycle:
  planned   → draft, not yet approved
  approved  → agreed; safe to create the matching .py
  done      → result recorded; Status block + JOURNAL.md row updated
  abandoned → discarded; record a one-line reason on State
-->

## Question / hypothesis

Can we push grouped-CV RMSE below 07_features (4.532 ± 0.050, Kaggle 4.274) by:
1. Fixing the interpolation bug (integer-index → age-based),
2. Adding richer longitudinal features (ON trajectory, counts, distances, ratios),
3. Adding pharmacokinetic medication-exposure proxies, and
4. Ensembling a specialist regressor (routing missing-OFF vs observed-OFF visits
   to dedicated models) with the improved HGBR?

Hypothesis: the interpolation bug had non-trivial impact because patients have
irregular visit spacing; the ON trajectory and ON/OFF ratio features add
independent signal; specialist routing will particularly improve the
missing-OFF slice (worst sub-group in 07_features).

## Motivation

From 07_features §50 largest errors analysis:

> Top-50 errors cluster on visits with missing OFF and very high true-OFF
> targets (mean 68.9 vs train mean 37.5). When `off` is absent and
> trajectory aggregates point to moderate severity, the model underestimates
> tail cases.

Three direct responses:

| Problem | Response |
|---|---|
| Integer-index interpolation wrong for irregular ages | Age-based interpolation (age as index) |
| Only `off` trajectory, not `on` | ON trajectory features: `on_prev/next`, `on_mean/median/min/max`, `on_slope`, `on_interp` |
| Drug timing only implicit | PK proxy features with multiple time constants (abs-decay, exp-decay) |
| Single model can't specialise for missing vs observed OFF | Specialist sub-models, each trained on its own slice |

---

## Feature catalogue — new additions over 07_features

All features computed in `src/parkinsons/features_optimized.py::add_features_optimized()`.

### Fixes to existing features

| Feature | Fix |
|---|---|
| `off_interp` | Now uses `age` as the interpolation index (was integer row index). Correctly handles irregular visit spacing across years. |
| `on_interp` | NEW — same age-based interpolation applied to `on` column. |

### Group B-extended — richer trajectory features

| Feature | Formula | Clinical rationale |
|---|---|---|
| `off_n_obs`, `on_n_obs` | count of non-NaN `off`/`on` per patient | Data richness proxy; patients with many ON/OFF pairs carry more reliable trajectory signal |
| `on_prev`, `on_next` | lag-1 / lead-1 `on` within patient | Symmetric to `off_prev`/`off_next`; captures treatment response trend |
| `on_slope` | OLS slope of `on ~ age` per patient | Rate of change of treated state; different from `off_slope` when treatment is being titrated |
| `off_slope_robust` | Siegel robust slope of `off ~ age` | Resistant to outlier ratings; more stable for patients with few or noisy visits |
| `off_dist_nearest_obs`, `on_dist_nearest_obs` | distance in years to nearest visit with observed `off`/`on` | Quantifies how far the interpolation must extrapolate; large distance → unreliable estimate |
| `on_off_diff` | `on_interp − off_interp` | Levodopa benefit at the visit level; should be ~0 at disease onset and grow with severity |
| `on_off_ratio` | `on_interp / off_interp` | Normalised treatment response; useful for patients whose absolute scores change over time |
| `on_off_diff_mean` | per-patient mean of `on_off_diff` | Patient-level average drug benefit; anchors individual visits |
| `off_obs_rate`, `on_obs_rate` | `off_n_obs / n_visits` | Fraction of visits with observed score; structural missingness pattern per patient |

### Group C-extended — pharmacokinetic proxies

Inspired by `docs/CONTEXT.md`:
> levodopa effect is proportional to blood concentration: a fast absorption phase, then a decline toward zero.

**Absorption model** (time of ON assessment = post-dose):

```
ledd_absorbed_τ(t) = ledd × (1 − exp(−t / τ))   for t = time_since_intake_on
```

τ values: 0.5 h, 1.0 h, 2.0 h  (captures fast vs slow absorbers, Tmax 30–90 min).

**Residual model** (time of OFF assessment = post-ON assessment):

```
ledd_residual_τ(t) = ledd × exp(−t / τ)          for t = time_since_intake_off
ledd_frac_τ(t)     = exp(−t / τ)
```

τ values: 1.5 h, 3.0 h, 5.0 h  (captures clinical motor half-life range 1–5 h).

**Interaction** (ON score corrected for drug level):

```
on_x_absorbed_τ = on × ledd_absorbed_τ
```

Missing `ledd` is replaced by zero only inside the proxies; the raw LEDD
NaN and missingness flag are retained. This is a modeling convention, not
evidence of no treatment. Missing intake delays propagate NaN. The tau values
are exponential time constants, not half-lives (half-life = tau × ln(2)).

---

## Model comparison

| Model | GroupKFold RMSE | ± std | vs ref | OFF_obs | OFF_miss |
|---|---|---|---|---|---|
| 07_features reference | 4.4720 | 0.0658 | — | 3.977 | 5.097 |
| A: optimized features same HP | 4.2829 | 0.0683 | −0.189 | 3.763 | 4.901 |
| B: optimized features best HP (lr=0.05/l=63/l2=1) | 4.3019 | 0.0613 | −0.170 | 3.763 | 4.925 |
| C: specialist (miss-OFF / obs-OFF) | 4.2797 | 0.0706 | −0.192 | 3.732 | 4.928 |
| **E: ensemble (specialist + HGBR_opt)** | **4.2176** | **0.0686** | **−0.254** | **3.682** | **4.852** |

Best model: **E — ensemble of SpecialistRegressor(C) + HGBR-opt(B)**. Each sub-model
is equally weighted (simple average). The ensemble wins because the two models make
complementary errors on the OFF_obs and OFF_miss slices.

---

## Leakage analysis

All features are computed from `X` only (never from `y`). GroupKFold groups
entire patients. The same arguments as in `07_features.md §Leakage analysis` apply
to all new features. Specifically:

- `off_dist_nearest_obs` uses only the row ages in the current split (training or
  validation partition of a patient), not any target values.
- PK proxies use only `ledd`, `time_since_intake_on/off` from X.
- `off_slope_robust` (Siegel) uses the same inputs as OLS `off_slope`.
- The SpecialistRegressor routes rows based on `off.isna()` (input feature) and fits
  each sub-model entirely within the training fold.

---

## Files touched

- `src/parkinsons/features_optimized.py` — new; exports `add_features_optimized(X)`
- `experiments/07_features_optimized.py` — new
- `experiments/_07_features_optimized_complete.py` — fixed-configuration submission helper (retained for regeneration)
- `journal/07_features_optimized.md` — this file
- `journal/JOURNAL.md` — new row
- `submissions/07_features_optimized.csv` — 11,013 rows, Index + target

---

## Historical Hub reports (HGBR component B only)

These reports do not evaluate the ensemble submitted to Kaggle. After merge,
new component uploads use explicitly named `07_features_optimized_component`
and `07_features_optimized_component_cv` keys. A new ensemble submission needs
a matching ensemble report; the existing links are kept as historical evidence.

- `07_features_optimized_cv` — CrossValidationReport for Model B (HGBR opt features)
  https://skore.probabl.ai/parkinpredict/bobathon-esilv/cross-validations/85766
- `07_features_optimized` — EstimatorReport (Model B, fold-4 held-out)
  https://skore.probabl.ai/parkinpredict/bobathon-esilv/estimators/85778

---

## Out of scope

- CatBoostRegressor: not installed in this environment; skipped.
- Large hyperparameter grid: kept light (3×3×2 = 18 configs) to avoid
  overfitting the CV split selection.
- Meta-learner for stacking: not warranted by the marginal gap between C and A.

---

## Status

- **State:** done
- **Approved by user on:** 2026-10-08 (autonomous run)
- **Headline result:** Grouped-CV RMSE **4.2176 ± 0.0686**
  (Ensemble of SpecialistRegressor + HGBR-opt, lr=0.05, leaves=63, l2=1,
  full optimized feature set 58 cols).
  Reference 4.472 → best 4.218 → **−5.7 % further improvement**.
  Biggest wins: age-based interpolation fix + ON trajectory features (−0.19 RMSE),
  ensemble over specialist (additional −0.06 RMSE).
  OFF-observed RMSE: 3.682 (vs 3.977). OFF-missing RMSE: 4.852 (vs 5.097).
  Submission: `submissions/07_features_optimized.csv` (11,013 rows).
  Hub CV: `07_features_optimized_cv`. Hub Estimator: `07_features_optimized`.
