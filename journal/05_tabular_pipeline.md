# 05_tabular_pipeline

<!--
Design note for experiments/05_tabular_pipeline.py (same stem, one to
one with the script). Write and agree on it before creating the script.

Lifecycle:
  planned   → draft, not yet approved
  approved  → agreed; safe to create the matching .py
  done      → result recorded; Status block + JOURNAL.md row updated
  abandoned → discarded; record a one-line reason on State
-->

## Question / hypothesis

Does adding the categorical columns (`cohort`, `gene`, `rater_id`) to the
HGBR model lower grouped-CV RMSE beyond the fold-to-fold noise? In particular,
does `rater_id` carry signal — i.e., does knowing which clinician scored the
visit help predict the true (de-biased) OFF score?

## Motivation

Experiment 04 used only 8 numeric columns. Three columns were left out:

- `cohort` (2 values, no missing) — may encode different clinical protocols or
  measurement conventions between Cohort A and B.
- `gene` (4 values + NaN, 32 % missing) — genetic variant; missingness itself
  may be informative (e.g. wild-type patients are often not tested).
- `rater_id` (50 unique values, no missing) — the clinician who scored the
  visit. Per CONTEXT.md, one of the four biases the `target` removes is
  **human subjectivity in scoring** (bias #1). If rater-specific patterns
  survive in `on`/`off` after the true-OFF target corrects for them, `rater_id`
  should carry residual signal. The ablation (without `rater_id`) isolates that.

`skrub.tabular_pipeline("regressor")` handles mixed types automatically via
`TableVectorizer`, so no manual encoding is needed.

- **Sourcing strategy:** user (explicit request — DAY2.md §5)
- **Why this matters:** If categoricals help, future experiments should keep
  them. If `rater_id` helps, we have evidence of rater bias in the data.

## Column cardinality (from EDA / actual data)

| Column | Type | n_unique | Missing % | Notes |
|---|---|---|---|---|
| `cohort` | string | **2** | 0 % | A (88.9 %), B (11.1 %) |
| `gene` | string | **4** + NaN | 32.4 % | No Mutation, LRRK2+, GBA+, OTHER+ |
| `rater_id` | string | **50** | 0 % | R01–R50-style IDs, 50 distinct raters |

## Expected encoder choices from TableVectorizer

`TableVectorizer` picks encoders by cardinality:

| Column | Cardinality | Expected encoder | Rationale |
|---|---|---|---|
| `cohort` | 2 | **OneHotEncoder** | Low cardinality — one binary column |
| `sexM` | 2 (numeric int) | pass-through (numeric) | Already 0/1 integer |
| `gene` | 4 (+NaN) | **OneHotEncoder** | Low cardinality, NaN treated as category |
| `rater_id` | 50 | **GapEncoder / StringEncoder** | High cardinality — skrub compresses into a few numeric dimensions |

*Actual encoder choices will be recorded after the run.*

## Method

- **Files touched:**
  - `experiments/05_tabular_pipeline.py` (new)
  - `journal/05_tabular_pipeline.md` (this file)
  - `journal/JOURNAL.md` (new row after run)

- **Features:**
  - Drop `Index` and `patient_id` (identifiers — test patients are unseen, memorising IDs is leakage).
  - Keep everything else: `cohort`, `sexM`, `gene`, `age_at_diagnosis`, `age`, `ledd`,
    `time_since_intake_on`, `time_since_intake_off`, `rater_id`, `on`, `off`.
  - No manual imputation — `tabular_pipeline` and HGBR handle NaN natively.

- **Main model:**
  ```python
  from skrub import tabular_pipeline
  model = tabular_pipeline("regressor")
  ```

- **Ablation (no rater_id):**
  ```python
  X_no_rater = X_full.drop(columns=["rater_id"])
  ablation_model = tabular_pipeline("regressor")
  ```
  Same `cv_splits`. Isolates the contribution of rater identity.

- **Cross-validation:** same `GroupKFold(n_splits=5)` on `patient_id` as 04.

- **Comparison report:**
  ```python
  evaluate(
      {"hgbr_numeric": hgbr_04, "tabular": model},
      X_full, y, splitter=cv_splits,
  )
  ```
  `hgbr_04` = bare `HistGradientBoostingRegressor(random_state=0)` on the
  8 numeric columns (same as 04, re-evaluated here for a fair same-data
  reference point).

- **Hub report:** `project.put("05_tabular_pipeline", comparison_report)`

- **No Kaggle submission** in this experiment — submission comes in §7 once the
  best model across 04/05/06 is chosen.

- **Out of scope:** skrub DataOps, temporal features, patient-level aggregates.

## Risks / things that could invalidate the result

- `rater_id` has 50 levels. GapEncoder compresses them; if the compression
  loses the per-rater signal the ablation may show no difference even if
  rater matters.
- `gene` is 32 % missing; if missingness is informative the NaN-as-category
  path inside skrub is important — verify it is triggered.
- `tabular_pipeline` wraps HGBR by default; if the internal hyperparameters
  differ from the bare HGBR used in 04, the comparison mixes model and
  feature effects.

## Workbook questions (DAY2 §5)
*Filled in after the run.*

**Q1. Did adding `cohort`, `gene`, `rater_id` lower RMSE beyond fold-to-fold std?**
> **Yes — by a clear margin.**
>
> | Model | RMSE mean | RMSE std |
> |---|---|---|
> | 04 numeric HGBR | 7.7520 | ±0.155 |
> | 05 tabular (all cols) | **7.5210** | ±0.134 |
> | 05 tabular (no rater) | 7.7456 | ±0.146 |
>
> Delta (numeric → tabular): **−0.231** — nearly **2× the fold-std of 0.134**.
> `tabular_full` wins on **5/5 folds**, confirming the improvement is real and
> consistent, not a lucky fold. Adding the categorical columns reduced RMSE by
> a further **3.0 %** on top of the numeric-only HGBR.

**Q2. Which encoder did TableVectorizer pick for each string column? Does it match cardinality?**
> Observed (cardinality_threshold = 40):
>
> | Column | n_unique | Kind | Encoder | Matches expectation? |
> |---|---|---|---|---|
> | `cohort` | 2 | low_cardinality | **ToCategorical** | Yes — low cardinality, HGBR reads pandas category dtype natively |
> | `gene` | 4 (+NaN) | low_cardinality | **ToCategorical** | Yes — 4 < 40, NaN treated as category |
> | `rater_id` | 50 | high_cardinality | **StringEncoder** | Yes — 50 > 40, compressed into numeric dims |
>
> Note: `ToCategorical` is skrub's equivalent of one-hot for HGBR — it converts
> the column to `pandas.Categorical` dtype, letting HGBR use its native
> categorical-split logic rather than creating binary columns. The 8 numeric
> columns pass through unchanged. This matches the cardinality analysis exactly.

**Q3. Effect of `rater_id` — does the ablation (without it) change RMSE?**
> **Yes — removing `rater_id` almost entirely erases the categorical gain.**
>
> | Model | RMSE |
> |---|---|
> | tabular_full (with rater_id) | **7.521** |
> | tabular_no_rater | 7.746 |
> | numeric HGBR (no categoricals) | 7.752 |
>
> `rater_effect` = 7.746 − 7.521 = **+0.225** — nearly the entire delta vs
> numeric HGBR (0.231) comes from `rater_id` alone. `cohort` and `gene`
> together contribute only **0.006** RMSE.
>
> **Interpretation:** Despite the true-OFF target being designed to remove
> rater-subjectivity bias, the rater identifier still carries residual
> predictive signal. Two explanations are compatible: (1) the de-biasing of
> the synthetic data was not perfect — some rater-level variance survives in
> `target`; (2) raters may serve as a proxy for clinical site or protocol
> differences that are genuine (not bias). In either case, keeping `rater_id`
> in the feature set is strictly beneficial. Future experiments should include
> it.

## Status

- **State:** done
- **Approved by user on:** 2026-10-07
- **Headline result:** Grouped-CV RMSE **7.521 ± 0.134** (tabular_pipeline, all cols including rater_id,
  5-fold GroupKFold on patient_id). Beats numeric HGBR (7.752) by 0.231 (> 1 fold-std), 5/5 folds.
  Total reduction from dummy (16.48): **−54.4 %**.
  Hub push pending `.skore` auth. No Kaggle submission in this experiment.
- **Key finding:** `rater_id` alone accounts for 0.225 of the 0.231 RMSE gain.
  `cohort` + `gene` contribute only 0.006.
- **Implication for next iteration:** The best current model is `tabular_pipeline`
  with all columns (RMSE 7.521). This is the candidate for the §7 final submission.
  Before that, §6 DataOps could bake the grouped-split into the graph; or skip
  directly to §7 and fit on all training rows → `submissions/05_tabular_pipeline.csv`.
