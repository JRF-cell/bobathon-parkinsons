# 04_hgbr

<!--
Design note for experiments/04_hgbr.py (same stem, one to
one with the script). Write and agree on it before creating the
script.

Lifecycle:
  planned   → draft, not yet approved
  approved  → agreed; safe to create the matching .py
  done      → result recorded; Status block + JOURNAL.md row updated
  abandoned → discarded; record a one-line reason on State
-->

## Question / hypothesis

Does HistGradientBoostingRegressor (HGBR) beat Ridge on patient-grouped CV
by treating missing `on`/`off`/`ledd`/timing values as a learned signal rather
than filling them with the median?

Secondary: does imputing (ablation) hurt HGBR — i.e., is the missingness
itself predictive?

## Motivation

Ridge had to median-impute 42 % of `off` rows and 30 % of `on` rows, erasing
the information that "this visit had no OFF measurement" (often meaning an
ON-only visit — see CONTEXT.md). HGBR natively routes NaN values left or right
at each tree split, learning from the missingness pattern. This experiment
checks whether that is worth the extra compute.

- **Sourcing strategy:** user (explicit request — DAY2.md §4)
- **Why this matters:** If keeping holes beats filling them, every downstream
  model should drop `SimpleImputer` for these columns.

## Method

- **Files touched:**
  - `experiments/04_hgbr.py` (new)
  - `journal/04_hgbr.md` (this file)
  - `journal/JOURNAL.md` (new row after run)

- **Features:** same 8 numeric clinical columns as 02/03 —
  `sexM`, `age_at_diagnosis`, `age`, `ledd`, `time_since_intake_on`,
  `time_since_intake_off`, `on`, `off`.
  No `Index`, `patient_id`, `cohort`, `gene`.

- **Model:**
  ```python
  HistGradientBoostingRegressor(random_state=0)
  ```
  No `SimpleImputer` — NaN handling is native.

- **Ablation (same splits):**
  ```python
  make_pipeline(SimpleImputer(strategy="median"),
                HistGradientBoostingRegressor(random_state=0))
  ```
  Purpose: isolate the effect of keeping holes vs. filling them.

- **Cross-validation:** same `cv_splits` as 03 —
  `GroupKFold(n_splits=5)` on `patient_id`. Precomputed as:
  ```python
  cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=X_full["patient_id"]))
  ```
  This blocks entire patients into folds, matching the Kaggle holdout structure.

- **Comparison report:**
  ```python
  evaluate({"ridge": ridge_02, "hgbr": hgbr}, X, y, splitter=cv_splits)
  ```
  `ridge_02` = `make_pipeline(SimpleImputer("median"), Ridge(alpha=0.01))`,
  the winner from experiment 02.

- **Hub report:**
  - `project.put("04_hgbr", comparison_report)` — the full CV comparison.
  - If HGBR beats Ridge by more than 1 fold-std: also produce an
    `EstimatorReport` from one held-out fold for a valid Kaggle hub URL
    (`put("04_hgbr_estimator", estimator_report)`).

- **Submission:**  only written if HGBR grouped-CV RMSE < Ridge grouped-CV RMSE
  − 1 fold-std. Final model fit on all training visits → `submissions/04_hgbr.csv`.

- **Permutation importance:** via the skore report if the API exposes it;
  otherwise computed manually with `sklearn.inspection.permutation_importance`
  on the held-out rows of one fold.

- **Out of scope:** string columns (`gene`, `cohort`) — those come in 05.

## Risks / things that could invalidate the result

- The 8 numeric columns may not have enough spread in missingness patterns
  across folds for HGBR to learn from NaNs. Result could be neutral.
- Permutation importance with CV has high variance; interpret fold-level
  means only.
- HGBR is ~10–50× slower to fit than Ridge. Worth it only if RMSE gain is
  larger than fold-level noise.

## Workbook questions (DAY2 §4)
*Filled in after the run.*

**Q1. Does HGBR beat Ridge on every fold, or only on average?**
> **HGBR wins on all 5 / 5 folds** — not just on average.
>
> | Fold | Ridge RMSE | HGBR RMSE | HGBR wins? |
> |---|---|---|---|
> | 0 | 10.663 | 7.831 | Yes |
> | 1 | 11.018 | 7.931 | Yes |
> | 2 | 10.446 | 7.581 | Yes |
> | 3 | 10.320 | 7.598 | Yes |
> | 4 | 10.535 | 7.818 | Yes |
>
> Mean RMSE: Ridge **10.60 ± 0.27**, HGBR **7.75 ± 0.15**.
> Delta = **2.84** (> 18× the HGBR fold-std of 0.15). This is not noise.
> The improvement is large, consistent, and present in every held-out patient group.

**Q2. Fit and predict time ratio vs Ridge. Is the gain worth it?**
> | | Fit time | Predict time |
> |---|---|---|
> | Ridge | 0.051 s | 0.004 s |
> | HGBR | 0.573 s | 0.022 s |
> | Ratio | ~11× slower fit | ~5× slower predict |
>
> HGBR takes ~11× longer to fit and ~5× longer to predict per fold.
> For a 44 k-row dataset that is still less than 1 second per fold —
> negligible in absolute terms. The RMSE gain of 2.84 (a **26.8 %** further
> reduction on top of Ridge's 36.3 % over dummy) is clearly worth it.
> Conclusion: **yes, the gain is worth the extra compute.**

**Q3. Top features by permutation importance vs Ridge coefficients. Different ranking?**
> Permutation importance (fold 0, scale-free):
>
> | Rank | Feature | Perm. importance |
> |---|---|---|
> | 1 | `off` | **7.58** |
> | 2 | `on` | 5.34 |
> | 3 | `age` | 2.19 |
> | 4 | `age_at_diagnosis` | 1.71 |
> | 5 | `time_since_intake_on` | 0.27 |
> | 6 | `ledd` | 0.06 |
> | 7 | `time_since_intake_off` | 0.004 |
> | 8 | `sexM` | 0.003 |
>
> Ridge coefficient ranking (by |coef|) had `time_since_intake_on` first (+1.70)
> — a scale artifact (hours, wide range). Permutation importance is scale-free:
> `off` jumps to #1, `on` to #2. This confirms the clinical intuition: the
> measured motor scores are the dominant signal, not the intake timing.
> Both rankings agree on `ledd`, `time_since_intake_off`, and `sexM` being weak.

**Q4. Imputation ablation result. Does imputing hurt HGBR?**
> **Yes, slightly.** HGBR with native NaN: RMSE **7.75**, with `SimpleImputer`:
> RMSE **7.79** — a cost of **+0.038** RMSE points (~0.5 %).
>
> The effect is small but directionally consistent with the hypothesis:
> filling missing `off` / `on` values with the median erases the "ON-only visit"
> signal. Keeping the holes is strictly better, and future experiments should
> keep `SimpleImputer` out of any HGBR-based pipeline.

## Status

- **State:** done
- **Approved by user on:** 2026-10-07
- **Headline result:** Grouped-CV RMSE **7.75 ± 0.15** (HGBR native NaN, 5-fold GroupKFold on patient_id)
  vs Ridge **10.60 ± 0.27** — delta **2.84**, wins on **5/5 folds**.
  Total RMSE reduction from dummy (16.48): **−53.0 %**.
  Submission CSV: `submissions/04_hgbr.csv` (11,013 rows) — hub push pending `.skore` auth.
  EstimatorReport for Kaggle URL: `04_hgbr_estimator` (fold 0) — also pending hub auth.
- **Implication for next iteration:** Adding categorical columns (`gene`, `cohort`) via
  `TableVectorizer` (experiment 05) is the next logical step. HGBR native-NaN should
  be used as the base estimator. The permutation importance shows `gene`/`cohort`
  are not in the current feature set — they may carry additional signal.
