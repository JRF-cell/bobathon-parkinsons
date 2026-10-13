# 02_ridge

<!--
Design note for experiments/02_ridge.py (same stem, one to
one with the script). Write and agree on it before creating the
script.

Lifecycle:
  planned   → draft, not yet approved
  approved  → agreed; safe to create the matching .py
  done      → result recorded; Status block + JOURNAL.md row updated
  abandoned → discarded; record a one-line reason on State
-->

## Question / hypothesis

Can a Ridge regression using only the 8 numeric clinical columns
(`sexM`, `age_at_diagnosis`, `age`, `ledd`, `time_since_intake_on`,
`time_since_intake_off`, `on`, `off`) beat the dummy floor of RMSE 16.48?
Which alpha (regularisation strength) minimises RMSE on a random holdout?

## Motivation

The EDA showed that `off` (r = 0.886) and `on` (r = 0.688) are strongly
correlated with the target. A linear model should capture that additive signal
immediately. This experiment verifies that the numeric features carry real
information before adding encoding complexity for strings (`gene`, `cohort`).

- **Sourcing strategy:** user (explicit request — DAY2.md §1 & §2)
- **Why this matters:** Sets the first "real model" baseline. Also tells us
  whether the improvement over dummy is large enough to trust, and whether
  alpha matters at this scale.

## Method

- **Files touched:**
  - `experiments/02_ridge.py` (new)
  - `journal/02_ridge.md` (this file)
  - `journal/JOURNAL.md` (new row after run)

- **Features:** numeric clinical columns only —
  `sexM`, `age_at_diagnosis`, `age`, `ledd`, `time_since_intake_on`,
  `time_since_intake_off`, `on`, `off`.
  Never `Index`, `patient_id`, `rater_id`, `cohort`, `gene`.

- **Pipeline:**
  ```python
  make_pipeline(SimpleImputer(strategy="median"), Ridge(alpha=alpha))
  ```
  Median imputation fills the holes (up to 79 % missing for
  `time_since_intake_off`), then Ridge fits a weighted sum.

- **Alpha sweep:** `[0.01, 0.1, 1, 10, 100, 1000]`.
  Print a table alpha → RMSE / MAE / R².
  Keep the alpha with the lowest RMSE.

- **Cross-validation:** `splitter=0.2` — the same random 80/20 visit-level
  holdout as `01_dummy`. ⚠ **Leakage note:** the same patient may appear on
  both sides of the split, which flatters the score relative to a true
  patient holdout. This is intentional here so the result is directly
  comparable to 01_dummy. Grouped CV (GroupKFold on `patient_id`) will be
  introduced in experiment 03 and should be used for all comparisons from
  there on.

- **Hub report:** `project.put("02_ridge", <EstimatorReport of best ridge>)`.
  The console URL is copied into the Kaggle Submission Description.

- **Comparison print:** `evaluate({"dummy": dummy, "ridge": best_ridge}, X, y)`
  to show both models side-by-side.

- **Submission:** `clone(best_pipeline).fit(X, y)` → predict `X_test` on the
  same 8 feature columns → `submissions/02_ridge.csv` (columns `Index,target`).

- **Out of scope:** string columns (`gene`, `cohort`), feature scaling,
  grouped CV — those come in experiments 03–05.

## Workbook questions (DAY2 §1 & §2)
*Filled in after the run.*

### §1 — Linear model

**Q1. % RMSE reduction vs dummy (RMSE 16.48)?**
> Ridge RMSE = **10.49** vs dummy RMSE = **16.48**.
> Reduction = (16.48 − 10.49) / 16.48 = **36.3 %**.
> That is a substantial win: the model removes more than a third of the
> dummy error using only 8 numeric columns and a linear combination. The
> clinical features (`on`, `off`, timing, age) clearly carry real predictive
> signal.

**Q2. Largest absolute coefficient; sign of `on` and `off` — clinically expected?**
>
> Full coefficient table (sorted by |coef|, alpha = 0.01, fitted on all training visits):
>
> | Feature | Coefficient |
> |---|---:|
> | time_since_intake_on | +1.704 |
> | age | +1.073 |
> | age_at_diagnosis | −0.958 |
> | off | +0.590 |
> | on | +0.478 |
> | sexM | −0.200 |
> | time_since_intake_off | −0.155 |
> | ledd | −0.000 |
>
> **Largest absolute coefficient:** `time_since_intake_on` (+1.704).
> This means that for every extra hour since the ON-medication intake the model
> predicts a higher true OFF score — i.e. a more medication-worn-off state,
> which is clinically sensible.
>
> **Sign of `on` and `off`:** both are **positive**, meaning a higher measured
> motor score (worse measured symptoms) predicts a higher true OFF score (worse
> underlying disease). This is exactly what clinical reasoning predicts.
> It is somewhat surprising that `time_since_intake_on` outweighs `off` in
> raw coefficient magnitude — but that is partly a scaling artifact (see Q3).

**Q3. Why can't unscaled coefficient sizes be compared across features?**
> Ridge minimises a penalised sum of squared residuals. A unit change in `age`
> (range ~20–90) is not the same magnitude as a unit change in `on` (range
> ~0–100) or `ledd` (range ~0–2000+). A feature with a small coefficient may
> still dominate predictions simply because its values span a much wider range.
> To compare importances across features you must standardise first.

### §2 — Tweaking Ridge

**Q1. How much does RMSE move across alphas? Why might alpha matter little?**
> With 44,590 rows and only 8 features the problem is vastly over-determined:
> OLS is already well-conditioned. Alpha merely shrinks coefficients towards
> zero — it does not change the model's structural expressivity.
>
> Observed: RMSE range = **10.4877 – 10.4881** (spread = **0.0004**).
> Alpha moves RMSE by less than 0.0004 across four orders of magnitude — i.e.
> the choice of alpha is irrelevant here. This is expected: with 44 k rows and
> 8 features the design matrix is far from singular, so Ridge converges to OLS.
> Only at alpha = 1000 does shrinkage start to bite, and even then the effect
> is negligible.

**Q2. Did coefficients change much between smallest and largest alpha?**
> Functionally no. From alpha = 0.01 to alpha = 100 the coefficients are
> identical to four decimal places; at alpha = 1000 they begin to shrink very
> slightly (RMSE rises by 0.0004). The model is not sensitive to alpha in this
> data regime.

## Status

- **State:** done
- **Approved by user on:** 2026-10-07
- **Headline result:** RMSE 10.49, MAE 8.44, R² 0.595 (80/20 random visit holdout, alpha = 0.01)
  — submission CSV at `submissions/02_ridge.csv` (11,013 rows)
  — hub push pending `.skore` authentication
- **% improvement over dummy:** −36.3 % RMSE (16.48 → 10.49)
- **Implication for next iteration:** The linear model confirms that numeric
  features carry strong signal. The next step (03) introduces GroupKFold to
  remove patient-level leakage and get a score that matches the Kaggle split.
  After that, HistGradientBoosting (04) should further reduce RMSE by keeping
  NaN cells as signal (42 % of `off` rows, 30 % of `on`) rather than filling
  them with the median.
