<!--
Exploratory data analysis summary for this workspace, written from
the data/eda.py run. Ground every claim in what the run actually
showed - do not invent facts. Keep "Modelling implications" as
candidate suggestions to weigh when designing the model, not final
decisions.
-->

# EDA: Parkinson's UPDRS Prediction

_Generated from `data/eda.py` on 2026-10-07._

## Dataset at a glance

- **Tables:** 1 training table (X_train merged with y_train); 1 test table (X_test, no target)
- **Shape:** 44,590 rows × 12 columns (train); 11,013 rows × 11 columns (test)
- **Target:** `target` — continuous float (regression: UPDRS score prediction)
- **Rich report:** [eda_train.html](eda_train.html)

## Per-column findings

| Column | Dtype | Missing % | n_unique | Notes |
|---|---|---|---|---|
| `patient_id` | string | 0% | 5,576 | Patient identifier — many rows per patient |
| `cohort` | string | 0% | 2 | Two cohort groups (A, B) |
| `sexM` | int64 | 0% | 2 | Binary: 0 = female, 1 = male |
| `gene` | string | **32.4%** | 4 | Genetic variant — significant missingness |
| `age_at_diagnosis` | float64 | 5.2% | 563 | Small missingness |
| `age` | float64 | 0% | 1,150 | Age at assessment |
| `ledd` | float64 | **36.6%** | 1,320 | Levodopa equivalent daily dose — heavy missingness |
| `time_since_intake_on` | float64 | **46.4%** | 64 | Minutes since last "ON" dose — heavy missingness |
| `time_since_intake_off` | float64 | **78.8%** | 178 | Minutes since last "OFF" dose — very heavy missingness |
| `on` | float64 | **29.6%** | 80 | UPDRS motor score in ON state |
| `off` | float64 | **42.4%** | 93 | UPDRS motor score in OFF state — strong predictor |
| `target` | float64 | 0% | 867 | **Target**: no missing values |

**Key observations:**
- `time_since_intake_off` is missing 79% of the time — the OFF assessment is rarely recorded.
- `ledd`, `time_since_intake_on`, `on`, `off` all miss 30–46% of values, indicating that many visits have incomplete measurement sets.
- `patient_id` has 5,576 unique values across 44,590 rows → **multiple rows per patient** (longitudinal / repeated-measures structure).
- `cohort` and `sexM` are complete and low cardinality; useful as categorical features.

## Target

- **Type:** Continuous regression (UPDRS score)
- **Mean:** 37.5 | **Median:** 37.3 | **Std:** 16.5
- **IQR:** 25.6 – 49.3 | **Range:** 0 – 109.5
- **Distribution:** Roughly bell-shaped, slightly right-skewed (a few high-score outliers above 80). The histogram bins are: ~2.5k (0–11), ~5.6k (11–22), ~9.8k (22–33), ~10.7k (33–44, mode), ~8.8k (44–55), ~5.2k (55–66), ~1.8k (66–77), ~180 (77–88), very few above.
- **No missing target values in training.**

The target is continuous and roughly symmetric — **mean-absolute-error (MAE) or RMSE** are natural first metrics. The right tail (scores 80–109) is sparse (~21 rows) and may be harder to predict.

## Structure

- **Datetime columns:** None detected by skrub.
- **Group structure: patient_id repeats.** With 5,576 unique patients over 44,590 rows, each patient appears ~8 times on average. This is a **longitudinal / panel dataset**.
  - `unique_ratio` for `patient_id` = 0.125 — confirming the group structure.
  - Standard k-fold splitting would put rows from the same patient in both train and test folds → **data leakage**.

## Associations

**With target (Pearson correlation):**

| Feature | Cramér V | Pearson r |
|---|---|---|
| `off` | 0.495 | **0.886** ← strongest |
| `on` | 0.403 | **0.688** |
| `ledd` | 0.236 | 0.298 |
| `age` | 0.114 | 0.310 |
| `age_at_diagnosis` | 0.051 | 0.133 |
| `time_since_intake_on` | 0.177 | ~0 |
| `time_since_intake_off` | 0.092 | ~0 |
| `cohort` | 0.090 | n/a (categorical) |
| `sexM` | 0.023 | ~0 |
| `patient_id` | 0.028 | n/a |
| `gene` | 0.052 | n/a |

**⚠ Potential leakage flag — `off` and `on`:** `off` has Pearson r = 0.886 with the target, and `on` has r = 0.688. Given that the target is a UPDRS score and `on`/`off` are also UPDRS motor scores in ON/OFF states, these features may be derived from or very closely related to the target measurement. **Confirm with domain knowledge** whether using `on` and `off` at test time is valid, or whether they constitute a leakage risk.

**Strong inter-feature links:**
- `age_at_diagnosis` ↔ `age`: r = 0.942 (near-collinear — both measure the same underlying aging process).
- `on` ↔ `off`: r = 0.867 (expected: both measure motor impairment).
- `on` ↔ `off` ↔ `target` cluster forms the dominant signal in the dataset.

## Modelling implications

- **`patient_id` repeats → use `GroupKFold` on `patient_id`.** Rows from the same patient are not independent. Standard k-fold would leak patient-level information across folds, inflating validation scores. `GroupKFold` (or `GroupShuffleSplit`) ensures no patient appears in both train and validation.
- **Heavy missingness in `ledd`, `time_since_intake_off`, `on`, `off` → imputation required.** Tree-based models handle this with `missing_values=np.nan`; linear models need explicit imputation. skrub's `tabular_pipeline` handles this by default.
- **`off` and `on` are very strong predictors (r = 0.886, 0.688)** but may be partially missing (42% and 30%). Confirm they are available at prediction time. If not, the model must work without them — a critical design question.
- **`age` and `age_at_diagnosis` are collinear (r = 0.94)** — dropping one or using a regularized learner reduces redundancy.
- **Target is roughly Gaussian — RMSE or MAE are appropriate first metrics.** No strong skew requiring a log-transform, though the sparse right tail (scores > 80) may warrant monitoring.
- **Test set has no `target` column** — this is a held-out leaderboard split. Evaluation should use cross-validation on the training set with `GroupKFold`.

## Open questions

1. **Are `on` / `off` available at prediction time?** Their very high correlation with `target` makes them the dominant signal, but if they are not available at inference (i.e., they are measured alongside `target`), using them would be leakage.
2. **What does `target` represent exactly?** Likely a composite UPDRS score — confirming its definition is important before choosing a loss function.
3. **Is `ledd` available at prediction time?** 36.6% missing — understanding whether that's structural (not prescribed) or observational (not recorded) shapes how to impute.
4. **What distinguishes `cohort` A vs B?** This may reflect a clinical trial design difference that could justify a stratified model.
5. **Is `gene` missing at random, or is missingness informative** (e.g., wild-type patients not tested)?
