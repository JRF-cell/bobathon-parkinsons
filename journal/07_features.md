# 07_features

> Audit clarification (2026-10-14): the historical results below need the
> qualifications in [DAY2_AUDIT.md](../docs/DAY2_AUDIT.md). `off_interp` uses
> numeric CSV index labels, not age. The tuning script uses 29 features; the
> continuation/submission script uses 32. Patient-disjoint test data permits
> using the visits supplied together for batch prediction; it does not prove
> that all lifetime visits are available, or justify prospective forecasting.

<!--
Design note for experiments/07_features.py (same stem, one to
one with the script). Write and agree on it before creating the script.

Lifecycle:
  planned   → draft, not yet approved
  approved  → agreed; safe to create the matching .py
  done      → result recorded; Status block + JOURNAL.md row updated
  abandoned → discarded; record a one-line reason on State
-->

## Question / hypothesis

Can domain-driven feature engineering — targeting the three key modelling
problems in `docs/CONTEXT.md` (temporal progression, drug timing, missing
data) — push grouped-CV RMSE below the 05/06 baseline of **7.52 ± 0.13**?

Hypothesis: trajectory features (`off_slope`, lag/lead visits) will give
the largest gain because the model currently sees each visit in isolation,
while per-patient temporal context captures the neurodegeneration trend
that produces the target.

## Motivation / clinical rationale

### Three modelling problems (from `docs/CONTEXT.md`)

| Problem | Why hard | What features target it |
|---|---|---|
| **Temporal progression** | Neurodegeneration is monotone ↑; without disease duration / visit rank the model treats all visits as i.i.d. | `disease_duration`, `years_since_first_visit`, `visit_rank`, `n_visits`, `off_prev`, `off_next`, `off_slope` |
| **Drug timing** | Assessed OFF score is biased by residual levodopa; time-since-intake interaction with measured score unbiases it | `ledd_missing`, `ledd` (raw), `time_since_intake_on/off` already in raw — kept; interactions deferred to tuning if needed |
| **Missing data** | 42 % of `off`, 30 % of `on`, 47 % of `time_since_intake_on`, 79 % of `time_since_intake_off`, 37 % of `ledd`, 32 % of `gene` are missing; HGBR handles NaN natively but missingness itself is informative | `off_missing`, `on_missing`, `ledd_missing`, `gene_missing`, `both_measured` |

---

## Feature catalogue

All features are derived from the **input matrix X only** — never from y.
They are computed by `add_features(X)` in `src/parkinsons/features.py` and
applied identically to train and test sets.

### Group A — Temporal features

| Feature | Formula | Clinical rationale |
|---|---|---|
| `disease_duration` | `age − age_at_diagnosis` | Direct surrogate of neurodegeneration stage; the further from diagnosis the greater expected motor deterioration. |
| `years_since_first_visit` | `age − min(age)` per patient | Captures within-study progression independent of diagnosis date (which is sometimes missing or approximate). |
| `visit_rank` | 0-based rank of visit within patient, sorted by age | Ordinal proxy for visit number; useful if the model learns that early vs late visits have different noise patterns. |
| `n_visits` | total number of visits per patient | A proxy for follow-up intensity / disease burden. Patients with many visits tend to be more severe or more studied. |

Per-patient aggregates are computed on the patient's **own** rows in X
(train rows for train patients, test rows for test patients). See leakage
analysis below.

### Group B — Patient trajectory features

These features embed the within-patient `off` trajectory into every row.

| Feature | Formula | Clinical rationale |
|---|---|---|
| `off_mean` | per-patient mean of `off` (non-NaN) | Represents the patient's average motor severity; anchors isolated visits. |
| `off_median` | per-patient median of `off` | Robust to outlier ratings. |
| `off_min` | per-patient min of `off` | Best-ever OFF state; baseline disease burden. |
| `off_max` | per-patient max of `off` | Worst-ever OFF state; disease ceiling. |
| `on_mean` | per-patient mean of `on` | Anchors the patient's treated state; correlated with OFF via patient-level trend. |
| `off_prev` | `off` of the **previous** visit (by age), NaN if first | Lag-1 feature for wear-off trend detection. |
| `off_next` | `off` of the **next** visit (by age), NaN if last | Lead-1 feature; only valid because all visits for test patients are in X_test. |
| `off_interp` | linear interpolation of `off` along age for each patient | Fills visit-level NaN in `off` using the patient's own trend; distinguishes "no OFF measured this visit" from "OFF is truly low". |
| `off_slope` | OLS slope of `off` ~ `age` per patient (NaN if < 2 non-NaN `off` values) | Direct quantification of motor decline rate; central to the temporal-progression problem. |

**Justification for `off_next`:** test patients have **all** their visits
in `X_test` — confirmed by `docs/CONTEXT.md` ("X_test patients do not
overlap train patients: the holdout is by patient"). Therefore computing
lead features from X_test uses only X_test rows of that patient, with no
access to future information from any target.

### Group C — Drug timing features

| Feature | Feature | Clinical rationale |
|---|---|---|
| `ledd` | (raw, already in X) | Total daily levodopa-equivalent dose; higher LEDD → greater dopamine replacement → lower measured ON score. Kept as-is; HGBR handles NaN natively. |
| `ledd_missing` | `ledd.isna().astype(int)` | Missing LEDD likely means the patient is early-stage or untreated — informative absence. |

*Note:* `time_since_intake_on` and `time_since_intake_off` are already raw
features in X and are passed to the model unchanged. Explicit interaction
terms (`on × f(time_since_intake_on)`) are deferred — the HGBR can learn
that interaction from the raw pair. They will be added only if trajectory
features alone do not give a sufficient gain.

### Group D — Missingness flags

| Feature | Formula | Clinical rationale |
|---|---|---|
| `off_missing` | `off.isna().astype(int)` | 42 % missing; most commonly absent when the patient did not do an OFF assessment (common in late-stage). |
| `on_missing` | `on.isna().astype(int)` | 30 % missing; missing if patient came in OFF state only. |
| `ledd_missing` | `ledd.isna().astype(int)` | 37 % missing; untreated or early-stage patients have no LEDD. Also in Group C. |
| `gene_missing` | `gene.isna().astype(int)` | 32 % missing; genetic test not performed. |
| `both_measured` | `(~off.isna() & ~on.isna()).astype(int)` | Both ON and OFF measured at the same visit — the richest data scenario, different from visits where only one is available. |

---

## Leakage analysis — per-patient features cannot leak across folds

**Claim:** per-patient aggregates (mean, slope, lag/lead) computed from X
cannot leak target information and cannot cross fold boundaries, given
GroupKFold on `patient_id`.

**Proof:**

1. **No target in the feature computation.** `add_features(X)` receives
   only the feature matrix. It never sees `y`. Trajectory features
   (`off_slope`, `off_mean`, etc.) are computed from the raw `off` column,
   which is an input feature — not the target `true_off`.

2. **GroupKFold groups entire patients.** When fold k is the validation set,
   every row from the corresponding patients is in the validation set, and
   no row from those patients appears in the training set for that fold.
   Because per-patient aggregates use only the patient's own rows, a
   validation patient's aggregate is computed purely from validation rows —
   it never sees training-patient rows or training-fold targets.

3. **Test patients have all visits in X_test.** `add_features(X_test)` sees
   all rows of every test patient at once. Lag/lead (prev/next visit) are
   therefore computed from the patient's own test visits — not from any
   training visit. This is the same computation as at CV time within a
   validation fold.

**Conclusion:** the per-patient features respect the GroupKFold boundary
by construction, and the CV score is comparable across experiments.

---

## Procedure

### Step 1 — Baseline

Reuse the best model from 04-06: `HistGradientBoostingRegressor(random_state=0)`
on raw features (`X_full` = all 11 columns minus `patient_id`) with
`skrub.TableVectorizer`.

Baseline from journal: **RMSE 7.521 ± 0.134** (05_tabular_pipeline).

### Step 2 — Incremental feature ablation

Add feature groups one at a time and compare grouped-CV RMSE:

| Config | Features added | Expected direction |
|---|---|---|
| `raw` | (baseline, raw X) | 7.52 |
| `+temporal` | Group A | ↓ likely |
| `+temporal+trajectory` | Group A + B | ↓ likely |
| `+temporal+trajectory+timing` | A + B + C | ↓ or neutral |
| `+all` | A + B + C + D | TBD |

A group is kept if the RMSE mean improves by more than ~1 fold-std (≈ 0.13)
**or** is neutral but clinically justified.

### Step 3 — Light hyperparameter tuning (HistGradientBoostingRegressor)

Grid (≤ 12 configs, same grouped CV):

| Parameter | Values |
|---|---|
| `learning_rate` | 0.03, 0.1 |
| `max_leaf_nodes` | 15, 31, 63 |
| `max_iter` | 1000 (fixed) |
| `early_stopping` | True (fixed) |
| `l2_regularization` | 0, 1 |

Total: 2 × 3 × 2 = 12 configs. Select the config with the lowest mean RMSE;
break ties by highest `l2_regularization` (simpler model).

### Step 4 — Hub push

```python
project.put("07_features", cv_report)            # CrossValidationReport
project.put("07_features_estimator", est_report)  # EstimatorReport (if needed)
```

Print the hub URL for Kaggle.

### Step 5 — Submission

Fit the final config on **all** training rows (`X_full` + engineered
features), predict `X_test`, write `submissions/07_features.csv`.

Format follows `data/sample_submission.csv`: `Index, target`.

---

## Files touched

- `src/parkinsons/features.py` — new; exports `add_features(X)`
- `experiments/07_features.py` — new
- `journal/07_features.md` — this file
- `journal/JOURNAL.md` — new row after run
- `submissions/07_features.csv` — new after run

---

## Out of scope

- Interaction terms `on × time_since_intake_on` (deferred; tree can learn it natively).
- `ledd_per_kg` (no weight column).
- External data sources.

---

## Workbook questions (DAY2 §7)

### Which single change gave the biggest gain?

**Trajectory features (Group B)** — adding `off_mean/median/min/max`,
`on_mean/median/min/max`, `off_prev`, `off_next`, `off_interp`, and
`off_slope` dropped RMSE from 7.42 (+temporal) to **4.64** — a gain of
**2.78 RMSE points**, which is 19× the fold-std of 0.14. This single group
accounts for ~93 % of the total improvement.

The mechanism: per-patient trajectory aggregates give the model direct access
to the patient's "typical" OFF level and their rate of motor decline, which
is the dominant signal in the `true_off` target.

### Which change gave none?

**Drug timing / LEDD missingness (Group C)** — adding `ledd_missing` moved
RMSE from 4.640 to 4.635 (delta −0.005), which is 0.06× the fold-std.
**Essentially zero gain.** HGBR already handles NaN in `ledd` natively and
implicitly learns the missingness pattern. An explicit flag is redundant
for tree-based models.

### 50 largest errors — what do they share?

| Feature | Top-50 | Full train | Comment |
|---|---|---|---|
| `off` missing | **58 %** | 42 % | 1.4× overrepresented |
| `gene` missing | **44 %** | 32 % | 1.4× overrepresented |
| `on` missing | 4 % | 30 % | `on` is *present* in top-50 |
| True target mean | **68.9** | 37.5 | Very high-severity visits |
| Disease duration mean | 6.35 y | 5.85 y | Slightly later-stage patients |
| Cohort A share | 82 % | 89 % | Slight shift toward cohort B |

**Summary:** Top-50 errors cluster on **visits with missing OFF and very
high true-OFF targets (mean 68.9 vs train mean 37.5)**. When `off` is absent
and trajectory aggregates point to moderate severity, the model underestimates
tail cases. Time-since-intake-off (14 h mean when available) is not the
primary driver. Rater R01 (7/50) is somewhat overrepresented but sample too
small to be conclusive.

### Which CONTEXT.md problem is still least addressed?

**Drug timing.** Raw `time_since_intake_on/off` are in X, but the
pharmacokinetic mechanism (fast absorption → exponential decay) is still
only implicit. For patients with long `time_since_intake_off` (> 12 h) the
model cannot distinguish "truly OFF" from "late residual drug" assessments.

**What to try next (Backlog B01):** Model levodopa blood level explicitly as
`ledd_effect = ledd × exp(−λ × time_since_intake_on)` with a data-driven
decay constant. Alternatively, bin `time_since_intake_off` into clinical
categories (< 2 h, 2–6 h, > 6 h, missing) as a categorical feature to
directly address the pharmacokinetic unbiasing problem in CONTEXT.md.

---

## Status

- **State:** done
- **Approved by user on:** 2026-10-07
- **Headline result:** Grouped-CV RMSE **4.5317 ± 0.0495**
  (lr=0.03, max_leaf_nodes=63, l2=1, all four feature groups, A+B+C+D).
  Baseline 7.521 → best 4.532 → **−39.8 %**. Submission: `submissions/07_features.csv`.
  Hub: `07_features` (CV report) + `07_features_estimator`.
  Biggest gain: trajectory features (Group B, −2.78 RMSE).
  Zero gain: drug timing flag (Group C, −0.005 RMSE).
