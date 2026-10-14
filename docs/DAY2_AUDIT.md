# Day 2 audit and Kaggle handoff

Audited locally on **2026-10-14**, against the pulled code and local CSVs.
The official [README](https://github.com/probabl-ai/bobathon-esilv),
[Day 2 guide](https://github.com/probabl-ai/bobathon-esilv/blob/main/docs/DAY2.md),
and local `docs/` are the competition reference. Historical journal results
are preserved; the measurements below are separate verification runs.

## Team-reported competition result

The team reports a **Kaggle public score of 3.966** on 2026-10-14 and supplied
[Skore EstimatorReport 83726](https://skore.probabl.ai/parkinpredict/bobathon-esilv/estimators/83726).
The report opens a sign-in page in the review environment. Its model identity
and metrics were not independently inspected, and the exact Kaggle-uploaded
CSV was not specified. This result is distinct from the local scores below;
it is not evidence that the regenerated 32-feature CSV has already been scored.

## Recommendation

Use the **32-feature TableVectorizer + HistGradientBoostingRegressor** in
`experiments/07_features_submit.py`. Its independently checked five-fold
patient-grouped RMSE was **4.4807 ± 0.0653** (sample standard deviation across
folds). The journal's **4.5317 ± 0.0495** is a historical result, not the
measured score of this 32-feature submission pipeline.

The subsequent end-to-end run of the updated submission script measured
**4.4814 ± 0.0601**. These are two separate runs of the same configuration,
consistent with its unseeded StringEncoder; quote approximately **4.48 RMSE**,
not an exact reproducibility guarantee. The saved CSV below came from the
end-to-end run, with Hub access deliberately disabled.

Parameters: `learning_rate=0.03`, `max_leaf_nodes=63`, `max_iter=1000`,
`early_stopping=True`, `l2_regularization=1`, `random_state=0`.
No model parameters or feature calculations were changed by this audit.

**CSV:** `submissions/07_features_32.csv`. The separate key prevents the
32-feature continuation from overwriting the 29-feature tuning output,
`submissions/07_features.csv`. The submission script now publishes its
EstimatorReport under the matching key `07_features_32`.

From the repository root, have **Bob** run:

```bash
./.venv/bin/python experiments/07_features_submit.py
```

This fits a held-out-fold EstimatorReport, attempts a Hub upload, fits the
final model, writes the CSV, and recomputes all five OOF folds for error
analysis and the printed validation score. It does not repeat the tuning grid.
Run without Python's `-O` flag so submission assertions remain enabled.

**Technical CSV validity and competition eligibility are separate.** The local
audit can verify the CSV, but an offline run cannot supply a valid Hub URL.
The official instructions require code and generation in Bob and the actual
Skore **EstimatorReport URL** in Kaggle's Submission Description. Regenerate
and have Bob review the small submission edits there before submitting; this
Codex audit is not evidence of Bob provenance. If the team has already used
`07_features_32` for a Kaggle upload, choose a fresh `REPORT_KEY` before the
next upload; the script derives the filename from that same key.

## Project map

| Location | Responsibility |
|---|---|
| `README.md`, `docs/SETUP.md`, `DAY1.md`, `DAY2.md`, `CONTEXT.md` | Rules, workflow, patient holdout, clinical/synthetic-data context |
| `data/` | Local raw competition tables; never modified during this audit |
| `experiments/` | Executable experiments; importing these files also executes training/Hub operations |
| `journal/` | Historical hypotheses, methods and recorded results; not proof that an artifact exists locally or was uploaded |
| `src/parkinsons/data.py` | Load training features and join targets by `Index`; load raw test rows |
| `src/parkinsons/features.py` | Stateless, patient-local feature calculations |
| `src/parkinsons/evaluate.py`, `pipeline.py` | Day 1 row-holdout setting and DummyRegressor only; later scripts construct their own models/splits |
| `src/parkinsons/hub.py` | Load credentials into environment variables; not a modeling component |
| `submissions/` | Ignored by Git; pulling teammates' code does not download their generated CSVs |
| `tests/` | Setup checks and a Day 1 prediction-shape smoke test; no existing comprehensive Day 2 tests |

There is no `03` experiment file. Grouped Ridge is implemented in `04_hgbr.py`.
Only `01_dummy.csv` existed in the checkout at audit start.

## Experiments and results

Every score below is RMSE in target units; **lower is better**.
Grouped scores are mean ± sample standard deviation of five fold RMSEs,
not a confidence interval or a single pooled OOF RMSE.

| Experiment / variant | Historical journal RMSE | Local verification RMSE | Validation |
|---|---:|---:|---|
| 01 dummy mean | 16.48 | 16.4779 | Random 80/20 rows |
| 02 Ridge, alpha 0.01 | 10.49 | 10.4877 | Random 80/20 rows |
| Ridge reference in 04 | 10.60 ± 0.27 | 10.5965 ± 0.2670 | GroupKFold(5), patient_id |
| 04 native-NaN HGBR | 7.75 ± 0.15 | 7.7556 ± 0.1528 | Same grouped folds |
| 04 median-imputed HGBR | 7.79 | 7.7951 ± 0.1681 | Same grouped folds |
| 05 tabular, all raw features | 7.5210 ± 0.1336 | 7.5336 ± 0.1372 | Same grouped folds |
| 05 without rater_id | 7.7456 ± 0.146 | 7.7559 ± 0.1454 | Same grouped folds |
| 06 DataOps | 7.5370 ± 0.1381 | 7.5368 ± 0.1397 | GroupKFold embedded in graph |
| 07 tuned, 29 features | 4.5317 ± 0.0495 recorded for 07 | 4.5329 ± 0.0674 | Same grouped folds, fixed recorded parameters |
| 07 submission, 32 features | Previously printed the same hard-coded 07 score | **4.4807 ± 0.0653** | Same grouped folds, fixed recorded parameters |

These checks used Skore evaluation; the 06 check evaluated the actual DataOp
graph. The entire 12-configuration tuning search was not repeated.

### 02_ridge.py

Uses `SimpleImputer(strategy="median") -> Ridge(alpha=...)`, with no scaling.
The eight numeric inputs are `sexM`, `age_at_diagnosis`, `age`, `ledd`,
`time_since_intake_on`, `time_since_intake_off`, `on`, and `off`.
Categoricals (`cohort`, `gene`, `rater_id`) are excluded. Imputation is fitted
inside the pipeline on each training split, so validation medians do not leak.

It tries alpha 0.01, 0.1, 1, 10, 100, 1000, selects lowest holdout RMSE,
compares against dummy, inspects full-data coefficients, and generates
`submissions/02_ridge.csv`. Local RMSE ranged only from 10.487659 to 10.488093.
Unscaled Ridge is consistent with the exercise, but its penalties and raw
coefficient magnitudes depend on feature units. Do not interpret coefficients
as comparable feature importance or causal effects.

Skore 0.26.0 uses seed 0 for this row holdout. **4,332 patients occur on both
sides**: this score is unsuitable for ranking models for unseen patients.

### 04_hgbr.py

Uses default `HistGradientBoostingRegressor(random_state=0)` on the same eight
numeric columns. No scaling, categorical encoding, or imputation in the main
model: trees route numeric NaNs natively. It compares against imputed Ridge
and against median-imputed HGBR using identical precomputed grouped folds.
It also calculates fold-0 permutation importance with negative RMSE scoring.

The large gain over Ridge appears on all five folds. The much smaller native
NaN/imputation gap is descriptive evidence, not proof that imputation always
hurts. The script writes `04_hgbr.csv` only if its improvement over Ridge
exceeds one HGBR fold standard deviation. That threshold is a heuristic, not
a formal significance test. Hub behavior was not verified in this audit.

### 05_tabular_pipeline.py

Uses `skrub.tabular_pipeline("regressor")`: TableVectorizer followed by HGBR.
Its 11 raw input columns are the eight numeric columns plus `cohort`, `gene`,
and `rater_id`. `Index` is the DataFrame index; `patient_id` is dropped.

In installed skrub 0.11.0 the factory configures `ToCategorical` for the two
low-cardinality string columns (`cohort`, `gene`). HGBR handles their category
dtypes natively; this is **not one-hot encoding**, and missing categories
are handled as missing values. Numeric NaNs remain available to HGBR.
`rater_id` has 50 values and goes through `StringEncoder` (threshold 40).
Encoding is fitted inside each outer training fold.

It compares numeric HGBR, all columns, and all columns minus `rater_id`.
Most of the improvement comes from retaining raters. This does not prove
that the target is biased: rater identity can help correct bias in the
measured ON/OFF inputs. The script evaluates/reports only; **it writes no CSV**.

### 06_dataops.py

Expresses `TableVectorizer() -> HGBR(random_state=0)` as a skrub graph.
`patient_id` is captured as the grouping variable before dropping it and
`target` from features. `Index` remains an index rather than a predictor.
GroupKFold and its groups are attached to `mark_as_X`; `evaluate` reads them.

The same 11 raw features are used, but standalone `TableVectorizer()` uses
**OneHotEncoder** for low-cardinality strings, unlike the 05 factory. Missing
categorical values can be encoded as a category; numeric NaNs pass through.
High-cardinality raters still use StringEncoder. Consequently 05 and 06 are
not exactly the same pipeline. Tiny differences cannot be attributed solely
to floating-point noise. It fits a full-data learner and smoke-predicts test
rows, but **does not save a submission CSV**.

### 07_features.py and 07_features_submit.py

`07_features.py` evaluates incremental groups, then selects among 12 HGBR
parameter combinations using the same grouped folds. Ablations use the
tabular factory; tuning uses standalone `TableVectorizer()`, so that stage
also changes low-cardinality encoding to one-hot. The continuation uses the
same standalone vectorizer and recorded parameters, without searching again.

Both preserve numeric NaNs, encode string columns within each training fold,
and use these additional features:

| Group | Features |
|---|---|
| A: time | disease_duration; years_since_first_visit; visit_rank; n_visits |
| B: trajectory in tuning script | off_mean, off_median, off_min, off_max; on_mean; off_prev, off_next; off_interp; off_slope |
| B: extra in submission script | **on_median, on_min, on_max** |
| C: timing/missingness | ledd_missing; raw LEDD and intake-delay columns are already included |
| D: missingness | off_missing, on_missing, gene_missing, both_measured |

Thus tuning uses 11 + 18 = **29 input columns**, while submission uses
11 + 21 = **32** (before encoding). The journal describes the trajectory
family more broadly than the tuning script actually selects. Historical
ablation scores (about 7.42 temporal, 4.64 with trajectories, 4.63 all groups)
were not rerun; they must not be read as a controlled comparison of all 32
submission features. The local audit directly evaluated both final variants.

The 32-feature model's fold RMSEs were **4.4239, 4.5621, 4.4915, 4.4061,
4.5199**, lower on every fold than the 29-feature model in this run. The mean
gain is modest (~0.052); it is not a guarantee of a better Kaggle score.

## Methodology checks and limitations

- **Data alignment passes.** Training has 44,590 visits / 5,576 patients;
  test has 11,013 visits / 1,395 patients. Train/target IDs match uniquely;
  the left join preserves feature/target alignment. No target is missing or
  infinite. Train and test patient sets are disjoint, with no missing IDs.
- **Outer grouped split passes.** All five folds have zero overlapping
  patients. Validation sizes are 8,918, 8,921, 8,917, 8,917, 8,917. Group
  labels and feature rows stay aligned, and every row is held out once.
- **No target-derived features found.** `off` is an observed input, not
  `target`. Patient summaries never consume targets. `patient_id` groups
  visits but is removed before model fitting. `rater_id` identifies the
  clinician and is an allowed input, not a patient identifier.
- **Patient-local feature isolation passes.** For every fold, computing
  `add_features` on held-out patients alone produced exactly the same frame
  as selecting those rows from features computed on all training patients.
  Thus computing these particular aggregates before outer CV does not mix
  training and validation patients. Learned vectorization/imputation stays
  inside the model pipeline.
- **Batch inference assumption.** Patient summaries and `off_next` use
  later supplied visits of the same patient. This is appropriate to the
  supplied batch task, where all test inputs are available together, but
  not a prospective prediction made before later visits occur. Patient
  holdout alone does not prove that all lifetime visits are supplied.
- **Interpolation documentation bug.** `off_interp` uses
  `interpolate(method="index")` on the numeric CSV `Index`, not age. This
  indirectly makes an engineered feature depend on an identifier's spacing.
  It is not target leakage, but is not a clinically timed interpolation.
  Original behavior was retained to avoid changing the evaluated model.
  Correcting it should be a separate experiment with new grouped validation.
- **Internal stopping is not grouped.** HGBR reserves rows from each outer
  training fold for early stopping. Patients and their precomputed summaries
  can span that inner split; it is not an unseen-patient validation estimate.
  Outer validation patients remain isolated. Both audited tuned models
  reached all 1,000 iterations in every fold. Full-data `fit` similarly
  receives all rows but reserves an internal validation fraction.
- **Selection optimism.** Features and parameters were chosen using the same
  five folds on which the winning result is reported. The score is a model
  selection estimate, not an untouched test result; repeated comparisons can
  make it optimistic. Fold standard deviation is not statistical certainty.
- **Reproducibility is partial.** StringEncoder has `random_state=None`;
  the 05 factory's HGBR also lacks an explicit seed. Therefore exact reruns
  can differ even when the tuned HGBR itself uses seed 0. Runtime versions
  are not fully pinned by `requirements.txt`. This audit used Python 3.14.6,
  scikit-learn 1.9.1, skore 0.26.0, skrub 0.11.0, pandas 3.0.6, NumPy 2.5.3.

HGBR's missing-value and early-stopping behavior is documented in the
[scikit-learn API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).
Encoder defaults were also inspected in the installed libraries and fitted
pipeline. No full claim of “no validation issues” is warranted given the
specific qualifications above; the outer patient holdout itself is sound.

## Submission validation and remaining handoff

The submission generator preserves test order through `set_index("Index")`,
feature engineering, selecting the same ordered feature list, and assigning
predictions positionally alongside those IDs. It writes with `index=False`.
The audit checks the **saved CSV**, not just an in-memory prediction array:

- Exactly `Index,target`, in that order; no accidental index column.
- Exactly 11,013 data rows, matching both test and sample submission.
- Unique IDs, identical to sample/test IDs in the same order.
- All predictions numeric and finite; no NaN or infinity.
- Same raw feature schema and engineered column order at fit and predict.

**All saved-file checks passed** for `submissions/07_features_32.csv`.
Predictions range from -0.8020 to 92.9684 (mean 36.6444); outputs were not
clipped or otherwise postprocessed. File SHA-256:
`9aab37cb0d013fc6e1e837fa4f54d0ef15f85ea36de9b756df552adb5d6ab764`.
Regeneration may change this hash because the encoder is not seeded.

The updated continuation script asserts those conditions before saving.
Its printed five-fold score is now computed from its OOF predictions instead
of copied from the historical 29-feature score. It uses the actual report
URL printed by `project.put`; the previous manually assembled
`api.skore.probabl.ai/workspaces/...` URL was not an EstimatorReport link.

**Hub remains unverified.** Setup checks passed 6/7 locally; the network
authentication check failed because the sandbox could not resolve the Hub
hostname. Automatic approval review rejected a network-enabled retry because
it would transmit the stored API key for authentication. No credentials were
printed and no Hub report was pushed by this audit. The Day 1 smoke test
passed. Local model/CSV verification deliberately did not load credentials.

Before uploading, complete the Bob generation/review requirement and obtain
the actual matching EstimatorReport URL. A workspace/API URL, the Day 1 URL,
or the historical 29-feature report is not a substitute. No Kaggle upload or Hub report publication was performed during the model audit.
Git publication was handled separately after the team requested it.

## Changes made during this audit

| File | Change |
|---|---|
| `docs/DAY2_AUDIT.md` | This experiment walkthrough, evidence, limitations and submission handoff |
| `README.md` | Link to the audit |
| `journal/JOURNAL.md` | Distinguish historical results from the verified 32-feature variant |
| `journal/07_features.md` | Qualify feature counts, interpolation and batch-inference claims |
| `src/parkinsons/features.py` | Correct interpolation comments; explain batch patient context; no behavior change |
| `src/parkinsons/data.py` | Clarify train/test Index handling; no behavior change |
| `experiments/02_ridge.py` | Correct the nonexistent experiment-03 reference |
| `experiments/07_features.py` | Explain tuning/early-stopping limitations and misleading workspace URL; no behavior change |
| `experiments/07_features_submit.py` | Explain feature mismatch; separate CSV/report key; check template/order/finite predictions; print computed OOF score and use actual Hub report URL |
| `submissions/07_features_32.csv` | Generated and validated local audit artifact (Git-ignored) |

The submission script completed with credentials/Hub operations disabled,
the Day 1 smoke test passed, all experiment/package files parsed, and
`git diff --check` passed. SHA-256 checks confirmed all four raw CSVs unchanged.
