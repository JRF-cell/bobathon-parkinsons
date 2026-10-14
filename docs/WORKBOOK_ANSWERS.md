# Bobathon workbook answers — Day 1 and Day 2

These answers follow every **Workbook questions** section in the original
[Day 1](DAY1.md) and [Day 2](DAY2.md) guides. They combine the team's historical
[journal](../journal/JOURNAL.md) with local verification on 2026-10-14.
Historical measurements are labeled where used. The [audit](DAY2_AUDIT.md)
explains validation and reproducibility limits.

The team supplied [Skore EstimatorReport 83726](https://skore.probabl.ai/parkinpredict/bobathon-esilv/estimators/83726).
It opens a sign-in page in the review environment, so its contents and model
identity have not been independently checked. A Skore validation metric must
not be relabeled as a Kaggle public leaderboard metric.

> Merge update: the **3.966 Kaggle public score** belongs to Yanice’s optimized
> branch. Its journal records ensemble CV **4.2176 ± 0.0686** (population fold
> standard deviation), and its output is `submissions/07_features_optimized.csv`.
> See the [merge review](OPTIMIZED_MERGE_REVIEW.md) for the preserved old models,
> the new model and limitations of its component-only Hub reports.

## Day 1 — §2: Explore the data

### Q1. How many visits/patients, and how many visits per patient?

| Split | Visits | Patients | Mean visits/patient | Median | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| Train | 44,590 | 5,576 | 8.00 | 7 | 4 | 12 |
| Test | 11,013 | 1,395 | 7.89 | 7 | 4 | 12 |

Train and test patient sets do not overlap. A row represents a visit, not an
independent patient; this motivates grouped validation.

### Q2. Which columns are most missing? Can ON and OFF both be missing?

The three most missing training features are `time_since_intake_off`
(**78.75%**), `time_since_intake_on` (**46.37%**) and `off` (**42.42%**).
`on` is missing for **29.65%** of visits. **No training visit has both ON and
OFF missing.** These rates describe the supplied training data.

### Q3. What is the target distribution?

`target` ranges from **0 to 109.5**, within the theoretical motor-score range
0–132. Its mean is **37.47**, median **37.3**, standard deviation **16.50**,
and quartiles **25.6 / 49.3**. Sample skewness is **0.051**, so it is nearly
symmetric, with only slight positive skew. There are no missing targets.

### Q4. Which features are associated with the target? Is OFF a copy?

Recomputed pairwise Pearson correlations are `off`: **0.871**, `on`:
**0.669**, `age`: **0.310**, and `ledd`: **0.298**. The observed OFF score is
strongly associated with the target but is not a copy: its mean absolute
error against the target, on visits where OFF is measured, is **7.45**.
ON is also informative but reflects the medicated state. The original task
explicitly supplies ON/OFF as inputs and asks us to recover the unbiased OFF
target; their correlation does not itself imply target leakage.

The initial EDA recorded slightly different correlations (0.886 / 0.688).
The figures above were recomputed on the current full training files with
pairwise non-missing rows and take precedence for this answer.

### Q5. Is LEDD missingness associated with cohort?

LEDD is missing in **35.90%** of cohort A visits versus **42.59%** of cohort B
visits, a gap of about **6.70 percentage points**. This is evidence that the
observed missingness rates differ by cohort; it does not establish the exact
missing-data mechanism. Differences in treatment or recording protocols
are possible explanations, not demonstrated clinical causes.

### Q6. Which columns should not be predictors?

Exclude `target` and anything computed from it. Exclude `Index`, which is an
identifier, and use `patient_id` only for splitting and patient-local
feature aggregation before dropping it. Retain `rater_id` as a categorical
clinical context feature: it identifies the scorer, not the patient.
The interpolation feature's indirect dependence on `Index` is a documented
limitation in the audit.

## Day 1 — §3: Dummy baseline

### Q1. What is the dummy RMSE, and why is it close to target standard deviation?

The reproduced 80/20 holdout RMSE is **16.4779**, versus target standard
deviation **16.4997**. A constant mean predictor leaves approximately the
variance of the target unexplained; taking its square root gives approximately
the standard deviation. Small differences come from the held-out sample and
the use of the training mean rather than the full-data mean.

### Q2. What does its prediction-error plot look like?

The model produces one constant prediction for all visits. With actual target
on the x-axis and prediction on the y-axis, points form a horizontal band;
if axes are reversed, they form a vertical band. A perfect model would lie
on the diagonal `prediction = actual`. This description follows the model's
constant outputs; it does not claim a fresh inspection of the remote plot.

### Q3. What baseline must later models beat? How much improvement is convincing?

The visible Day 1 reference is **RMSE 16.48**. A useful improvement should be
consistent on patient-held-out folds and large relative to paired fold-to-fold
variation. There is no universal fixed margin or valid significance test
based solely on “one standard deviation.” Ridge's approximately 36% reduction
is substantial, but its row-holdout result must first be checked using grouped
CV. For fair grouped comparisons, the dummy reference is **16.4992 ± 0.3429**.

## Day 2 — §1: Linear model

### Q1. By what percentage does Ridge improve RMSE?

From 16.4779 to 10.4877: **36.35%** lower RMSE on the same row-holdout protocol.
That is a large gain, indicating that the eight numeric inputs contain useful
signal, though this split does not measure generalization to unseen patients.

### Q2. Which coefficient is largest? Are ON/OFF signs sensible?

For alpha 0.01, fitted on all training rows, `time_since_intake_on` has the
largest absolute coefficient (**+1.704**). `off` is **+0.590** and `on` is
**+0.478**. Positive signs agree with the expectation that higher observed
motor impairment is associated with higher underlying OFF impairment.
These are conditional associations in a fitted model, not causal effects.

### Q3. Why are raw coefficients not comparable feature importances?

Inputs have different units and scales, and the model does not standardize
them. A one-unit change in age is different from a one-unit change in LEDD.
Their numerical coefficients cannot be ranked as equivalent clinical effects;
correlation between features also affects coefficient interpretation.

## Day 2 — §2: Ridge tuning

### Q1. How much does RMSE move across alpha values?

For alpha `[0.01, 0.1, 1, 10, 100, 1000]`, reproduced RMSE ranges from
**10.487659 to 10.488093**, a spread of about **0.000434**. Alpha 0.01 is the
minimum, but the practical difference is negligible. With many rows and few
features, these penalties have little effect on predictions in this setup;
feature scaling also determines the effective strength of the penalty.

### Q2. Do coefficients change much from the smallest to largest alpha?

Full-data fits give the following comparison:

| Feature | Alpha 0.01 | Alpha 1000 |
|---|---:|---:|
| time_since_intake_on | 1.7042 | 1.6535 |
| age | 1.0730 | 1.0710 |
| age_at_diagnosis | -0.9579 | -0.9556 |
| off | 0.5897 | 0.5902 |
| on | 0.4776 | 0.4764 |
| sexM | -0.1997 | -0.1818 |

Signs and dominant terms remain stable. Ridge need not shrink every individual
coefficient monotonically in the presence of correlated predictors.

## Day 2 — §3: Patient-grouped validation

### Q1. Which score changes more after switching to grouped CV?

| Model | Random row holdout | Grouped-CV mean | Change |
|---|---:|---:|---:|
| Dummy | 16.4779 | 16.4992 | +0.0213 |
| Ridge (alpha 0.01) | 10.4877 | 10.5965 | +0.1089 |

Ridge changes more. The row split shares **4,332 patients** across train and
validation; the grouped folds share none. Dummy ignores patient features, so
its score mostly tracks target spread. These small differences should not be
interpreted as a precisely isolated leakage effect because the validation
protocol and sampled rows also differ.

### Q2. Which estimate is closer to the Kaggle public score?

The team reports a **Kaggle public score of 3.966** (2026-10-14). The supplied
[Skore report](https://skore.probabl.ai/parkinpredict/bobathon-esilv/estimators/83726)
is linked separately; it requires sign-in in the review environment.
The team subsequently confirmed that **3.966 belongs to the optimized branch**,
whose completion script produces `07_features_optimized.csv`, rather than the
32-feature model. The optimized journal reports grouped CV **4.2176**, a gap
of **0.2516** above the public score. The submitted file’s exact bytes have not
been independently compared with a regenerated artifact.

No matching random-row evaluation of this optimized ensemble is recorded.
Comparing its Kaggle result against Ridge's 10.4877 row score would compare
different models, not isolate the validation strategy. Therefore the requested
same-model “which split is closer?” conclusion still cannot be established;
the available grouped-CV/public-score comparison is now explicitly recorded.

**Grouped CV remains the appropriate selection protocol** because the
competition holds out patients. A lower public score can reflect the held-out
sample and the final fit using more training data; its difference from local
CV does not by itself establish either leakage or a methodological error.

## Day 2 — §4: HistGradientBoosting

### Q1. Does HGBR beat Ridge on every fold?

Yes, **all five folds**. Local grouped RMSE is **7.7556 ± 0.1528** for numeric
HGBR versus **10.5965 ± 0.2670** for Ridge, a mean reduction of about **2.84**.
HGBR models nonlinear relationships and preserves informative numeric NaNs.

### Q2. How much slower is it, and is the gain worthwhile?

The historical journal measured approximately **0.573 s vs 0.051 s** for fit
and **0.022 s vs 0.004 s** for prediction per fold: about **11×** and **5×**
respectively. These are machine/run-specific historical timings, not new
benchmarks. At this dataset size, the absolute cost was small compared with
the consistent 2.84 RMSE improvement.

### Q3. Which features matter most, compared with Ridge coefficients?

Historical fold-0 permutation importance ranked **off, on, age,
age_at_diagnosis**, then intake timing and LEDD. Ridge's largest raw coefficient
was intake timing, but coefficient size depends on units. Permutation
importance measures deterioration in held-out RMSE on a common scale, so it
supports ON/OFF measurements as the dominant predictors. Correlated features
can substitute for one another, and a single fold is an incomplete summary.

## Day 2 — §5: Mixed-type tabular pipeline

### Q1. Do categoricals improve on numeric HGBR beyond fold spread?

The verified full pipeline scores **7.5336 ± 0.1372** versus numeric HGBR's
**7.7556 ± 0.1528**: about **0.222** lower mean RMSE, with improvements on
all five folds. However, removing `rater_id` gives **7.7559 ± 0.1454**.
Thus most of the gain is associated with **rater_id**, not with `gene` and
`cohort` alone. Exceeding one fold standard deviation is only a descriptive
comparison, not a formal significance test.

### Q2. What encoders were selected, and do they match cardinality?

For the installed `tabular_pipeline("regressor")`:

| Column | Non-missing distinct values | Encoder |
|---|---:|---|
| cohort | 2 | ToCategorical; HGBR native categorical splits |
| gene | 4 | ToCategorical; HGBR native categorical splits |
| rater_id | 50 | StringEncoder |

The cardinality threshold is 40, so these choices match the observed counts.
`ToCategorical` does not create one-hot columns. Numeric features pass
through; HGBR handles missing numeric/categorical values. Standalone
`TableVectorizer()` in later experiments instead uses one-hot encoding for
low-cardinality string columns.

## Day 2 — §6: DataOps

### Q1. Are its scores identical to experiment 05? Why or why not?

No. Local verification gave **7.5368 ± 0.1397** for DataOps and **7.5336 ±
0.1372** for 05. DataOps uses standalone `TableVectorizer()` with one-hot
encoding; the 05 factory uses native categorical conversion. StringEncoder
is unseeded, and the factory's HGBR does not explicitly set its random seed.
The pipelines therefore are not exactly equivalent despite using the same
raw columns and grouped folds. The small observed gap is not evidence that
DataOps itself improves or harms prediction.

*Update:* `06_dataops.py` now applies the same `tabular_pipeline("regressor")`
inside the graph, so the encoder difference is gone; only the unseeded
StringEncoder/HGBR randomness can still separate the two scores.

### Q2. What columns are in X_op? Are identifiers absent?

The 11 columns are `cohort`, `sexM`, `gene`, `age_at_diagnosis`, `age`, `ledd`,
`time_since_intake_on`, `time_since_intake_off`, `rater_id`, `on`, and `off`.
`target` and `patient_id` are explicitly dropped; `Index` is a DataFrame index,
not a predictor. `patient_id` is captured separately for the grouped split.
If either identifier appeared as a predictor in feature importance, we would
investigate the feature construction. An identifier does not automatically
leak labels under grouped CV, but it encourages reliance on arbitrary
identity patterns rather than useful clinical signal.

## Day 2 — §7: Wrap-up

### Q1. Which change helped most? Which did not?

In the **historical 07 ablation**, adding patient trajectory features reduced
RMSE from about **7.42 to 4.64** (about **2.78**), the largest feature-group
gain. The `ledd_missing` flag changed 4.640 to 4.635 (about **0.005**), offering
little measurable benefit there. The earlier model change from grouped Ridge
to numeric HGBR gave a similarly large, slightly larger numerical reduction
of about **2.84**. These stages use different baselines, so neither should be
claimed to explain all improvement by itself. Native missingness versus
imputation made a smaller difference (~0.04 RMSE).

### Q2. What do the largest errors have in common?

The following is the retained pre-optimized analysis, not an error analysis of
Yanice’s newer ensemble.

For the **32-feature submission variant**, the audited top 50 absolute OOF
errors range from about **18.9 to 31.1**. Compared with the full training set:

| Characteristic | Top 50 errors | Full training set |
|---|---:|---:|
| OFF missing | 64% | 42.42% |
| ON missing | 4% | 29.65% |
| Gene missing | 44% | 32.37% |
| Mean target | 66.43 | 37.47 |
| Mean disease duration (available values) | 6.90 | 5.85 |
| Cohort B share | 18% | 11.1% |

The errors concentrate on high-severity visits and missing OFF measurements.
Among the 16 top-error visits with observed OFF intake delay, mean delay is
13.42 in the supplied feature units. This alone does not establish timing as
the cause. Rater R01 appears six times; that small count is insufficient for
a strong rater-specific conclusion. These current results differ from the
historical 29-feature error summary because the model and run differ.

### Q3. What modeling problem is least addressed, and what could come next?

Explicit **drug timing** is least addressed. Raw dose and intake delays are
included, but no pharmacokinetic mechanism is modeled directly. A next
experiment could test dose–delay interactions or a dose × decay feature,
after checking the units and assumptions. Any decay parameter must be chosen
using training folds, followed by the same patient-grouped validation. This
is a proposed experiment, not a validated improvement or a clinical claim.

### Follow-up after merging the optimized branch

The “next experiment” above was implemented: `features_optimized.py` adds
absorption/decay proxies, ON interpolation and trajectories, observation counts,
distances and robust slopes. The new ensemble’s recorded mean RMSE is 4.2176.
These changes were introduced together, so their individual contributions
cannot be inferred from the combined improvement alone. The constants are
heuristic time scales, and zero-filling LEDD inside the proxies is a modeling
convention rather than proof that a patient was untreated. The earlier
workbook answers remain the record of what was known at each exercise stage.
