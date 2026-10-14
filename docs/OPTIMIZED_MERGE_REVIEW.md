# Optimized branch merge review

Reviewed on 2026-10-14. This merges Yanice’s `07-features-optimized-yanice`
commit `d1f61d3` into the documented `main` at `7ba0e77`. The original
experiments, journal entries, [Day 1/Day 2 answers](WORKBOOK_ANSWERS.md),
[earlier audit](DAY2_AUDIT.md) and official teaching guides are retained.
The [official Bobathon repository](https://github.com/probabl-ai/bobathon-esilv)
and the local [Day 2 instructions](DAY2.md) remain the challenge references.

## Recommended model and score attribution

Use the fixed configuration in
[`experiments/_07_features_optimized_complete.py`](../experiments/_07_features_optimized_complete.py).
It averages a general HistGradientBoostingRegressor and a specialist predictor
that fits separate HGBRs for visits with observed/missing OFF measurements.
The specialist also fits a full-data fallback for undersized slices. Each
pipeline uses `skrub.TableVectorizer`, then HGBR with learning rate 0.05,
63 maximum leaves, 1,000 maximum iterations, L2 regularization 1,
early stopping and `random_state=0`. The final average is clipped to [0, 132].

The **Kaggle public score of 3.966 is team-reported and belongs to this
optimized branch**, according to the team’s clarification. It must not be
attributed to the earlier 32-feature model. The original uploaded CSV was not
available for a byte-for-byte comparison with the regenerated file.

| Evidence | RMSE | Meaning |
|---|---:|---|
| Teammate’s historical ensemble CV | 4.2176 ± 0.0686 | Five patient-grouped folds; population fold standard deviation |
| Fresh verification of the completion helper’s ensemble | **4.2160 ± 0.0724** | Five patient-grouped folds; sample fold standard deviation |
| Fresh completion-script CV output | 4.2926 ± 0.0625 | General HGBR component B only, not the ensemble |
| Kaggle public leaderboard | **3.966** | Team-reported score of the optimized version |

Fresh ensemble fold RMSEs were 4.2093, 4.2905, 4.1778, 4.1191 and 4.2833.
Their population standard deviation is 0.0648. Verification loaded the
completion helper’s model definitions without executing its Hub login,
wrapped the exact equal-average/clip prediction rule in a sklearn estimator,
and evaluated it with Skore over the same five `GroupKFold` splits.
The full 18-configuration search and optional CatBoost path were not rerun.

The historical tuning script clips the specialist predictions before its
ensemble CV average; the completion helper averages raw component predictions
and then clips. The fresh verification follows the latter, actual submission
recipe. Unseeded string encoding can also produce small run-to-run changes;
reproduction of the same recipe does not guarantee identical CSV bytes.

## Features and methodology

[`features_optimized.py`](../src/parkinsons/features_optimized.py) produces
58 columns: raw clinical/categorical inputs, patient ON/OFF summaries,
age-based interpolation, neighboring observations, trajectory slopes,
distances to measured visits, missingness indicators and medication proxies.
The interpolation uses age gaps rather than the earlier CSV-index spacing.

- **Patient isolation:** `patient_id` defines both the folds and patient-local
  features, and is removed before modeling. All five outer folds have zero
  overlapping patients. Training and competition test patient sets are also
  disjoint. Recomputing each validation fold’s features independently gave
  exactly the same frame as selecting it from the full feature matrix.
- **Target handling:** feature engineering consumes input measurements only;
  neither `target` nor the CSV `Index` is a model feature. Observed `off` is
  a provided predictor, distinct from the unbiased target. No target leakage
  was found in the reviewed feature calculations.
- **Preprocessing:** each general/specialist pipeline fits its vectorizer on
  its own training slice. TableVectorizer handles categorical inputs; HGBR
  handles remaining numeric missing values. Missing LEDD becomes zero only
  inside medication proxies; this is a modeling convention, not evidence
  that treatment was absent. Raw missing values and indicators are retained.
- **Evaluation:** RMSE is computed on patients absent from training. The
  search reuses these folds for selection, so its best CV score is not an
  independent final performance estimate. HGBR’s internal early-stopping
  split is row-based and may share patients internally; the outer validation
  patients remain excluded from model fitting.
- **Batch assumption:** trajectories use other visits, including later visits,
  for the same held-out patient. This fits prediction from the supplied batch
  of patient records; it is not a prospective, past-visits-only evaluation.
- **Medication proxies:** the exponential parameters are time constants, not
  literal half-lives (half-life = time constant × ln(2)). These are heuristic
  features based on the supplied intake-time variables, not validated clinical
  pharmacokinetic estimates.

## Merge fixes and preserved behavior

All teammate source files were retained. Feature formulas, chosen model
parameters and the completion helper’s prediction recipe were preserved.
Review fixed execution paths in the full search script that could otherwise
fail or fit the wrong final model:

- Correctly unpack the two-item ensemble candidates and map their names to
  the intended model implementations.
- Keep the reference model on its original feature matrix during fitting
  and prediction; reject unknown model names instead of silently falling back.
- Predict through the fitted estimator inside `EstimatorReport`.
- Preserve CatBoost’s categorical-string preparation during final inference.
- Enforce feature schema, row order, template columns and finite predictions
  before writing the submission. Invalid features are not silently invented.

Comments clarify patient-local batch features, medication proxies, historical
metrics and report provenance. The journal’s mistaken description of the
fixed ensemble as C+A was corrected to C+B. README and workbook score
comparisons now attribute 3.966 to the optimized version; earlier results
remain labeled as historical evidence.

## Submission verification and regeneration

From the repository root with the documented virtual environment active:

```bash
python experiments/_07_features_optimized_complete.py
```

Output: **`submissions/07_features_optimized.csv`**. The regenerated file passed:

- Exactly `Index,target`, in the same order as `data/sample_submission.csv`.
- Exactly 11,013 rows, with unique IDs in the exact sample and test row order.
- Finite predictions throughout, ranging from 0 to 91.6451; no NaN or infinity.
- No accidental pandas index column; output is written with `index=False`.

SHA-256 of the file checked during this merge:
`de36c0bfc3c161600d7d21aae6544b6d74301d21209f9f0c4315d5e370c735fe`.
Generated CSVs stay local and Git-ignored. All four raw dataset hashes remained
unchanged. The completion script was executed with Hub access disabled by an
offline verification stub; no credentials were read or transmitted for these
checks, and no new Hub report was published.

The CSV passes the technical upload checks. **For a new Bobathon submission,
the official matching-report requirement still needs to be satisfied:**
the optimized branch’s existing
[CV report 85766](https://skore.probabl.ai/parkinpredict/bobathon-esilv/cross-validations/85766)
and [EstimatorReport 85778](https://skore.probabl.ai/parkinpredict/bobathon-esilv/estimators/85778)
are recorded as reports for component B only. The scripts now label new
component uploads explicitly with `_component`/`_component_cv` keys instead
of presenting them as reports for the ensemble. The separately supplied
[report 83726](https://skore.probabl.ai/parkinpredict/bobathon-esilv/estimators/83726)
requires sign-in and its model identity was not independently verified.
Generate the matching ensemble report through the required Bob/Skore workflow
before a new competition upload; follow the official naming and submission
description requirements. A component report does not verify the ensemble.

## Checks completed

`python -m pytest tests/smoke -q`: **4 passed**. Added regression checks cover
ensemble candidate routing, reference-feature selection, age interpolation,
patient isolation, index relabeling and row/input preservation. Real-data
checks also verified matching 58-column train/test feature schemas, no
infinite numeric features and outer-fold patient isolation. The actual
completion script and independent ensemble CV both completed successfully.
Authenticated setup tests were not run during this merge.
