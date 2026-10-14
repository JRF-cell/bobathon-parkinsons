# JOURNAL

<!--
Durable index of every experiment in this workspace. Four sections,
in order: Status, Data understanding (EDA), History, Backlog - keep
them so the file stays quick to scan. Each journal/NN_short_name.md
design note pairs one-to-one with experiments/NN_short_name.py (same
stem).
-->

## Status

- **Kaggle public score:** **3.966**, team-reported on 2026-10-14; confirmed by the team as Yanice’s optimized branch (`07_features_optimized.csv`).
- **Merge review:** [optimized model and verification](../docs/OPTIMIZED_MERGE_REVIEW.md).
- **Workbook:** [Day 1 and Day 2 answers](../docs/WORKBOOK_ANSWERS.md).
- **Team-supplied Hub report:** [EstimatorReport 83726](https://skore.probabl.ai/parkinpredict/bobathon-esilv/estimators/83726); model identity and metrics require authenticated inspection.
- **2026-10-14 audit:** see [Day 2 audit](../docs/DAY2_AUDIT.md) for local
  verification, submission checks and limitations. Scores below are historical
  run records. The two 07 scripts use different feature sets (29 versus 32);
  the submission variant now uses the separate key/CSV `07_features_32`.
  Earlier 32-feature verification: **4.4807 ± 0.0653** grouped-CV RMSE for the
  32-feature model; Bob regeneration and a matching Hub URL remain required.
- **Project / dataset:** Parkinson's UPDRS score regression
- **Goal:** Predict UPDRS motor scores for Parkinson's patients from clinical features (longitudinal data, multiple visits per patient).
- **Last experiment:** 07_features_optimized - done
- **Last result:** RMSE 4.2176 ± 0.0686 (grouped-CV 5-fold, Ensemble specialist+HGBR-opt, lr=0.05 leaves=63 l2=1, 58 optimized features) — **−5.7 % vs 07_features 4.472** (re-measured)

- **Workspace decisions** (immutable unless the user pivots):
  - tabular library: pandas - recorded: 2026-10-07
  - env manager: pip+venv - recorded: 2026-10-07
  - package name: `parkinsons` (`src/parkinsons/`)
  - skore mode: hub - recorded: 2026-10-07
  - skore hub workspace: `parkinpredict` (team-supplied report URL)
  - student prior: beginner - recorded: 2026-10-07
  - CV splitter family: GroupKFold (patient_id) - recorded: 2026-10-07

## Data understanding (EDA)

- **Status:** done - 2026-10-07
- **Summary:** 44,590 training rows × 12 columns; longitudinal panel with ~8 visits per patient (5,576 unique patients). Regression target (UPDRS score, mean ≈ 37.5, roughly Gaussian). `off` (r = 0.886) and `on` (r = 0.688) are dominant predictors but 42% / 30% missing — confirm availability at inference. GroupKFold on `patient_id` is mandatory to prevent leakage.
- **Report:** [data/eda.md](../data/eda.md)

## History

<!-- One row per experiment; newest at the bottom. -->
| NN | Name | Status | Headline metric | Design note |
|---|---|---|---|---|
| 01 | dummy | done | RMSE 16.48, MAE 13.48, R²≈0 (80/20 visit holdout) | [journal/01_dummy.md](01_dummy.md) |
| 02 | ridge | done | RMSE 10.49, MAE 8.44, R²=0.595 (80/20 visit holdout, alpha=0.01, numeric cols) — −36.3 % vs dummy | [journal/02_ridge.md](02_ridge.md) |
| 04 | hgbr | done | RMSE 7.75 ± 0.15 (grouped-CV 5-fold, native NaN) vs Ridge 10.60 ± 0.27 — delta 2.84, wins 5/5 folds, −53.0 % vs dummy | [journal/04_hgbr.md](04_hgbr.md) |
| 05 | tabular_pipeline | done | RMSE 7.521 ± 0.134 (grouped-CV 5-fold, all cols + rater_id via StringEncoder) — delta −0.231 vs numeric HGBR, wins 5/5 folds, −54.4 % vs dummy | [journal/05_tabular_pipeline.md](05_tabular_pipeline.md) |
| 06 | dataops | done | RMSE 7.537 ± 0.138 (grouped-CV 5-fold, DataOps graph, same cols as 05) — diff +0.016 vs 05 (< 1 fold-std, within noise); learner smoke-test 11013 rows OK | [journal/06_dataops.md](06_dataops.md) |
| 07 | features | done | RMSE **4.532 ± 0.050** (grouped-CV 5-fold, HGBR lr=0.03/leaves=63/l2=1, all feature groups A+B+C+D) — **−39.8 %** vs raw baseline 7.52; trajectory group alone −2.78 RMSE; submission 07_features.csv | [journal/07_features.md](07_features.md) |
| 07 | features_optimized | done | RMSE **4.2176 ± 0.0686** (grouped-CV 5-fold, Ensemble specialist+HGBR-opt, lr=0.05/leaves=63/l2=1, 58 cols) — **−5.7 %** vs 07_features 4.472 (re-measured on same splits); age-based interp fix + ON trajectory + PK proxies + specialist routing; submission 07_features_optimized.csv; Hub: 07_features_optimized_cv + 07_features_optimized | [journal/07_features_optimized.md](07_features_optimized.md) |

## Backlog

<!-- Proposed experiments not yet scheduled. -->
<!-- | B<N> | idea | source | -->
| B01 (implemented in optimized branch) | pk_timing | Pharmacokinetic drug-timing feature: `ledd_effect = ledd × exp(−λ × time_since_intake_on)` to explicitly unbias the OFF score for residual levodopa. Directly targets the drug-timing problem in CONTEXT.md, which is still the least-addressed. | journal/07_features.md (§CONTEXT.md problem still least addressed) |
| B02 | catboost_native_cats | Try CatBoostRegressor with native categorical handling for cohort/gene/rater_id once environment allows install. Skipped in 07_features_optimized because catboost not installed. | journal/07_features_optimized.md |
| B03 | stacking_meta_learner | Replace simple-average ensemble with a Ridge meta-learner stacked on OOF predictions of specialist + HGBR. May squeeze another 0.05–0.1 RMSE if sub-models disagree structurally. | journal/07_features_optimized.md |
