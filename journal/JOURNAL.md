# JOURNAL

<!--
Durable index of every experiment in this workspace. Four sections,
in order: Status, Data understanding (EDA), History, Backlog - keep
them so the file stays quick to scan. Each journal/NN_short_name.md
design note pairs one-to-one with experiments/NN_short_name.py (same
stem).
-->

## Status

- **Project / dataset:** Parkinson's UPDRS score regression
- **Goal:** Predict UPDRS motor scores for Parkinson's patients from clinical features (longitudinal data, multiple visits per patient).
- **Last experiment:** 01_dummy - done
- **Last result:** RMSE 16.48 (80/20 visit holdout, DummyRegressor mean)

- **Workspace decisions** (immutable unless the user pivots):
  - tabular library: pandas - recorded: 2026-10-07
  - env manager: pip+venv - recorded: 2026-10-07
  - agent feature: <installed> - recorded: <YYYY-MM-DD>
  - optional features: <none> - recorded: <YYYY-MM-DD>
  - package name (`src/<pkg>/`): <pkg> - recorded: <YYYY-MM-DD>
  - skore mode: hub - recorded: 2026-10-07
  - skore hub workspace: <hub-workspace-name | n/a> - recorded: <YYYY-MM-DD>
  - skore mlflow tracking uri: <mlflow-tracking-uri | n/a> - recorded: <YYYY-MM-DD>
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

## Backlog

<!-- Proposed experiments not yet scheduled. -->
<!-- | B<N> | idea | source | -->
