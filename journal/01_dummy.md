# 01_dummy

<!--
Design note for experiments/01_dummy.py (same stem, one to
one with the script). Write and agree on it before creating the
script.

Lifecycle:
  planned   → draft, not yet approved
  approved  → agreed; safe to create the matching .py
  done      → result recorded; Status block + JOURNAL.md row updated
  abandoned → discarded; record a one-line reason on State
-->

## Question / hypothesis

What is the simplest possible floor-level prediction for the UPDRS motor score, and what RMSE does always predicting the training mean achieve?

## Motivation

Before building any real model, we need a baseline that tells us the *worst acceptable* performance. The mean-predictor is the mathematically provable lower bound for RMSE when you use no features at all — it corresponds to fitting a horizontal line through the data. Any real model must beat this number to be worth anything.

- **Sourcing strategy:** user (explicit request this turn)
- **Source(s):** user request — "Run a dummy regressor from sklearn to act as my baseline model. Push to the hub with the name '01_dummy'."
- **Why this matters:** Establishes the RMSE floor. If a later model only marginally beats it, that tells us the features are weak. If it beats it easily, we know real signal is there to exploit.

## Method

- **Files touched:** `src/parkinsons/__init__.py`, `src/parkinsons/data.py`, `src/parkinsons/pipeline.py`, `src/parkinsons/evaluate.py`, `src/parkinsons/hub.py`, `experiments/01_dummy.py`
- **Change versus baseline (or previous experiment):** This IS the baseline. `DummyRegressor(strategy="mean")` — always predicts the mean of the training targets, no features used at all.
- **Cross-validation:** `splitter=0.2` (random 80/20 visit-level holdout). Per the lab guide, `01_dummy` uses a simple row holdout. The same patient may appear on both sides of the split — that is intentional for this experiment; it illustrates the leakage problem that grouped CV (planned for a later experiment) solves.
- **Out of scope for this experiment:** any feature engineering, imputation, grouped CV by patient.

## Risks / things that could invalidate the result

- The 80/20 split is visit-level, not patient-level. The same patient may appear in both train and test. This inflates the score compared to a proper patient holdout — the risk is noted, not fixed here (that is the next experiment's job).
- `DummyRegressor(strategy="mean")` ignores all features. The result is therefore purely a function of target variance — it is not a meaningful model, only a floor.

## Status

- **State:** done
- **Approved by user on:** 2026-10-07
- **Headline result:** RMSE 16.48, MAE 13.48, R² ≈ 0 (80/20 visit holdout) — report at https://skore.probabl.ai/parkinpredict/bobathon-esilv/estimators/45500
- **Implication for next iteration:** Any model using features should easily beat RMSE 16.48 (= the target's std). `off` (r=0.886) and `on` (r=0.688) are the strongest predictors from the EDA — a Ridge or gradient-boosted tree using them should cut RMSE substantially.
