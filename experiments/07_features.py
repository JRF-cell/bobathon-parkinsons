# %% [markdown]
# # Experiment 07_features: domain feature engineering + light tuning
#
# **Date:** 2026-10-07
# **Goal:** Add clinically-motivated feature groups targeting the three
#           modelling problems from docs/CONTEXT.md (temporal progression,
#           drug timing, missing data) and lightly tune HGBR.
# **Baseline:** 05/06 tabular_pipeline — RMSE 7.521 ± 0.134.
# **Splitter:** GroupKFold(n_splits=5) on patient_id — same as 04-06.

# %%
import skore
from sklearn.pipeline import Pipeline

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.features import add_features

# %% [markdown]
# ## Hub login (graceful degradation if .skore missing)

# %%
_hub_available = False
project = None
try:
    from skore import login
    from parkinsons.hub import load_skore_credentials

    cfg = load_skore_credentials()
    login(mode="hub")
    project = skore.Project(
        name="bobathon-esilv",
        mode="hub",
        workspace=cfg["workspace"],
    )
    _hub_available = True
    print("Hub login successful.")
except Exception as _hub_err:
    print(f"Hub unavailable ({_hub_err}). Offline run.")

# %% [markdown]
# ## Data
#
# Load raw X_train + y_train.  patient_id is kept for grouping in add_features()
# and in GroupKFold — it is dropped by add_features() before modelling.

# %%
import pandas as pd
import numpy as np

X_raw, y = load_dataset()

print(f"X_raw shape: {X_raw.shape}")
print(f"Columns: {X_raw.columns.tolist()}")

# %% [markdown]
# ## Grouped CV splits — same as 03-06

# %%
from sklearn.model_selection import GroupKFold

groups = X_raw["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X_raw, y, groups=groups))
print(f"GroupKFold: 5 folds on {groups.nunique()} unique patients")

# %% [markdown]
# ## Step 1 — Baseline: best model from 04-06 on raw features
#
# Reuses the reported RMSE 7.521 from the journal.  We also run it here
# so the comparison table is produced on the same splits.

# %%
from sklearn.ensemble import HistGradientBoostingRegressor
import skrub

# Raw feature matrix (no patient_id, no engineered cols)
X_baseline = X_raw.drop(columns=["patient_id"])

print("\n--- Baseline: HGBR + TableVectorizer on raw features ---")
baseline_report = skore.evaluate(
    skrub.tabular_pipeline("regressor"),
    X_baseline,
    y,
    splitter=cv_splits,
)
base_rmse_df   = baseline_report.metrics.rmse()
base_rmse_mean = float(base_rmse_df.iloc[0, 0])
base_rmse_std  = float(base_rmse_df.iloc[0, 1])
print(f"Baseline RMSE: {base_rmse_mean:.4f} +/- {base_rmse_std:.4f}")

# %% [markdown]
# ## Step 2 — Incremental feature ablation
#
# add_features() is called on X_raw (which still contains patient_id for
# grouping). It returns a new DataFrame without patient_id.
# The cv_splits were built on X_raw rows — row order is preserved by
# add_features so the integer indices are still valid.

# %%
# Full engineered feature matrix (groups A+B+C+D)
X_eng = add_features(X_raw)
print(f"\nEngineered feature matrix: {X_eng.shape}")
print(f"New columns: {[c for c in X_eng.columns if c not in X_baseline.columns]}")
print(f"\nMissing values in engineered features:")
new_cols = [c for c in X_eng.columns if c not in X_baseline.columns]
print(X_eng[new_cols].isna().mean().round(3).to_string())

# %%
# Helper: run CV and return (mean, std)
def cv_rmse(X, label):
    rep = skore.evaluate(
        skrub.tabular_pipeline("regressor"),
        X,
        y,
        splitter=cv_splits,
    )
    df   = rep.metrics.rmse()
    mean = float(df.iloc[0, 0])
    std  = float(df.iloc[0, 1])
    print(f"  {label:<45} RMSE {mean:.4f} +/- {std:.4f}")
    return mean, std, rep

# %%
print("\n=== Incremental feature ablation ===")

# Group A: temporal
TEMPORAL_COLS = ["disease_duration", "years_since_first_visit", "visit_rank", "n_visits"]

# Group B: trajectory
TRAJECTORY_COLS = [
    "off_mean", "off_median", "off_min", "off_max",
    "on_mean",
    "off_prev", "off_next", "off_interp", "off_slope",
]

# Group C: drug timing (ledd_missing)
TIMING_COLS = ["ledd_missing"]

# Group D: missingness flags
MISSINGNESS_COLS = ["off_missing", "on_missing", "gene_missing", "both_measured"]
# ledd_missing is already in C; no need to duplicate

# ---- run ablations ----
base_mean, base_std, _ = cv_rmse(X_baseline, "raw (baseline)")

X_a = X_eng[list(X_baseline.columns) + TEMPORAL_COLS]
a_mean, a_std, rep_a = cv_rmse(X_a, "+temporal (A)")

X_ab = X_eng[list(X_baseline.columns) + TEMPORAL_COLS + TRAJECTORY_COLS]
ab_mean, ab_std, rep_ab = cv_rmse(X_ab, "+temporal+trajectory (A+B)")

X_abc = X_eng[list(X_baseline.columns) + TEMPORAL_COLS + TRAJECTORY_COLS + TIMING_COLS]
abc_mean, abc_std, rep_abc = cv_rmse(X_abc, "+temporal+trajectory+timing (A+B+C)")

X_all = X_eng[list(X_baseline.columns) + TEMPORAL_COLS + TRAJECTORY_COLS + TIMING_COLS + MISSINGNESS_COLS]
all_mean, all_std, rep_all = cv_rmse(X_all, "+all (A+B+C+D)")

# %%
print("\n=== Ablation table ===")
print(f"{'Config':<45} {'RMSE mean':>10} {'RMSE std':>10} {'vs baseline':>12}")
print("-" * 80)
rows = [
    ("raw (baseline)", base_mean, base_std),
    ("+temporal (A)", a_mean, a_std),
    ("+temporal+trajectory (A+B)", ab_mean, ab_std),
    ("+temporal+trajectory+timing (A+B+C)", abc_mean, abc_std),
    ("+all (A+B+C+D)", all_mean, all_std),
]
for name, mean, std in rows:
    delta = mean - base_mean
    print(f"  {name:<43} {mean:>10.4f} {std:>10.4f} {delta:>+12.4f}")

# Decide best feature set: choose config with lowest mean RMSE
best_configs = sorted(rows, key=lambda r: r[1])
best_name, best_mean, best_std = best_configs[0]
print(f"\nBest feature config after ablation: '{best_name}' — RMSE {best_mean:.4f} +/- {best_std:.4f}")

# Map config name → X matrix
_config_map = {
    "raw (baseline)": X_baseline,
    "+temporal (A)": X_a,
    "+temporal+trajectory (A+B)": X_ab,
    "+temporal+trajectory+timing (A+B+C)": X_abc,
    "+all (A+B+C+D)": X_all,
}
X_best_ablation = _config_map[best_name]

# %% [markdown]
# ## Step 3 — Light hyperparameter tuning of HGBR
#
# Grid: learning_rate ∈ {0.03, 0.1}, max_leaf_nodes ∈ {15, 31, 63},
#       l2_regularization ∈ {0, 1}, max_iter=1000, early_stopping=True.
# Total: 2 × 3 × 2 = 12 configs.

# %%
from itertools import product as iproduct

tuning_results = []
print("\n=== Light HGBR tuning ===")
print(f"{'lr':>5} {'leaves':>6} {'l2':>4}   RMSE mean +/- std")
print("-" * 50)

for lr, leaves, l2 in iproduct([0.03, 0.1], [15, 31, 63], [0, 1]):
    hgbr = HistGradientBoostingRegressor(
        learning_rate=lr,
        max_leaf_nodes=leaves,
        max_iter=1000,
        early_stopping=True,
        l2_regularization=l2,
        random_state=0,
    )
    pipe = Pipeline([
        ("tv", skrub.TableVectorizer()),
        ("hgbr", hgbr),
    ])
    rep = skore.evaluate(pipe, X_best_ablation, y, splitter=cv_splits)
    df   = rep.metrics.rmse()
    mean = float(df.iloc[0, 0])
    std  = float(df.iloc[0, 1])
    tuning_results.append((lr, leaves, l2, mean, std, rep))
    print(f"  {lr:>5} {leaves:>6} {l2:>4}   {mean:.4f} +/- {std:.4f}")

# %%
# Select best: lowest mean; break ties by highest l2 (simpler model)
tuning_results_sorted = sorted(tuning_results, key=lambda r: (r[3], -r[2]))
best_lr, best_leaves, best_l2, best_tuned_mean, best_tuned_std, best_cv_report = tuning_results_sorted[0]

print(f"\n=== Best tuned config ===")
print(f"  learning_rate={best_lr}, max_leaf_nodes={best_leaves}, l2_regularization={best_l2}")
print(f"  RMSE: {best_tuned_mean:.4f} +/- {best_tuned_std:.4f}")
print(f"  Gain vs ablation best: {best_tuned_mean - best_mean:+.4f}")
print(f"  Gain vs raw baseline:  {best_tuned_mean - base_mean:+.4f}")

# %% [markdown]
# ## Fold-level breakdown for the final model

# %%
print("\n=== Fold-level RMSE — final model ===")
fold_rmses = []
for i, fold_rep in enumerate(best_cv_report.reports_):
    fr = float(fold_rep.metrics.rmse())
    fold_rmses.append(fr)
    print(f"  Fold {i}: {fr:.4f}")
fold_series = pd.Series(fold_rmses)
print(f"Mean: {fold_series.mean():.4f}  Std: {fold_series.std():.4f}")

# %% [markdown]
# ## Step 4 — Hub push

# %%
if _hub_available and project is not None:
    project.put("07_features", best_cv_report)
    print("\nReport '07_features' pushed to hub.")
else:
    print("\nSkipping hub push — hub unavailable.")

# %% [markdown]
# ## Step 4b — EstimatorReport for Kaggle URL

# %%
# Fit on full training set for the EstimatorReport
final_pipe = Pipeline([
    ("tv", skrub.TableVectorizer()),
    ("hgbr", HistGradientBoostingRegressor(
        learning_rate=best_lr,
        max_leaf_nodes=best_leaves,
        max_iter=1000,
        early_stopping=True,
        l2_regularization=best_l2,
        random_state=0,
    )),
])

print("\nFitting final pipeline on full training set ...")
final_pipe.fit(X_best_ablation, y)
print("Fit complete.")

# %%
# EstimatorReport: use an unfitted estimator (it fits internally) on the
# last fold so we get a held-out evaluation for the Kaggle URL.
from sklearn.pipeline import Pipeline as _Pipeline
train_idx, test_idx = cv_splits[-1]
X_held = X_best_ablation.iloc[test_idx]
y_held = y.iloc[test_idx]

est_pipe_unfitted = Pipeline([
    ("tv", skrub.TableVectorizer()),
    ("hgbr", HistGradientBoostingRegressor(
        learning_rate=best_lr,
        max_leaf_nodes=best_leaves,
        max_iter=1000,
        early_stopping=True,
        l2_regularization=best_l2,
        random_state=0,
    )),
])

est_report = skore.EstimatorReport(
    est_pipe_unfitted,
    X_train=X_best_ablation.iloc[train_idx],
    y_train=y.iloc[train_idx],
    X_test=X_held,
    y_test=y_held,
)

if _hub_available and project is not None:
    project.put("07_features_estimator", est_report)
    print("EstimatorReport '07_features_estimator' pushed to hub.")
    # Print the URL for Kaggle
    hub_url = cfg.get("hub_url", "").rstrip("/")
    workspace = cfg.get("workspace", "")
    print(f"\n=== Skore Hub URL ===")
    print(f"  {hub_url}/workspaces/{workspace}")
    print("  (paste this on Kaggle — look for '07_features_estimator' in the UI)")
else:
    print("Hub unavailable — EstimatorReport not pushed.")

# %% [markdown]
# ## Step 5 — Predict X_test and write submission

# %%
X_test_raw = load_test_dataset()

# Preserve the Index column for the submission CSV
# load_test_dataset returns a DataFrame with 'Index' as a column (not the index)
X_test_indexed = X_test_raw.set_index("Index")

# Apply the same feature engineering — add_features expects patient_id for grouping
# add_features() handles NaN age_at_diagnosis gracefully (disease_duration → NaN)
X_test_eng_full = add_features(X_test_indexed)

print(f"\nX_test engineered shape: {X_test_eng_full.shape}")

# Keep only the columns used by the best ablation config
final_feature_cols = list(X_best_ablation.columns)
X_test_final = X_test_eng_full[final_feature_cols]

test_preds = final_pipe.predict(X_test_final)
print(f"Test predictions: {len(test_preds)} rows  range {test_preds.min():.2f}–{test_preds.max():.2f}  mean {test_preds.mean():.2f}")

# Write submission
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

submission = pd.DataFrame({
    "Index": X_test_indexed.index,
    "target": test_preds,
})
out_path = submissions_dir / "07_features.csv"
submission.to_csv(out_path, index=False)
print(f"\nSubmission written: {out_path}  ({len(submission)} rows)")
print(submission.head())

# %% [markdown]
# ## 50 largest errors — what do they share?

# %%
print("\n=== 50 largest CV errors — analysis ===")

# Collect OOF predictions across all 5 folds
oof_preds = pd.Series(np.nan, index=X_best_ablation.index)
for tr_idx, te_idx in cv_splits:
    fold_pipe = Pipeline([
        ("tv", skrub.TableVectorizer()),
        ("hgbr", HistGradientBoostingRegressor(
            learning_rate=best_lr,
            max_leaf_nodes=best_leaves,
            max_iter=1000,
            early_stopping=True,
            l2_regularization=best_l2,
            random_state=0,
        )),
    ])
    fold_pipe.fit(X_best_ablation.iloc[tr_idx], y.iloc[tr_idx])
    oof_preds.iloc[te_idx] = fold_pipe.predict(X_best_ablation.iloc[te_idx])

oof_errors = (oof_preds - y).abs()
top50_idx = oof_errors.nlargest(50).index

# Join back with raw X for clinical context
raw_context = X_raw.loc[top50_idx].copy()
raw_context["oof_pred"] = oof_preds.loc[top50_idx]
raw_context["true_target"] = y.loc[top50_idx]
raw_context["abs_error"] = oof_errors.loc[top50_idx]

print(f"\nTop-50 error summary (n=50 visits):")
print(f"  off_missing rate:    {raw_context['off'].isna().mean():.2%}  (train mean: {X_raw['off'].isna().mean():.2%})")
print(f"  on_missing rate:     {raw_context['on'].isna().mean():.2%}  (train mean: {X_raw['on'].isna().mean():.2%})")
print(f"  ledd_missing rate:   {raw_context['ledd'].isna().mean():.2%}  (train mean: {X_raw['ledd'].isna().mean():.2%})")
print(f"  gene_missing rate:   {raw_context['gene'].isna().mean():.2%}  (train mean: {X_raw['gene'].isna().mean():.2%})")

print(f"\n  Cohort distribution:")
print(raw_context["cohort"].value_counts(normalize=True).round(3).to_string())

print(f"\n  Rater distribution (top-5):")
print(raw_context["rater_id"].value_counts().head(5).to_string())

print(f"\n  Disease duration stats:")
raw_context["disease_duration"] = raw_context["age"] - raw_context["age_at_diagnosis"]
print(raw_context["disease_duration"].describe().round(2).to_string())

print(f"\n  time_since_intake_off stats (hours):")
print(raw_context["time_since_intake_off"].describe().round(2).to_string())

print(f"\n  abs_error stats:")
print(raw_context["abs_error"].describe().round(2).to_string())

# %% [markdown]
# ## Summary

# %%
print("\n" + "="*60)
print("EXPERIMENT 07_features — SUMMARY")
print("="*60)
print(f"  Baseline (raw, 05/06):         RMSE {base_mean:.4f} +/- {base_std:.4f}")
print(f"  Best feature config:           '{best_name}'")
print(f"                                 RMSE {best_mean:.4f} +/- {best_std:.4f}  (delta {best_mean-base_mean:+.4f})")
print(f"  Best tuned HGBR:               lr={best_lr}, leaves={best_leaves}, l2={best_l2}")
print(f"                                 RMSE {best_tuned_mean:.4f} +/- {best_tuned_std:.4f}  (delta {best_tuned_mean-base_mean:+.4f})")
print(f"  Submission:                    submissions/07_features.csv  ({len(submission)} rows)")
print("="*60)
