# %% [markdown]
# # Experiment 04_hgbr: HistGradientBoosting — missingness as signal
#
# **Date:** 2026-10-07
# **Goal:** Beat Ridge (grouped-CV RMSE from experiment 02 recomputed here) by
#           keeping NaN cells as learned signal instead of median-imputing them.
# **Splitter:** GroupKFold(n_splits=5) on patient_id — blocks entire patients
#               per fold, matching the Kaggle holdout structure.
# **Result:** filled in after the run.

# %%
import skore

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset

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

# %%
import pandas as pd

X_full, y = load_dataset()

FEATURE_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]

X = X_full[FEATURE_COLS]
print(f"Training set: {X.shape[0]} rows x {X.shape[1]} features")
print("Missing values per column:")
print(X.isna().mean().round(3).to_string())

# %% [markdown]
# ## Grouped CV splits
#
# GroupKFold on patient_id ensures entire patients go to one side only.
# cv_splits is a list of (train_idx, test_idx) pairs — reused unchanged by
# experiment 05, so results are directly comparable.

# %%
from sklearn.model_selection import GroupKFold

groups = X_full["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
print(f"Grouped CV: 5 folds, {len(set(groups))} unique patients")

# %% [markdown]
# ## Models
#
# ridge_02  : exact pipeline from experiment 02 (imputed Ridge alpha=0.01)
# hgbr      : no imputer — NaN handled natively
# hgbr_imputed : ablation — imputed HGBR to measure cost of hiding the holes

# %%
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

ridge_02 = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=0.01),
)

hgbr = HistGradientBoostingRegressor(random_state=0)

hgbr_imputed = make_pipeline(
    SimpleImputer(strategy="median"),
    HistGradientBoostingRegressor(random_state=0),
)

# %% [markdown]
# ## Main comparison: Ridge vs HGBR (grouped CV)

# %%
print("\nRunning Ridge vs HGBR comparison (grouped CV) ...")
comparison_report = skore.evaluate(
    {"ridge": ridge_02, "hgbr": hgbr},
    X,
    y,
    splitter=cv_splits,
)
print("\n=== Ridge vs HGBR (grouped CV) ===")
summary = comparison_report.metrics.summarize()
print(summary)

# Extract RMSE mean and std for both
# ComparisonReport.metrics.rmse() returns a DataFrame with multi-level columns:
# ('mean', 'ridge'), ('mean', 'hgbr'), ('std', 'ridge'), ('std', 'hgbr')
rmse_df = comparison_report.metrics.rmse()
ridge_rmse_mean = float(rmse_df.loc["RMSE", ("mean", "ridge")])
ridge_rmse_std  = float(rmse_df.loc["RMSE", ("std",  "ridge")])
hgbr_rmse_mean  = float(rmse_df.loc["RMSE", ("mean", "hgbr")])
hgbr_rmse_std   = float(rmse_df.loc["RMSE", ("std",  "hgbr")])

print(f"\nRidge RMSE: {ridge_rmse_mean:.4f} +/- {ridge_rmse_std:.4f}")
print(f"HGBR  RMSE: {hgbr_rmse_mean:.4f} +/- {hgbr_rmse_std:.4f}")
delta = ridge_rmse_mean - hgbr_rmse_mean
print(f"Delta (Ridge - HGBR): {delta:.4f}  (positive = HGBR wins)")
beats_by_more_than_std = delta > hgbr_rmse_std
print(f"HGBR beats Ridge by > 1 fold-std: {beats_by_more_than_std}")

# %% [markdown]
# ## Fold-level RMSE breakdown — does HGBR win on every fold?

# %%
print("\n=== Fold-level RMSE ===")
# Each cv report has 5 EstimatorReport entries
# For the comparison report, access per-estimator CV reports via .reports_
# structure: comparison_report.reports_ = {"ridge": CrossValReport, "hgbr": CrossValReport}
fold_data = []
for est_name in ["ridge", "hgbr"]:
    cv_rep = comparison_report.reports_[est_name]
    for i, fold_rep in enumerate(cv_rep.reports_):
        fold_rmse = float(fold_rep.metrics.rmse())
        fold_data.append({"estimator": est_name, "fold": i, "rmse": fold_rmse})

fold_df = pd.DataFrame(fold_data)
fold_pivot = fold_df.pivot(index="fold", columns="estimator", values="rmse")
fold_pivot["hgbr_wins"] = fold_pivot["hgbr"] < fold_pivot["ridge"]
print(fold_pivot.to_string())
hgbr_wins_all = fold_pivot["hgbr_wins"].all()
hgbr_wins_count = fold_pivot["hgbr_wins"].sum()
print(f"\nHGBR wins on {hgbr_wins_count}/5 folds, all folds: {hgbr_wins_all}")

# %% [markdown]
# ## Timings comparison

# %%
print("\n=== Fit / predict timings ===")
print(comparison_report.metrics.timings())

# %% [markdown]
# ## Ablation: HGBR with imputation vs native NaN

# %%
print("\nRunning HGBR ablation (imputed vs native) ...")
ablation_report = skore.evaluate(
    {"hgbr_native": hgbr, "hgbr_imputed": hgbr_imputed},
    X,
    y,
    splitter=cv_splits,
)
print("\n=== HGBR ablation: native NaN vs median-imputed ===")
print(ablation_report.metrics.summarize())

abl_rmse = ablation_report.metrics.rmse()
native_rmse  = float(abl_rmse.loc["RMSE", ("mean", "hgbr_native")])
imputed_rmse = float(abl_rmse.loc["RMSE", ("mean", "hgbr_imputed")])
print(f"\nHGBR native   RMSE: {native_rmse:.4f}")
print(f"HGBR imputed  RMSE: {imputed_rmse:.4f}")
print(f"Cost of imputing (native - imputed, positive = imputing hurts): {imputed_rmse - native_rmse:.4f}")

# %% [markdown]
# ## Permutation importance — fold 0 held-out set

# %%
from sklearn.inspection import permutation_importance

print("\nComputing permutation importance on fold 0 held-out set ...")
hgbr_cv_report = comparison_report.reports_["hgbr"]
fold0_report = hgbr_cv_report.reports_[0]

X_te_fold0 = fold0_report.X_test
y_te_fold0  = fold0_report.y_test
fitted_hgbr = fold0_report.estimator_

pi_result = permutation_importance(
    fitted_hgbr,
    X_te_fold0,
    y_te_fold0,
    n_repeats=10,
    random_state=0,
    scoring="neg_root_mean_squared_error",
)
pi_series = pd.Series(
    pi_result.importances_mean,
    index=FEATURE_COLS,
).sort_values(ascending=False)
print("\n=== HGBR permutation importance (fold 0, neg-RMSE scorer) ===")
print(pi_series.to_string())

# %% [markdown]
# ## Hub push

# %%
if _hub_available and project is not None:
    project.put("04_hgbr", comparison_report)
    print("\nComparison report '04_hgbr' pushed to hub.")
    # If HGBR beats Ridge by more than 1 fold-std, also push a single-fold
    # EstimatorReport for a valid Kaggle hub URL
    if beats_by_more_than_std:
        project.put("04_hgbr_estimator", fold0_report)
        print("Single-fold EstimatorReport '04_hgbr_estimator' pushed - use this URL for Kaggle submission.")
else:
    print("Skipping hub push - run 'python scripts/skore-agent' to authenticate.")

# %% [markdown]
# ## Final model + submission (only if HGBR beats Ridge by > 1 fold-std)

# %%
from sklearn.base import clone

if beats_by_more_than_std:
    print(f"\nHGBR beats Ridge by {delta:.4f} > std {hgbr_rmse_std:.4f} - writing submission.")
    submissions_dir = PROJECT_ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)

    X_test_full = load_test_dataset()
    X_test = X_test_full[FEATURE_COLS]

    final_hgbr = clone(hgbr)
    final_hgbr.fit(X, y)
    predictions = final_hgbr.predict(X_test)

    submission = pd.DataFrame({"Index": X_test_full["Index"], "target": predictions})
    submission_path = submissions_dir / "04_hgbr.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}  ({len(submission)} rows)")
    print("Upload this file on Kaggle and paste the hub URL in the Submission Description.")
else:
    print(f"\nHGBR delta {delta:.4f} <= fold-std {hgbr_rmse_std:.4f} - no submission written.")
    print("HGBR does not beat Ridge by more than noise; no Kaggle submission for 04_hgbr.")
