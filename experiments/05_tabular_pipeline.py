# %% [markdown]
# # Experiment 05_tabular_pipeline: mixed-type features with skrub
#
# **Date:** 2026-10-07
# **Goal:** Add categorical columns (cohort, gene, rater_id) to the HGBR
#           via skrub.tabular_pipeline and measure whether grouped-CV RMSE
#           drops below the numeric-only HGBR from experiment 04.
# **Splitter:** GroupKFold(n_splits=5) on patient_id — same as 04.
# **Result:** filled in after the run.

# %%
import skore

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset

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
# X_full = all columns except Index (index) and patient_id (identifier).
# Keep: cohort, sexM, gene, age_at_diagnosis, age, ledd,
#       time_since_intake_on, time_since_intake_off, rater_id, on, off.

# %%
import pandas as pd

X_raw, y = load_dataset()

NUMERIC_COLS = [
    "sexM", "age_at_diagnosis", "age", "ledd",
    "time_since_intake_on", "time_since_intake_off", "on", "off",
]

# X_full: drop only the patient identifier
X_full = X_raw.drop(columns=["patient_id"])
# X_numeric: only the 8 numeric columns (04 reference)
X_numeric = X_raw[NUMERIC_COLS]
# X_no_rater: full minus rater_id (ablation)
X_no_rater = X_raw.drop(columns=["patient_id", "rater_id"])

print(f"X_full shape:     {X_full.shape}")
print(f"X_numeric shape:  {X_numeric.shape}")
print(f"X_no_rater shape: {X_no_rater.shape}")
print(f"\nX_full columns: {X_full.columns.tolist()}")
print(f"\nMissing values (X_full):")
print(X_full.isna().mean().round(3).to_string())

# %% [markdown]
# ## Grouped CV splits (same construction as 04)

# %%
from sklearn.model_selection import GroupKFold

groups = X_raw["patient_id"]
# cv_splits built on X_full rows — applies to X_numeric/X_no_rater equally
cv_splits = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))
print(f"\nGrouped CV: 5 folds, {groups.nunique()} unique patients")

# %% [markdown]
# ## Run all three models as separate CV evaluations
#
# skore.evaluate requires all models in a dict to share the same X.
# Since numeric HGBR and tabular_pipeline need different X shapes,
# we run them separately and compare the summary numbers manually.

# %%
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor

print("\n--- 1/3 Numeric HGBR reference (04 baseline, X_numeric) ---")
hgbr_ref_report = skore.evaluate(
    HistGradientBoostingRegressor(random_state=0),
    X_numeric,
    y,
    splitter=cv_splits,
)
# Single-estimator CrossValidationReport: rmse() columns are (EstimatorName, 'mean')
# and (EstimatorName, 'std') — use iloc[0, 0/1] to avoid depending on the name.
ref_rmse_df   = hgbr_ref_report.metrics.rmse()
ref_rmse_mean = float(ref_rmse_df.iloc[0, 0])
ref_rmse_std  = float(ref_rmse_df.iloc[0, 1])
print(f"Numeric HGBR:  RMSE {ref_rmse_mean:.4f} +/- {ref_rmse_std:.4f}")

print("\n--- 2/3 tabular_pipeline (all cols, X_full) ---")
tabular_report = skore.evaluate(
    skrub.tabular_pipeline("regressor"),
    X_full,
    y,
    splitter=cv_splits,
)
tab_rmse_df   = tabular_report.metrics.rmse()
tab_rmse_mean = float(tab_rmse_df.iloc[0, 0])
tab_rmse_std  = float(tab_rmse_df.iloc[0, 1])
print(f"tabular_pipeline (all): RMSE {tab_rmse_mean:.4f} +/- {tab_rmse_std:.4f}")

print("\n--- 3/3 tabular_pipeline without rater_id (ablation, X_no_rater) ---")
ablation_report = skore.evaluate(
    skrub.tabular_pipeline("regressor"),
    X_no_rater,
    y,
    splitter=cv_splits,
)
abl_rmse_df    = ablation_report.metrics.rmse()
abl_rmse_mean  = float(abl_rmse_df.iloc[0, 0])
abl_rmse_std   = float(abl_rmse_df.iloc[0, 1])
print(f"tabular (no rater_id): RMSE {abl_rmse_mean:.4f} +/- {abl_rmse_std:.4f}")

# %% [markdown]
# ## Summary comparison

# %%
print("\n=== Summary: grouped-CV RMSE ===")
print(f"{'Model':<35} {'RMSE mean':>10} {'RMSE std':>10}")
print("-" * 58)
print(f"{'04 numeric HGBR (reference)':<35} {ref_rmse_mean:>10.4f} {ref_rmse_std:>10.4f}")
print(f"{'05 tabular_pipeline (all cols)':<35} {tab_rmse_mean:>10.4f} {tab_rmse_std:>10.4f}")
print(f"{'05 tabular (no rater_id)':<35} {abl_rmse_mean:>10.4f} {abl_rmse_std:>10.4f}")

delta_vs_numeric = ref_rmse_mean - tab_rmse_mean
rater_effect     = abl_rmse_mean - tab_rmse_mean   # positive = rater_id helps
print(f"\nCategoricals vs numeric HGBR: delta = {delta_vs_numeric:+.4f}"
      f"  (positive = tabular wins)")
print(f"rater_id effect:              delta = {rater_effect:+.4f}"
      f"  (positive = rater_id helps)")
beats_by_more_than_std = delta_vs_numeric > tab_rmse_std
print(f"Beats numeric by > 1 fold-std ({tab_rmse_std:.4f}): {beats_by_more_than_std}")

# %% [markdown]
# ## Fold-level RMSE breakdown

# %%
print("\n=== Fold-level RMSE ===")
fold_rows = []
for label, cv_rep in [("hgbr_numeric", hgbr_ref_report),
                       ("tabular_full",  tabular_report),
                       ("tabular_no_rater", ablation_report)]:
    for i, fold_rep in enumerate(cv_rep.reports_):
        fold_rows.append({
            "model": label,
            "fold": i,
            "rmse": float(fold_rep.metrics.rmse()),
        })

fold_df = pd.DataFrame(fold_rows)
fold_pivot = fold_df.pivot(index="fold", columns="model", values="rmse")
print(fold_pivot.to_string())
tabular_wins = (fold_pivot["tabular_full"] < fold_pivot["hgbr_numeric"]).sum()
print(f"\ntabular_full wins on {tabular_wins}/5 folds vs numeric HGBR")

# %% [markdown]
# ## Inspect TableVectorizer encoder choices (fitted on full train)

# %%
print("\nFitting tabular_pipeline on full training set to inspect encoder choices ...")
inspector = skrub.tabular_pipeline("regressor")
inspector.fit(X_full, y)
tv = inspector.named_steps["tablevectorizer"]

print(f"\ncardinality_threshold: {tv.cardinality_threshold}")
print("\nEncoder per column:")
for col, enc in tv.transformers_.items():
    kind = [k for k, cols in tv.kind_to_columns_.items() if col in cols]
    kind_str = kind[0] if kind else "?"
    print(f"  {col!r:30s} -> {type(enc).__name__:25s} ({kind_str})")

print()
print(tv.describe_transformations())

# %% [markdown]
# ## Timings comparison

# %%
print("\n=== Fit / predict timings ===")
for label, rep in [("hgbr_numeric", hgbr_ref_report),
                    ("tabular_full", tabular_report),
                    ("tabular_no_rater", ablation_report)]:
    # metrics.fit_time() / predict_time() return DataFrames on CrossValidationReport
    fit_df  = rep.metrics.fit_time()
    pred_df = rep.metrics.predict_time()
    fit_mean  = float(fit_df.iloc[0, 0])
    pred_mean = float(pred_df.iloc[0, 0])
    print(f"  {label:<25}  fit {fit_mean:.3f}s  predict {pred_mean:.4f}s")

# %% [markdown]
# ## Hub push

# %%
if _hub_available and project is not None:
    project.put("05_tabular_pipeline", tabular_report)
    print("\nReport '05_tabular_pipeline' pushed to hub.")
else:
    print("\nSkipping hub push - run 'python scripts/skore-agent' to authenticate.")
