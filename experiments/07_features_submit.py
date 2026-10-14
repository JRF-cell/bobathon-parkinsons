"""
Continuation script for 07_features:
- EstimatorReport + hub push
- X_test prediction + submission CSV
- 50 largest OOF errors analysis
Runs only the post-CV steps; does not redo the ablation/tuning grid.

Audit note: this script uses 32 input features, including on_median,
on_min and on_max; 07_features.py's all-feature configuration uses 29.
They are distinct candidates and now use separate CSV/report keys.
See docs/DAY2_AUDIT.md for independently checked scores and upload conditions.
"""
import sys
import numpy as np
import pandas as pd
import skore
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.features import add_features

# ---- known best config from the run ----
BEST_LR      = 0.03
BEST_LEAVES  = 63
BEST_L2      = 1
# Historical raw-feature baseline, not recomputed in this continuation.
BASE_MEAN    = 7.5224
# Choose a fresh matching key for any subsequent Kaggle submission.
REPORT_KEY   = "07_features_32"

# ---- hub login ----
_hub_available = False
project = None
cfg = {}
try:
    from skore import login
    from parkinsons.hub import load_skore_credentials
    cfg = load_skore_credentials()
    login(mode="hub")
    project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])
    _hub_available = True
    print("Hub login successful.")
except Exception as e:
    print(f"Hub unavailable ({e}). Offline run.")

# ---- data ----
X_raw, y = load_dataset()
groups    = X_raw["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X_raw, y, groups=groups))

X_eng     = add_features(X_raw)
X_baseline_cols = [c for c in X_raw.columns if c != "patient_id"]

# Expanded feature set: also includes on_median/on_min/on_max, which are
# absent from the tuning script's TRAJECTORY_COLS. Preserve for the audit.
TEMPORAL_COLS    = ["disease_duration", "years_since_first_visit", "visit_rank", "n_visits"]
TRAJECTORY_COLS  = ["off_mean", "off_median", "off_min", "off_max",
                    "on_mean", "on_median", "on_min", "on_max",
                    "off_prev", "off_next", "off_interp", "off_slope"]
TIMING_COLS      = ["ledd_missing"]
MISSINGNESS_COLS = ["off_missing", "on_missing", "gene_missing", "both_measured"]
ALL_ENG_COLS     = TEMPORAL_COLS + TRAJECTORY_COLS + TIMING_COLS + MISSINGNESS_COLS
X_best = X_eng[X_baseline_cols + ALL_ENG_COLS]

# ---- EstimatorReport (unfitted pipe, last CV fold) ----
train_idx, test_idx = cv_splits[-1]
est_pipe = Pipeline([
    ("tv", skrub.TableVectorizer()),
    ("hgbr", HistGradientBoostingRegressor(
        learning_rate=BEST_LR,
        max_leaf_nodes=BEST_LEAVES,
        max_iter=1000,
        early_stopping=True,
        l2_regularization=BEST_L2,
        random_state=0,
    )),
])

print("\nBuilding EstimatorReport (fits on fold-4 train) ...")
est_report = skore.EstimatorReport(
    est_pipe,
    X_train=X_best.iloc[train_idx],
    y_train=y.iloc[train_idx],
    X_test=X_best.iloc[test_idx],
    y_test=y.iloc[test_idx],
)
print("EstimatorReport built.")

if _hub_available and project is not None:
    project.put(REPORT_KEY, est_report)
    print(f"\nEstimatorReport '{REPORT_KEY}' pushed.")
    print("Use the actual EstimatorReport URL printed by project.put above in Kaggle's Description.")
else:
    print("Hub unavailable — EstimatorReport not pushed.")

# All rows are passed to fit; HGBR internally reserves a row-level fraction
# for early stopping. This does not use outer GroupKFold validation targets.
# ---- Fit final pipe on ALL training rows ----
final_pipe = Pipeline([
    ("tv", skrub.TableVectorizer()),
    ("hgbr", HistGradientBoostingRegressor(
        learning_rate=BEST_LR,
        max_leaf_nodes=BEST_LEAVES,
        max_iter=1000,
        early_stopping=True,
        l2_regularization=BEST_L2,
        random_state=0,
    )),
])
print("\nFitting final pipeline on ALL training rows ...")
final_pipe.fit(X_best, y)
print("Fit complete.")

# ---- X_test prediction and submission ----
X_test_raw     = load_test_dataset()
X_test_indexed = X_test_raw.set_index("Index")
X_test_eng     = add_features(X_test_indexed)
X_test_final   = X_test_eng[X_baseline_cols + ALL_ENG_COLS]

test_preds = final_pipe.predict(X_test_final)
print(f"Test predictions: {len(test_preds)} rows  "
      f"range {test_preds.min():.2f}–{test_preds.max():.2f}  mean {test_preds.mean():.2f}")
sample = pd.read_csv(PROJECT_ROOT / "data" / "sample_submission.csv")
assert len(test_preds) == len(sample) == len(X_test_indexed), "Prediction count differs from test/template"
assert X_test_final.index.equals(X_test_indexed.index), "Feature engineering changed test-row order"
assert np.isfinite(test_preds).all(), "Predictions contain NaN or infinity"

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
submission = pd.DataFrame({"Index": X_test_indexed.index, "target": test_preds})
assert submission.columns.tolist() == sample.columns.tolist() == ["Index", "target"]
assert submission["Index"].is_unique, "Duplicate submission IDs"
assert submission["Index"].equals(sample["Index"]), "Submission IDs/order differ from template"
out_path = submissions_dir / f"{REPORT_KEY}.csv"
submission.to_csv(out_path, index=False)
print(f"\nSubmission written: {out_path}  ({len(submission)} rows)")
print(submission.head())

# ---- 50 largest OOF errors ----
print("\n=== 50 largest CV errors — analysis ===")
oof_preds = pd.Series(np.nan, index=X_best.index)
fold_rmses = []
for tr_idx, te_idx in cv_splits:
    fp = Pipeline([
        ("tv", skrub.TableVectorizer()),
        ("hgbr", HistGradientBoostingRegressor(
            learning_rate=BEST_LR,
            max_leaf_nodes=BEST_LEAVES,
            max_iter=1000,
            early_stopping=True,
            l2_regularization=BEST_L2,
            random_state=0,
        )),
    ])
    fp.fit(X_best.iloc[tr_idx], y.iloc[tr_idx])
    oof_preds.iloc[te_idx] = fp.predict(X_best.iloc[te_idx])
    fold_rmses.append(root_mean_squared_error(y.iloc[te_idx], oof_preds.iloc[te_idx]))

# Match the journal/Skore convention: mean fold RMSE and sample fold std.
# Recompute for THESE 32 features rather than printing the 29-feature score.
cv_mean = float(np.mean(fold_rmses))
cv_std = float(np.std(fold_rmses, ddof=1))

oof_errors  = (oof_preds - y).abs()
top50_idx   = oof_errors.nlargest(50).index
raw_context = X_raw.loc[top50_idx].copy()
raw_context["oof_pred"]    = oof_preds.loc[top50_idx]
raw_context["true_target"] = y.loc[top50_idx]
raw_context["abs_error"]   = oof_errors.loc[top50_idx]
raw_context["disease_duration"] = raw_context["age"] - raw_context["age_at_diagnosis"]

print(f"\nTop-50 error stats:")
print(f"  abs_error range: {raw_context['abs_error'].min():.1f} – {raw_context['abs_error'].max():.1f}  "
      f"mean {raw_context['abs_error'].mean():.1f}")

print(f"\n  Missingness rates (top-50 vs full train):")
for col in ["off", "on", "ledd", "gene", "time_since_intake_off"]:
    top_rate  = raw_context[col].isna().mean()
    full_rate = X_raw[col].isna().mean()
    print(f"    {col:<25} top-50: {top_rate:.2%}  full: {full_rate:.2%}  ratio {top_rate/(full_rate+1e-9):.1f}x")

print(f"\n  Cohort distribution (top-50):")
print(raw_context["cohort"].value_counts(normalize=True).round(3).to_string())
print(f"\n  Cohort distribution (full train):")
print(X_raw["cohort"].value_counts(normalize=True).round(3).to_string())

print(f"\n  Rater distribution top-5 (top-50):")
print(raw_context["rater_id"].value_counts().head(5).to_string())

print(f"\n  Disease duration stats (top-50):")
print(raw_context["disease_duration"].describe().round(2).to_string())
print(f"  Full train disease_duration mean: {(X_raw['age'] - X_raw['age_at_diagnosis']).mean():.2f}")

print(f"\n  time_since_intake_off stats hours (top-50):")
print(raw_context["time_since_intake_off"].describe().round(2).to_string())

print(f"\n  true_target (y) stats (top-50):")
print(raw_context["true_target"].describe().round(2).to_string())

print(f"\n  off stats (top-50 — measured values only):")
print(raw_context["off"].describe().round(2).to_string())

print("\n=== SUMMARY ===")
print(f"  Baseline (raw, 05/06):         RMSE {BASE_MEAN:.4f}")
print("  Feature config:                32 features (A+B+C+D, including all ON aggregates)")
print(f"  Grouped CV (lr={BEST_LR}, leaves={BEST_LEAVES}, l2={BEST_L2}): RMSE {cv_mean:.4f} +/- {cv_std:.4f}")
print(f"  Gain vs historical baseline:   {cv_mean - BASE_MEAN:+.4f}  ({(cv_mean-BASE_MEAN)/BASE_MEAN*100:+.1f}%)")
print(f"  Submission:                    {out_path}  ({len(submission)} rows)")
if not _hub_available:
    print("  Upload eligibility pending: no Hub EstimatorReport URL was obtained in this run.")
