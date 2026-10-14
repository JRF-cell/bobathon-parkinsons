# Completion script: fit final model and generate submission for 07_features_optimized
# Runs the winning config (ensemble of specialist + HGBR_opt) on full training data
# and writes the submission CSV, then pushes to Skore Hub.

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skore
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
import skrub

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.features_optimized import add_features_optimized

# Historical journal results below are not recomputed ensemble metrics.
# This helper preserves the submitted recipe: equal average of C and B,
# both at learning_rate=0.05. The main experiment can select other candidates.
# ---------- fixed results from the run above ----------
BEST_LR, BEST_LEAVES, BEST_L2 = 0.05, 63, 1
REF_MEAN    = 4.4720
BEST_MEAN   = 4.2176
BEST_STD    = 0.0686
BEST_OBS    = 3.682
BEST_MISS   = 4.852
BEST_CONFIG = "E: ensemble best2"

# ---------- Hub login ----------
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
    print(f"Hub unavailable: {e}")

# ---------- Data ----------
X_raw, y = load_dataset()
groups = X_raw["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X_raw, y, groups=groups))

X_opt = add_features_optimized(X_raw)
print(f"Optimized feature matrix: {X_opt.shape}")

# ---------- Helper ----------
def make_hgbr(lr=0.05, leaves=63, l2=1):
    return HistGradientBoostingRegressor(
        learning_rate=lr, max_leaf_nodes=leaves, max_iter=1000,
        early_stopping=True, l2_regularization=l2, random_state=0,
    )

def make_pipe(lr=0.05, leaves=63, l2=1):
    return Pipeline([("tv", skrub.TableVectorizer()), ("hgbr", make_hgbr(lr, leaves, l2))])

# ---------- Specialist regressor (Model C component) ----------
class SpecialistRegressor:
    def __init__(self, lr=0.05, leaves=63, l2=1):
        self.lr = lr; self.leaves = leaves; self.l2 = l2

    def _split_mask(self, X_df):
        if "off" in X_df.columns:
            return X_df["off"].isna()
        return pd.Series(False, index=X_df.index)

    def fit(self, X, y):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y = y if isinstance(y, pd.Series) else pd.Series(y)
        miss_mask = self._split_mask(X)
        obs_mask  = ~miss_mask
        self._pipe_obs  = make_pipe(self.lr, self.leaves, self.l2)
        self._pipe_miss = make_pipe(self.lr, self.leaves, self.l2)
        self._pipe_full = make_pipe(self.lr, self.leaves, self.l2)
        self._has_obs  = obs_mask.sum() > 10
        self._has_miss = miss_mask.sum() > 10
        if self._has_obs:  self._pipe_obs.fit(X[obs_mask],  y[obs_mask])
        if self._has_miss: self._pipe_miss.fit(X[miss_mask], y[miss_mask])
        self._pipe_full.fit(X, y)
        return self

    def predict(self, X):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        preds = np.full(len(X), np.nan)
        miss_mask = self._split_mask(X).values
        obs_mask  = ~miss_mask
        if obs_mask.sum() > 0:
            src = self._pipe_obs if self._has_obs else self._pipe_full
            preds[obs_mask]  = src.predict(X.iloc[np.where(obs_mask)[0]])
        if miss_mask.sum() > 0:
            src = self._pipe_miss if self._has_miss else self._pipe_full
            preds[miss_mask] = src.predict(X.iloc[np.where(miss_mask)[0]])
        nan_mask = np.isnan(preds)
        if nan_mask.any():
            preds[nan_mask] = self._pipe_full.predict(X.iloc[np.where(nan_mask)[0]])
        return preds

# ---------- Fit ensemble sub-models on ALL training data ----------
print("\nFitting specialist model on full training data...")
spec = SpecialistRegressor(lr=BEST_LR, leaves=BEST_LEAVES, l2=BEST_L2)
spec.fit(X_opt, y)

print("Fitting HGBR (opt) model on full training data...")
pipe_hgbr = make_pipe(BEST_LR, BEST_LEAVES, BEST_L2)
pipe_hgbr.fit(X_opt, y)
print("Both models fitted.")

# ---------- EstimatorReport for Skore Hub ----------
print("\nBuilding EstimatorReport (last fold held-out)...")
train_idx, test_idx = cv_splits[-1]
est_pipe_unfitted = make_pipe(BEST_LR, BEST_LEAVES, BEST_L2)
est_report = skore.EstimatorReport(
    est_pipe_unfitted,
    X_train=X_opt.iloc[train_idx],
    y_train=y.iloc[train_idx],
    X_test=X_opt.iloc[test_idx],
    y_test=y.iloc[test_idx],
)
# Compute held-out RMSE from the fitted estimator inside the report
_inner_est = est_report.estimator_
held_preds = _inner_est.predict(X_opt.iloc[test_idx])
held_rmse  = float(np.sqrt(((held_preds - y.iloc[test_idx].values)**2).mean()))
print(f"Held-out (fold 4) RMSE: {held_rmse:.4f}")

# Build a quick CV report for Model B to push
print("\nBuilding CV report for Model B (HGBR optimized features)...")
rep_b = skore.evaluate(make_pipe(BEST_LR, BEST_LEAVES, BEST_L2), X_opt, y, splitter=cv_splits)
df_b  = rep_b.metrics.rmse()
print(f"CV RMSE: {float(df_b.iloc[0, 0]):.4f} +/- {float(df_b.iloc[0, 1]):.4f}")

if _hub_available and project is not None:
    project.put("07_features_optimized_component_cv", rep_b)
    print("Component CV report '07_features_optimized_component_cv' pushed to hub.")
    project.put("07_features_optimized_component", est_report)
    print("Component EstimatorReport '07_features_optimized_component' pushed to hub.")
    print("Use the actual URLs printed by project.put; these evaluate component B only.")
    print("A matching ensemble report is still needed for a new ensemble submission.")
    ESTIMATOR_URL = "See project.put output (component B only, not ensemble)"
else:
    print("Hub unavailable.")
    ESTIMATOR_URL = "N/A"

# ---------- Predict test set ----------
print("\nPredicting X_test...")
X_test_raw     = load_test_dataset()
X_test_indexed = X_test_raw.set_index("Index")
X_test_opt     = add_features_optimized(X_test_indexed)

# Align columns
final_cols = list(X_opt.columns)
X_test_final = X_test_opt[final_cols]
assert X_test_final.index.equals(X_test_indexed.index), "Test row order changed"

# Ensemble average
preds_spec = spec.predict(X_test_final)
preds_hgbr = pipe_hgbr.predict(X_test_final)
test_preds  = (preds_spec + preds_hgbr) / 2.0
assert np.isfinite(test_preds).all(), "Non-finite ensemble predictions"
test_preds  = np.clip(test_preds, 0.0, 132.0)

print(f"Test rows: {len(test_preds)}")
print(f"Range: {test_preds.min():.2f} – {test_preds.max():.2f}  mean: {test_preds.mean():.2f}")
print(f"NaN count: {np.isnan(test_preds).sum()}")

sample = pd.read_csv(PROJECT_ROOT / "data" / "sample_submission.csv")
assert len(test_preds) == len(sample) == len(X_test_indexed)
assert X_test_indexed.index.is_unique
assert np.array_equal(X_test_indexed.index, sample["Index"])
assert sample.columns.tolist() == ["Index", "target"]

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)
submission = pd.DataFrame({"Index": X_test_indexed.index, "target": test_preds})
out_path = submissions_dir / "07_features_optimized.csv"
submission.to_csv(out_path, index=False)
print(f"\nSubmission written: {out_path}  ({len(submission)} rows)")
print(submission.head(3))

# ---------- Final summary ----------
print("\n" + "="*70)
print("07_features_optimized — FINAL SUMMARY")
print("="*70)
print(f"  Reference (07_features):    RMSE {REF_MEAN:.4f}")
print(f"  Historical ensemble CV:     RMSE {BEST_MEAN:.4f} +/- {BEST_STD:.4f}")
print(f"  Improvement:                {BEST_MEAN - REF_MEAN:+.4f}  ({(BEST_MEAN - REF_MEAN)/REF_MEAN*100:+.1f}%)")
print(f"  Historical OFF-obs RMSE:               {BEST_OBS:.4f}")
print(f"  Historical OFF-miss RMSE:              {BEST_MISS:.4f}")
print(f"  Submission:                 {out_path}")
print(f"  Skore URL:                  {ESTIMATOR_URL}")
print("="*70)
