# Completion script: fit final model and generate submission for 07_features_optimized
# Runs the winning config (ensemble of specialist + HGBR_opt) on full training data,
# evaluates that SAME ensemble with grouped CV, writes the submission CSV, then
# pushes the matching ensemble reports to Skore Hub.

import numpy as np
import pandas as pd
import skore
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor
import skrub

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.features_optimized import add_features_optimized

# ---------- Selected configuration (see journal/07_features_optimized.md) ----------
BEST_LR, BEST_LEAVES, BEST_L2 = 0.05, 63, 1
# Each HGBR early-stops on a random 10 % row split, so its predictions depend on
# random_state. Averaging a few seeds reduces that variance. SEEDS = (0,)
# reproduces the original single-seed recipe (historical CV 4.2176 +/- 0.0686).
SEEDS = (0, 1, 2)

# ---------- Hub login ----------
_hub_available = False
project = None
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
def make_hgbr(lr=0.05, leaves=63, l2=1, seed=0):
    return HistGradientBoostingRegressor(
        learning_rate=lr, max_leaf_nodes=leaves, max_iter=1000,
        early_stopping=True, l2_regularization=l2, random_state=seed,
    )

def make_pipe(lr=0.05, leaves=63, l2=1, seed=0):
    return Pipeline([("tv", skrub.TableVectorizer()), ("hgbr", make_hgbr(lr, leaves, l2, seed))])

# ---------- Specialist regressor (Model C component) ----------
class SpecialistRegressor(RegressorMixin, BaseEstimator):
    """Route visits to one model trained on observed-OFF rows, one on missing-OFF rows."""

    def __init__(self, lr=0.05, leaves=63, l2=1, seed=0):
        self.lr = lr; self.leaves = leaves; self.l2 = l2; self.seed = seed

    def fit(self, X, y):
        miss = X["off"].isna().to_numpy()
        y = np.asarray(y)
        # Both slices hold thousands of rows in every fold, so the former
        # full-data fallback model was never used and has been removed.
        self.pipe_obs_  = make_pipe(self.lr, self.leaves, self.l2, self.seed).fit(X[~miss], y[~miss])
        self.pipe_miss_ = make_pipe(self.lr, self.leaves, self.l2, self.seed).fit(X[miss], y[miss])
        return self

    def predict(self, X):
        miss = X["off"].isna().to_numpy()
        preds = np.empty(len(X))
        if (~miss).any():
            preds[~miss] = self.pipe_obs_.predict(X[~miss])
        if miss.any():
            preds[miss] = self.pipe_miss_.predict(X[miss])
        return preds

# ---------- Ensemble (Model E): what is actually submitted ----------
class EnsembleRegressor(RegressorMixin, BaseEstimator):
    """Equal average of specialist (C) and HGBR_opt (B), over each seed, clipped to [0, 132].

    Being a scikit-learn estimator, it can be evaluated by skore exactly as submitted.
    """

    def __init__(self, lr=0.05, leaves=63, l2=1, seeds=(0,)):
        self.lr = lr; self.leaves = leaves; self.l2 = l2; self.seeds = seeds

    def fit(self, X, y):
        self.models_ = []
        for seed in self.seeds:
            self.models_.append(SpecialistRegressor(self.lr, self.leaves, self.l2, seed).fit(X, y))
            self.models_.append(make_pipe(self.lr, self.leaves, self.l2, seed).fit(X, y))
        return self

    def predict(self, X):
        preds = np.mean([model.predict(X) for model in self.models_], axis=0)
        return np.clip(preds, 0.0, 132.0)

ensemble = EnsembleRegressor(BEST_LR, BEST_LEAVES, BEST_L2, seeds=SEEDS)

# ---------- Grouped CV of the ensemble ----------
print(f"\nEvaluating the ensemble with grouped CV (seeds={SEEDS}) ...")
cv_report = skore.evaluate(ensemble, X_opt, y, splitter=cv_splits)
rmse_df = cv_report.metrics.rmse()
cv_mean, cv_std = float(rmse_df.iloc[0, 0]), float(rmse_df.iloc[0, 1])
print(f"Ensemble CV RMSE: {cv_mean:.4f} +/- {cv_std:.4f}")

# Last fold's EstimatorReport: a held-out report of the submitted ensemble
est_report = cv_report.reports_[-1]
print(f"Held-out (fold 4) RMSE: {float(est_report.metrics.rmse()):.4f}")

if _hub_available and project is not None:
    project.put("07_features_optimized_ensemble_cv", cv_report)
    print("Ensemble CV report '07_features_optimized_ensemble_cv' pushed to hub.")
    project.put("07_features_optimized_ensemble", est_report)
    print("Ensemble EstimatorReport '07_features_optimized_ensemble' pushed to hub - "
          "paste its URL in the Kaggle Submission Description.")
else:
    print("Skipping hub push - run 'python scripts/skore-agent' to authenticate.")

# ---------- Fit ensemble on ALL training data ----------
print("\nFitting the ensemble on full training data...")
final_model = clone(ensemble).fit(X_opt, y)

# ---------- Predict test set ----------
print("\nPredicting X_test...")
X_test_raw     = load_test_dataset()
X_test_indexed = X_test_raw.set_index("Index")
X_test_opt     = add_features_optimized(X_test_indexed)

# Align columns
X_test_final = X_test_opt[list(X_opt.columns)]
assert X_test_final.index.equals(X_test_indexed.index), "Test row order changed"

test_preds = final_model.predict(X_test_final)
assert np.isfinite(test_preds).all(), "Non-finite ensemble predictions"

print(f"Test rows: {len(test_preds)}")
print(f"Range: {test_preds.min():.2f} – {test_preds.max():.2f}  mean: {test_preds.mean():.2f}")

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
print(f"  Seeds:                      {SEEDS}")
print(f"  Ensemble CV RMSE:           {cv_mean:.4f} +/- {cv_std:.4f}")
print(f"  Historical single-seed CV:  4.2176 +/- 0.0686 (journal)")
print(f"  Submission:                 {out_path}")
print("="*70)
