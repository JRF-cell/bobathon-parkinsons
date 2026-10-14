# %% [markdown]
# # Experiment 02_ridge: Ridge linear model — numeric features
#
# **Date:** 2026-10-07
# **Goal:** Beat the dummy floor (RMSE 16.48) with a Ridge regression on the
#           8 numeric clinical columns. Sweep alpha in [0.01, 0.1, 1, 10, 100, 1000]
#           and keep the alpha with the lowest RMSE.
# **Splitter:** random 80/20 visit holdout (splitter=0.2) — same as 01_dummy
#               so results are directly comparable. ⚠ patient leakage is present;
#               grouped Ridge is evaluated in 04_hgbr.py (no 03 script exists).
# **Result:** filled in after the run.

# %%
import skore

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset

# %% [markdown]
# ## Hub login (requires .skore — skip gracefully if missing)

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
    print(f"Hub unavailable ({_hub_err}). Offline run - metrics and CSV will still be produced.")

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
print(f"Training set: {X.shape[0]} rows × {X.shape[1]} features")

# %% [markdown]
# ## Alpha sweep
#
# Try alpha in [0.01, 0.1, 1, 10, 100, 1000].
# For each alpha fit on the 80/20 holdout and record RMSE, MAE, R².

# %%
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

ALPHAS = [0.01, 0.1, 1, 10, 100, 1000]

results = []  # list of (alpha, rmse, mae, r2, report)

print(f"\n{'alpha':>8}  {'RMSE':>8}  {'MAE':>8}  {'R²':>8}")
print("-" * 42)

for alpha in ALPHAS:
    pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        Ridge(alpha=alpha),
    )
    report = skore.evaluate(pipeline, X, y, splitter=0.2)
    # splitter=0.2 → single EstimatorReport → metrics return plain floats
    rmse = float(report.metrics.rmse())
    mae  = float(report.metrics.mae())
    r2   = float(report.metrics.r2())
    results.append((alpha, rmse, mae, r2, report))
    print(f"{alpha:>8}  {rmse:>8.4f}  {mae:>8.4f}  {r2:>8.4f}")

# %%
best_alpha, best_rmse, best_mae, best_r2, best_report = min(results, key=lambda t: t[1])
print(f"\nBest alpha: {best_alpha}  →  RMSE {best_rmse:.4f}, MAE {best_mae:.4f}, R² {best_r2:.4f}")

# RMSE spread across alphas
rmses = [r[1] for r in results]
print(f"RMSE range across all alphas: {min(rmses):.4f} – {max(rmses):.4f}  (spread = {max(rmses)-min(rmses):.4f})")

# %% [markdown]
# ## Dummy vs best Ridge — side-by-side comparison

# %%
from sklearn.dummy import DummyRegressor

dummy = DummyRegressor(strategy="mean")
best_ridge = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=best_alpha),
)

comparison_report = skore.evaluate(
    {"dummy": dummy, "ridge": best_ridge},
    X,
    y,
    splitter=0.2,
)
print("\n=== Dummy vs Ridge comparison ===")
print(comparison_report.metrics.summarize())

# %% [markdown]
# ## Ridge coefficients
#
# Fit on the full training set to inspect coefficients.

# %%
inspector = make_pipeline(
    SimpleImputer(strategy="median"),
    Ridge(alpha=best_alpha),
)
inspector.fit(X, y)
ridge_model = inspector.named_steps["ridge"]
coef_series = pd.Series(ridge_model.coef_, index=FEATURE_COLS).sort_values(key=abs, ascending=False)
print("\n=== Ridge coefficients (sorted by |coef|) ===")
print(coef_series.to_string())

# %% [markdown]
# ## Push best Ridge report to hub (only when .skore is available)

# %%
if _hub_available and project is not None:
    project.put("02_ridge", best_report)
    print("\nReport '02_ridge' pushed to hub - paste the URL above into the Kaggle Submission Description.")
else:
    print("\nSkipping hub push - run 'python scripts/skore-agent' to authenticate, then re-run.")

# %% [markdown]
# ## Final model — fit on ALL training visits, predict X_test, write CSV

# %%
from pathlib import Path
from sklearn.base import clone

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test_full = load_test_dataset()
X_test = X_test_full[["Index"] + FEATURE_COLS]

final = clone(best_ridge)
final.fit(X, y)
predictions = final.predict(X_test[FEATURE_COLS])

submission = pd.DataFrame({"Index": X_test["Index"], "target": predictions})
submission_path = submissions_dir / "02_ridge.csv"
submission.to_csv(submission_path, index=False)
print(f"\nSubmission written to {submission_path}  ({len(submission)} rows)")
print("Upload this file on Kaggle and paste the hub URL in the Submission Description.")
