# %% [markdown]
# # Experiment 01_dummy: Mean-predictor baseline
#
# **Date:** 2026-10-07
# **Goal:** Establish the RMSE floor by always predicting the training-set mean.
#           Any real model must beat this number.
# **Result:** filled in after the run.

# %%
import skore
from skore import login

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.evaluate import splitter
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_learner

# %% [markdown]
# ## Hub login
#
# Credentials are read from the nearest `.skore` file (written once
# by `scripts/install_skore.py`). The API key is exported to env vars
# so `login(mode="hub")` does not open a browser.

# %%
cfg = load_skore_credentials()
login(mode="hub")

# %% [markdown]
# ## Project
#
# Open the shared hub project. All experiment reports for this
# workspace live together under their experiment-stem keys so they
# can be compared across runs.

# %%
project = skore.Project(
    name="bobathon-esilv",
    mode="hub",
    workspace=cfg["workspace"],
)

# %% [markdown]
# ## Data and learner
#
# Load the training features and target, then build the unfit
# DummyRegressor (strategy="mean") — no features, just the mean.

# %%
X, y = load_dataset()
learner = build_learner()

# %% [markdown]
# ## Evaluate
#
# `splitter=0.2` is a random 80/20 visit-level holdout.
# skore returns an EstimatorReport with RMSE, MAE, R², and more.

# %%
report = skore.evaluate(learner, X, y, splitter=splitter)
report

# %% [markdown]
# ## Persist
#
# Push the report to the hub under the key `01_dummy`.
# Reusing this key in a future run would overwrite the stored report,
# so fork into a new experiment file if you want to keep both.

# %%
project.put("01_dummy", report)

# %% [markdown]
# ## Submit
#
# Refit on **all** training rows (no CV split), predict the test
# visits, and write `submissions/01_dummy.csv`.
# The Kaggle submission description must include the hub report URL
# printed above.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test = load_test_dataset()

final = build_learner()
final.fit(X, y)
final_predictions = final.predict(X_test.drop(columns=["Index"], errors="ignore"))

import pandas as pd  # noqa: E402

submission = pd.DataFrame({"Index": X_test["Index"], "target": final_predictions})
submission.to_csv(submissions_dir / "01_dummy.csv", index=False)
print(f"Submission written to submissions/01_dummy.csv ({len(submission)} rows)")
