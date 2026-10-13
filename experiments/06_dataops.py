# %% [markdown]
# # Experiment 06_dataops: skrub DataOps — groups baked into the graph
#
# **Date:** 2026-10-07
# **Goal:** Reproduce experiment 05 using skrub DataOps so that the
#           GroupKFold split is baked into the computation graph and cannot
#           drift away from the data.
# **Splitter:** reads cv/groups from the DataOp graph (no explicit splitter=).
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
#
# Build the full "visits" table: X_train joined with y_train on Index.
# This is the single table the DataOp graph will receive.

# %%
import pandas as pd
import numpy as np

X_raw, y = load_dataset()

# Re-attach target so the DataOp can reach both X and y from one var.
visits = X_raw.copy()
visits["target"] = y.values

print(f"visits shape: {visits.shape}")
print(f"visits columns: {visits.columns.tolist()}")

# %% [markdown]
# ## Build the DataOp graph
#
# Key design decisions:
# 1. Drop BOTH "target" and "patient_id" with errors="ignore".
#    - "target" must be dropped from X (it is the label).
#    - "patient_id" is an identifier — test patients are unseen, memorising
#      IDs is leakage. errors="ignore" also makes the drop safe at predict
#      time when the test table has no "target" column.
#    (See journal/06_dataops.md for full explanation.)
# 2. groups = data["patient_id"] — extracted BEFORE the drop so it stays
#    attached to the correct rows inside the graph.
# 3. cv=GroupKFold(n_splits=5) and split_kwargs={"groups": groups} are
#    passed to mark_as_X so they live on the graph node, not a separate list.

# %%
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

data   = skrub.var("visits", visits)
groups = data["patient_id"]

X_op = (
    data
    .drop("target",     axis=1, errors="ignore")
    .drop("patient_id", axis=1, errors="ignore")
    .skb.mark_as_X(
        cv=GroupKFold(n_splits=5),
        split_kwargs={"groups": groups},
    )
)

y_op = data["target"].skb.mark_as_y()

pred = (
    X_op
    .skb.apply(skrub.TableVectorizer())
    .skb.apply(HistGradientBoostingRegressor(random_state=0), y=y_op)
)

print("DataOp graph built successfully.")

# %% [markdown]
# ## Inspect what X_op actually contains (preview)

# %%
print("\nX_op preview (first 3 rows):")
X_preview = X_op.skb.preview()
print(f"Columns in X_op: {X_preview.columns.tolist()}")
print(f"Shape: {X_preview.shape}")
assert "patient_id" not in X_preview.columns, "patient_id must not be in features!"
assert "target"     not in X_preview.columns, "target must not be in features!"
assert "Index"      not in X_preview.columns or X_preview.index.name == "Index", \
    "Index must not be a feature column (may be the DataFrame index)!"
print("Checks passed: patient_id and target are absent from X_op.")

# %% [markdown]
# ## Evaluate — splitter comes from the graph

# %%
print("\nRunning skore.evaluate(pred, data={'visits': visits}) — cv/groups from graph ...")
report = skore.evaluate(pred, data={"visits": visits})

rmse_df   = report.metrics.rmse()
rmse_mean = float(rmse_df.iloc[0, 0])
rmse_std  = float(rmse_df.iloc[0, 1])
print(f"\n06_dataops grouped-CV RMSE: {rmse_mean:.4f} +/- {rmse_std:.4f}")
print(report.metrics.summarize())

# %% [markdown]
# ## Compare with experiment 05

# %%
EXP05_RMSE = 7.5210
EXP05_STD  = 0.1336

diff = rmse_mean - EXP05_RMSE
print(f"\n=== Comparison with 05_tabular_pipeline ===")
print(f"05 RMSE: {EXP05_RMSE:.4f} +/- {EXP05_STD:.4f}")
print(f"06 RMSE: {rmse_mean:.4f} +/- {rmse_std:.4f}")
print(f"Difference (06 - 05): {diff:+.4f}")
if abs(diff) < 0.001:
    print("Scores are IDENTICAL (within rounding) — DataOp reproduces 05 exactly.")
elif abs(diff) < EXP05_STD:
    print("Scores differ by less than 1 fold-std — minor numerical variation, effectively the same.")
else:
    print(f"Scores differ by {abs(diff):.4f} (> 1 fold-std {EXP05_STD:.4f}) — investigate column set or fold assignment.")

# %% [markdown]
# ## Fold-level RMSE breakdown

# %%
print("\n=== Fold-level RMSE ===")
fold_rmses = []
for i, fold_rep in enumerate(report.reports_):
    fold_rmse = float(fold_rep.metrics.rmse())
    fold_rmses.append(fold_rmse)
    print(f"  Fold {i}: {fold_rmse:.4f}")
print(f"Mean: {sum(fold_rmses)/len(fold_rmses):.4f}  Std: {pd.Series(fold_rmses).std():.4f}")

# %% [markdown]
# ## Inspect TableVectorizer encoders (fitted on full training data)

# %%
print("\nFitting learner on full training set ...")
learner = pred.skb.make_learner()
learner.fit({"visits": visits})

# Inspect TableVectorizer encoders by fitting a standalone instance on X_op.
# X_op.skb.eval() returns the evaluated feature DataFrame.
X_for_tv = X_op.skb.eval()
tv_inspector = skrub.TableVectorizer()
tv_inspector.fit(X_for_tv, y)
print(f"\nTableVectorizer encoders (cardinality_threshold={tv_inspector.cardinality_threshold}):")
for col, enc in tv_inspector.transformers_.items():
    kind = [k for k, cols in tv_inspector.kind_to_columns_.items() if col in cols]
    print(f"  {col!r:30s} -> {type(enc).__name__:25s} ({kind[0] if kind else '?'})")

# %% [markdown]
# ## Learner smoke-test — predict on X_test
#
# This proves the learner works for §7 submission:
# X_test has no "target" column — the drop(errors="ignore") handles this.

# %%
print("\nSanity-checking learner.predict on X_test ...")
X_test_full = load_test_dataset()
# load_test_dataset keeps "Index" as a column; set it as the index to match
# the training data shape produced by load_dataset (which uses index_col="Index").
X_test_for_learner = X_test_full.set_index("Index")
# X_test does not have "target"; drop is safe because errors="ignore"
test_preds = learner.predict({"visits": X_test_for_learner})

assert len(test_preds) == 11013, f"Expected 11013 predictions, got {len(test_preds)}"
print(f"Predictions: {len(test_preds)} rows — shape matches X_test. OK.")
print(f"Prediction range: {test_preds.min():.2f} – {test_preds.max():.2f}  (mean {test_preds.mean():.2f})")

# %% [markdown]
# ## Hub push

# %%
if _hub_available and project is not None:
    project.put("06_dataops", report)
    print("\nReport '06_dataops' pushed to hub.")
else:
    print("\nSkipping hub push - run 'python scripts/skore-agent' to authenticate.")
