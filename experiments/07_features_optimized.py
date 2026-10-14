# %% [markdown]
# # Experiment 07_features_optimized
#
# **Date:** 2026-10-08
# **Goal:** Substantially improve on 07_features (RMSE 4.532 ± 0.050) by:
#   1. Fixing age-based interpolation (was integer-index based)
#   2. Adding rich longitudinal features (slopes, ON/OFF ratios, distances)
#   3. Adding pharmacokinetic medication-exposure proxies
#   4. Comparing specialist regressors (missing-OFF vs observed-OFF)
#   5. Optionally trying CatBoostRegressor
# **Reference:** 07_features — RMSE 4.532 ± 0.050, Kaggle 4.274
# **Splitter:** GroupKFold(n_splits=5) on patient_id — same as 04-07

# %%
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
from parkinsons.features import add_features
from parkinsons.features_optimized import add_features_optimized

# %% [markdown]
# ## Hub login

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
X_raw, y = load_dataset()
print(f"X_raw shape: {X_raw.shape}")
print(f"Columns: {X_raw.columns.tolist()}")
print(f"Target stats: mean={y.mean():.2f}  std={y.std():.2f}  min={y.min():.0f}  max={y.max():.0f}")

# %% [markdown]
# ## Grouped CV splits — same as 07_features

# %%
groups = X_raw["patient_id"]
cv_splits = list(GroupKFold(n_splits=5).split(X_raw, y, groups=groups))
print(f"GroupKFold: 5 folds on {groups.nunique()} unique patients")

# %% [markdown]
# ## Reference: 07_features feature matrix
#
# Re-evaluate the 07_features best config (lr=0.03, leaves=63, l2=1)
# on the *same* cv_splits so the comparison table is fully apples-to-apples.

# %%
X_ref = add_features(X_raw)  # original features.py
REF_FEATURE_COLS = [
    c for c in X_ref.columns
    if c not in ("patient_id",)
]
X_ref = X_ref[REF_FEATURE_COLS]

print(f"\nReference (07_features) feature matrix: {X_ref.shape}")

# Best config from 07_features
REF_LR, REF_LEAVES, REF_L2 = 0.03, 63, 1

def make_hgbr(lr=0.03, leaves=63, l2=1):
    return HistGradientBoostingRegressor(
        learning_rate=lr,
        max_leaf_nodes=leaves,
        max_iter=1000,
        early_stopping=True,
        l2_regularization=l2,
        random_state=0,
    )

def make_pipe(lr=0.03, leaves=63, l2=1):
    return Pipeline([
        ("tv", skrub.TableVectorizer()),
        ("hgbr", make_hgbr(lr, leaves, l2)),
    ])

def cv_rmse_split(X, pipe, label, off_mask=None):
    """Run CV and return (mean, std, report).  Also compute RMSE split by
    off_missing if off_mask (boolean Series aligned to X.index) is supplied."""
    rep = skore.evaluate(pipe, X, y, splitter=cv_splits)
    df   = rep.metrics.rmse()
    mean = float(df.iloc[0, 0])
    std  = float(df.iloc[0, 1])

    # Manual fold-level OOF for split-by-off analysis
    oof = pd.Series(np.nan, index=X.index, dtype=float)
    for tr_idx, te_idx in cv_splits:
        p = Pipeline([
            ("tv", skrub.TableVectorizer()),
            ("hgbr", make_hgbr(
                pipe.named_steps["hgbr"].learning_rate,
                pipe.named_steps["hgbr"].max_leaf_nodes,
                pipe.named_steps["hgbr"].l2_regularization,
            )),
        ])
        p.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof.iloc[te_idx] = p.predict(X.iloc[te_idx])

    rmse_off_obs   = np.nan
    rmse_off_miss  = np.nan
    if off_mask is not None:
        obs_mask  = off_mask.reindex(X.index, fill_value=False)
        miss_mask = ~obs_mask
        if obs_mask.sum() > 0:
            rmse_off_obs  = float(np.sqrt(((oof[obs_mask]  - y[obs_mask])**2).mean()))
        if miss_mask.sum() > 0:
            rmse_off_miss = float(np.sqrt(((oof[miss_mask] - y[miss_mask])**2).mean()))

    print(f"  {label:<55} RMSE {mean:.4f} +/- {std:.4f}  "
          f"[OFF_obs={rmse_off_obs:.3f}  OFF_miss={rmse_off_miss:.3f}]")
    return mean, std, rep, oof

# OFF observed mask (True where off is not NaN)
off_obs_mask = X_raw["off"].notna()

print("\n--- Reference: 07_features best config ---")
ref_pipe = make_pipe(REF_LR, REF_LEAVES, REF_L2)
ref_mean, ref_std, ref_rep, ref_oof = cv_rmse_split(
    X_ref, ref_pipe, "07_features (lr=0.03/l=63/l2=1)", off_mask=off_obs_mask
)

# %% [markdown]
# ## Build optimized feature matrix

# %%
X_opt = add_features_optimized(X_raw)
print(f"\nOptimized feature matrix: {X_opt.shape}")
new_cols = [c for c in X_opt.columns if c not in X_ref.columns]
print(f"New columns ({len(new_cols)}): {new_cols}")

print(f"\nMissing values in new cols:")
print(X_opt[new_cols].isna().mean().round(3).sort_values(ascending=False).head(20).to_string())

# %% [markdown]
# ## Model A: HGBR + optimized features, same hyperparams as 07_features

# %%
print("\n=== Model A: HGBR + optimized features (same HPs) ===")
pipe_a = make_pipe(REF_LR, REF_LEAVES, REF_L2)
a_mean, a_std, rep_a, oof_a = cv_rmse_split(
    X_opt, pipe_a, "A: optimized features (same HP)", off_mask=off_obs_mask
)

# %% [markdown]
# ## Model B: HGBR + optimized features, light HP search

# %%
print("\n=== Model B: light HP search on optimized features ===")
from itertools import product as iproduct

tuning_results = []
for lr, leaves, l2 in iproduct([0.03, 0.05, 0.1], [31, 63, 127], [0, 1]):
    rep = skore.evaluate(make_pipe(lr, leaves, l2), X_opt, y, splitter=cv_splits)
    df   = rep.metrics.rmse()
    mean = float(df.iloc[0, 0])
    std  = float(df.iloc[0, 1])
    tuning_results.append((lr, leaves, l2, mean, std, rep))

tuning_results.sort(key=lambda r: r[3])
best_lr, best_leaves, best_l2, b_mean, b_std, rep_b = tuning_results[0]
print(f"  Best B: lr={best_lr}, leaves={best_leaves}, l2={best_l2}  =>  {b_mean:.4f} +/- {b_std:.4f}")

# Full split analysis for best B
pipe_b = make_pipe(best_lr, best_leaves, best_l2)
b_mean, b_std, rep_b, oof_b = cv_rmse_split(
    X_opt, pipe_b,
    f"B: optimized features (lr={best_lr}/l={best_leaves}/l2={best_l2})",
    off_mask=off_obs_mask,
)

# %% [markdown]
# ## Model C: Specialist regressors — separate model for OFF-missing vs OFF-observed visits

# %%
print("\n=== Model C: specialist regressors (missing-OFF vs observed-OFF) ===")

class SpecialistRegressor:
    """Fits two HGBR models: one for rows where off is NaN, one where off is observed.

    At predict time, routes each row to the right sub-model.
    Uses the same hyperparams for both sub-models.
    """

    def __init__(self, lr=0.03, leaves=63, l2=1, off_col_idx=None):
        self.lr = lr
        self.leaves = leaves
        self.l2 = l2
        self.off_col_idx = off_col_idx  # position of 'off' in the vectorized feature space

    def _split_mask(self, X_df):
        """Return boolean mask where off is NaN (needs missing-OFF model)."""
        if "off" in X_df.columns:
            return X_df["off"].isna()
        return pd.Series(False, index=X_df.index)

    def fit(self, X, y):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y = pd.Series(y) if not isinstance(y, pd.Series) else y

        miss_mask = self._split_mask(X)
        obs_mask  = ~miss_mask

        # Observed-OFF model
        self._pipe_obs = Pipeline([
            ("tv", skrub.TableVectorizer()),
            ("hgbr", make_hgbr(self.lr, self.leaves, self.l2)),
        ])
        if obs_mask.sum() > 10:
            self._pipe_obs.fit(X[obs_mask], y[obs_mask])
            self._has_obs = True
        else:
            self._has_obs = False

        # Missing-OFF model
        self._pipe_miss = Pipeline([
            ("tv", skrub.TableVectorizer()),
            ("hgbr", make_hgbr(self.lr, self.leaves, self.l2)),
        ])
        if miss_mask.sum() > 10:
            self._pipe_miss.fit(X[miss_mask], y[miss_mask])
            self._has_miss = True
        else:
            self._has_miss = False

        # Fallback: full model on all rows
        self._pipe_full = Pipeline([
            ("tv", skrub.TableVectorizer()),
            ("hgbr", make_hgbr(self.lr, self.leaves, self.l2)),
        ])
        self._pipe_full.fit(X, y)

        return self

    def predict(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        preds = np.full(len(X), np.nan)

        miss_mask = self._split_mask(X)
        obs_mask  = ~miss_mask

        if obs_mask.sum() > 0 and self._has_obs:
            preds[np.where(obs_mask)[0]] = self._pipe_obs.predict(X[obs_mask])
        elif obs_mask.sum() > 0:
            preds[np.where(obs_mask)[0]] = self._pipe_full.predict(X[obs_mask])

        if miss_mask.sum() > 0 and self._has_miss:
            preds[np.where(miss_mask)[0]] = self._pipe_miss.predict(X[miss_mask])
        elif miss_mask.sum() > 0:
            preds[np.where(miss_mask)[0]] = self._pipe_full.predict(X[miss_mask])

        # Any remaining NaN → fallback
        nan_mask = np.isnan(preds)
        if nan_mask.any():
            preds[nan_mask] = self._pipe_full.predict(X.iloc[np.where(nan_mask)[0]])

        return preds


# Manual CV for SpecialistRegressor (skore.evaluate needs a sklearn-compatible estimator)
oof_c = pd.Series(np.nan, index=X_opt.index, dtype=float)
fold_rmses_c = []
for tr_idx, te_idx in cv_splits:
    spec = SpecialistRegressor(lr=best_lr, leaves=best_leaves, l2=best_l2)
    spec.fit(X_opt.iloc[tr_idx], y.iloc[tr_idx])
    preds = spec.predict(X_opt.iloc[te_idx])
    preds = np.clip(preds, 0, 132)
    oof_c.iloc[te_idx] = preds
    fold_rmse = float(np.sqrt(((preds - y.iloc[te_idx].values)**2).mean()))
    fold_rmses_c.append(fold_rmse)

c_mean = float(np.mean(fold_rmses_c))
c_std  = float(np.std(fold_rmses_c))

rmse_c_obs  = float(np.sqrt(((oof_c[off_obs_mask]  - y[off_obs_mask])**2).mean()))
rmse_c_miss = float(np.sqrt(((oof_c[~off_obs_mask] - y[~off_obs_mask])**2).mean()))

print(f"  {'C: specialist (miss-OFF / obs-OFF)':<55} RMSE {c_mean:.4f} +/- {c_std:.4f}  "
      f"[OFF_obs={rmse_c_obs:.3f}  OFF_miss={rmse_c_miss:.3f}]")

# %% [markdown]
# ## Model D: CatBoostRegressor (if available)

# %%
print("\n=== Model D: CatBoostRegressor (if available) ===")
_catboost_available = False
d_mean, d_std = np.nan, np.nan
oof_d = None
try:
    from catboost import CatBoostRegressor as CBR

    cat_features = ["cohort", "gene", "rater_id"]  # categorical cols

    # Simple wrapper pipeline: CatBoost handles NaN and cats natively
    # so we do not need TableVectorizer
    def _cv_catboost(X_df, lr_cb=0.05, depth=8, iters=1000, label="CatBoost"):
        cat_idx = [i for i, c in enumerate(X_df.columns) if c in cat_features]
        _oof = pd.Series(np.nan, index=X_df.index, dtype=float)
        fold_rmses = []
        for tr_idx, te_idx in cv_splits:
            X_tr = X_df.iloc[tr_idx]
            X_te = X_df.iloc[te_idx]
            y_tr = y.iloc[tr_idx]
            y_te = y.iloc[te_idx]

            # CatBoost needs string NaN for categoricals
            for c in cat_features:
                if c in X_tr.columns:
                    X_tr = X_tr.copy()
                    X_te = X_te.copy()
                    X_tr[c] = X_tr[c].fillna("__missing__").astype(str)
                    X_te[c] = X_te[c].fillna("__missing__").astype(str)

            cb = CBR(
                iterations=iters,
                learning_rate=lr_cb,
                depth=depth,
                cat_features=cat_idx,
                random_seed=0,
                verbose=0,
            )
            cb.fit(X_tr.values, y_tr.values)
            preds = cb.predict(X_te.values)
            preds = np.clip(preds, 0, 132)
            _oof.iloc[te_idx] = preds
            fold_rmses.append(float(np.sqrt(((preds - y_te.values)**2).mean())))

        _mean = float(np.mean(fold_rmses))
        _std  = float(np.std(fold_rmses))
        _obs  = float(np.sqrt(((_oof[off_obs_mask]  - y[off_obs_mask])**2).mean()))
        _miss = float(np.sqrt(((_oof[~off_obs_mask] - y[~off_obs_mask])**2).mean()))
        print(f"  {label:<55} RMSE {_mean:.4f} +/- {_std:.4f}  "
              f"[OFF_obs={_obs:.3f}  OFF_miss={_miss:.3f}]")
        return _mean, _std, _oof

    _catboost_available = True
    d_mean, d_std, oof_d = _cv_catboost(X_opt, lr_cb=0.05, depth=8, label="D: CatBoost (opt features)")

except ImportError:
    print("  CatBoost not installed — skipping Model D.")
except Exception as _cb_err:
    print(f"  CatBoost error ({_cb_err}) — skipping Model D.")

# %% [markdown]
# ## Model E: Ensemble of best complementary models

# %%
print("\n=== Model E: ensemble (average best HGBR + specialist or CatBoost) ===")

# Collect OOF predictions from the best candidates
candidates = {
    "hgbr_opt": (b_mean, oof_b),
    "specialist": (c_mean, oof_c),
}
if oof_d is not None and not np.isnan(d_mean):
    candidates["catboost"] = (d_mean, oof_d)

# Average the two best (by CV RMSE)
sorted_candidates = sorted(candidates.items(), key=lambda kv: kv[1][0])
best2 = sorted_candidates[:2]
print(f"  Ensembling: {[k for k, _ in best2]}")

ensemble_oof = sum(v[1] for _, v in best2) / len(best2)
rmse_e = float(np.sqrt(((ensemble_oof - y)**2).mean()))
rmse_e_obs  = float(np.sqrt(((ensemble_oof[off_obs_mask]  - y[off_obs_mask])**2).mean()))
rmse_e_miss = float(np.sqrt(((ensemble_oof[~off_obs_mask] - y[~off_obs_mask])**2).mean()))

e_fold_rmses = []
for tr_idx, te_idx in cv_splits:
    fold_oof_vals = [oof for _, (_, oof) in best2]
    fold_avg = sum(o.iloc[te_idx].values for o in fold_oof_vals) / len(fold_oof_vals)
    e_fold_rmses.append(float(np.sqrt(((fold_avg - y.iloc[te_idx].values)**2).mean())))

e_mean = float(np.mean(e_fold_rmses))
e_std  = float(np.std(e_fold_rmses))
print(f"  {'E: ensemble':<55} RMSE {e_mean:.4f} +/- {e_std:.4f}  "
      f"[OFF_obs={rmse_e_obs:.3f}  OFF_miss={rmse_e_miss:.3f}]")

# %% [markdown]
# ## Comparison table

# %%
print("\n" + "="*100)
print("COMPARISON TABLE")
print("="*100)
print(f"{'Model':<55} {'RMSE mean':>10} {'RMSE std':>10} {'vs ref':>8}  {'OFF_obs':>8}  {'OFF_miss':>9}")
print("-"*100)

off_obs_mask_aligned = off_obs_mask.reindex(X_opt.index, fill_value=False)

def get_split_rmse(oof_series):
    obs_rmse  = float(np.sqrt(((oof_series[off_obs_mask]  - y[off_obs_mask])**2).mean()))
    miss_rmse = float(np.sqrt(((oof_series[~off_obs_mask] - y[~off_obs_mask])**2).mean()))
    return obs_rmse, miss_rmse

ref_obs, ref_miss = get_split_rmse(ref_oof)
a_obs, a_miss     = get_split_rmse(oof_a)
b_obs, b_miss     = get_split_rmse(oof_b)

rows = [
    ("07_features reference",          ref_mean, ref_std,  ref_obs,  ref_miss, ref_oof),
    ("A: optimized features same HP",  a_mean,   a_std,    a_obs,    a_miss,   oof_a),
    ("B: optimized features best HP",  b_mean,   b_std,    b_obs,    b_miss,   oof_b),
    ("C: specialist miss/obs",         c_mean,   c_std,    rmse_c_obs, rmse_c_miss, oof_c),
    ("E: ensemble best2",              e_mean,   e_std,    rmse_e_obs, rmse_e_miss, ensemble_oof),
]
if oof_d is not None:
    d_obs, d_miss = get_split_rmse(oof_d)
    rows.insert(4, ("D: CatBoost opt features", d_mean, d_std, d_obs, d_miss, oof_d))

for name, mean, std, obs_r, miss_r, _ in rows:
    delta = mean - ref_mean
    print(f"  {name:<53} {mean:>10.4f} {std:>10.4f} {delta:>+8.4f}  {obs_r:>8.3f}  {miss_r:>9.3f}")

# %% [markdown]
# ## Select best model

# %%
# Determine best by grouped-CV RMSE mean
best_rows = sorted(rows, key=lambda r: r[1])
best_name, best_mean, best_std, best_obs, best_miss, best_oof = best_rows[0]
print(f"\nBest model: '{best_name}'  RMSE {best_mean:.4f} +/- {best_std:.4f}")
print(f"  vs reference 07_features: {best_mean - ref_mean:+.4f}  ({(best_mean - ref_mean)/ref_mean*100:+.1f}%)")

# Identify which config won so we can fit it on all training data
# We map back to config letter
_config_oof_map = {r[0]: r for r in rows}
_best_config = best_name

# %% [markdown]
# ## Fit final model on full training data

# %%
print("\n=== Fit final model on all training data ===")

def fit_final_model(config_name):
    """Fit the winning model on all training rows and return the fitted object."""
    # Candidate keys used by best2 must resolve to the same evaluated models.
    config_name = {"hgbr_opt": "B:", "specialist": "C:", "catboost": "D:"}.get(
        config_name, config_name
    )
    if config_name.startswith("A:"):
        pipe = make_pipe(REF_LR, REF_LEAVES, REF_L2)
        pipe.fit(X_opt, y)
        return pipe, X_opt
    elif config_name.startswith("B:"):
        pipe = make_pipe(best_lr, best_leaves, best_l2)
        pipe.fit(X_opt, y)
        return pipe, X_opt
    elif config_name.startswith("C:"):
        spec = SpecialistRegressor(lr=best_lr, leaves=best_leaves, l2=best_l2)
        spec.fit(X_opt, y)
        return spec, X_opt
    elif config_name.startswith("D:"):
        cat_features_local = ["cohort", "gene", "rater_id"]
        from catboost import CatBoostRegressor as CBR
        X_cb = X_opt.copy()
        for c in cat_features_local:
            if c in X_cb.columns:
                X_cb[c] = X_cb[c].fillna("__missing__").astype(str)
        cat_idx = [i for i, c in enumerate(X_cb.columns) if c in cat_features_local]
        cb = CBR(iterations=1000, learning_rate=0.05, depth=8, cat_features=cat_idx,
                 random_seed=0, verbose=0)
        cb.fit(X_cb.values, y.values)
        return cb, X_cb
    elif config_name.startswith("E:"):
        # Ensemble: fit both sub-models
        models = []
        X_sets = []
        for cname, _ in best2:
            m, x = fit_final_model(cname)
            models.append(m)
            X_sets.append(x)
        return ("ensemble", models, X_sets), X_opt
    elif config_name == "07_features reference":
        pipe = make_pipe(REF_LR, REF_LEAVES, REF_L2)
        pipe.fit(X_ref, y)
        return pipe, X_ref
    else:
        raise ValueError(f"Unknown model configuration: {config_name}")

final_model, X_final_train = fit_final_model(_best_config)
print(f"Final model fitted on {len(X_final_train)} rows.")

# %% [markdown]
# ## Component B EstimatorReport (last fold held out, not an ensemble report)

# %%
# Build EstimatorReport using the last CV fold as a held-out test set.
# Only possible for sklearn-compatible estimators (not ensemble or SpecialistRegressor).
print("\n=== EstimatorReport (last fold held-out) ===")

train_idx_last, test_idx_last = cv_splits[-1]
X_held = X_opt.iloc[test_idx_last]
y_held = y.iloc[test_idx_last]
X_train_last = X_opt.iloc[train_idx_last]
y_train_last  = y.iloc[train_idx_last]

# Use the best sklearn-compatible pipe for the EstimatorReport
# (Model B is always sklearn-compatible)
est_pipe_for_report = make_pipe(best_lr, best_leaves, best_l2)

est_report = skore.EstimatorReport(
    est_pipe_for_report,
    X_train=X_train_last,
    y_train=y_train_last,
    X_test=X_held,
    y_test=y_held,
)

held_rmse = float(np.sqrt(((est_report.estimator_.predict(X_held) - y_held.values)**2).mean()))
print(f"EstimatorReport held-out RMSE (fold 4): {held_rmse:.4f}")

# %% [markdown]
# ## Push to Skore Hub

# %%
if _hub_available and project is not None:
    # CV report for best model B (always a skore CrossValReport)
    project.put("07_features_optimized_component_cv", rep_b)
    print("Component CV report '07_features_optimized_component_cv' pushed.")

    project.put("07_features_optimized_component", est_report)
    print("Component EstimatorReport '07_features_optimized_component' pushed.")

    print("The URLs printed by project.put describe HGBR component B, not the selected ensemble.")
    print("An ensemble submission needs a report evaluating that same ensemble.")
else:
    print("Hub unavailable — not pushing.")

# %% [markdown]
# ## Predict X_test and write submission

# %%
X_test_raw = load_test_dataset()
X_test_indexed = X_test_raw.set_index("Index")
X_test_opt = add_features_optimized(X_test_indexed)

print(f"\nX_test optimized shape: {X_test_opt.shape}")
print(f"Training cols: {X_opt.shape[1]}  Test cols: {X_test_opt.shape[1]}")

# Require the fitted feature schema and preserve the competition row order.
# A winning reference model must receive the original feature calculations.
X_test_features = (add_features(X_test_indexed)
                   if _best_config == "07_features reference" else X_test_opt)
final_cols = list(X_final_train.columns)
X_test_final = X_test_features[final_cols]
assert X_test_final.index.equals(X_test_indexed.index), "Test row order changed"


def predict_component(model, features):
    # CatBoost used explicit string categories during CV; repeat at inference.
    if type(model).__module__.startswith("catboost"):
        features = features.copy()
        for col in ["cohort", "gene", "rater_id"]:
            features[col] = features[col].fillna("__missing__").astype(str)
        return model.predict(features.values)
    return model.predict(features)

# Generate predictions from the winning model
if isinstance(final_model, tuple) and final_model[0] == "ensemble":
    # Ensemble: average predictions of sub-models
    _, sub_models, sub_X_sets = final_model
    test_preds_list = []
    for sm, _ in zip(sub_models, sub_X_sets):
        tp = predict_component(sm, X_test_final)
        test_preds_list.append(tp)
    test_preds = np.mean(test_preds_list, axis=0)
else:
    test_preds = predict_component(final_model, X_test_final)

# Reject invalid outputs before clipping could conceal infinite predictions.
assert np.isfinite(test_preds).all(), "Non-finite model predictions"
# Clip to valid UPDRS domain [0, 132]
test_preds = np.clip(test_preds, 0.0, 132.0)

print(f"Test predictions: {len(test_preds)} rows")
print(f"Range: {test_preds.min():.2f} – {test_preds.max():.2f}  mean: {test_preds.mean():.2f}")
print(f"NaN count: {np.isnan(test_preds).sum()}")

sample = pd.read_csv(PROJECT_ROOT / "data" / "sample_submission.csv")
assert len(test_preds) == len(sample) == len(X_test_indexed)
assert X_test_indexed.index.is_unique
assert np.array_equal(X_test_indexed.index, sample["Index"])
assert sample.columns.tolist() == ["Index", "target"]

submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

submission = pd.DataFrame({
    "Index": X_test_indexed.index,
    "target": test_preds,
})
out_path = submissions_dir / "07_features_optimized.csv"
submission.to_csv(out_path, index=False)
print(f"\nSubmission written: {out_path}  ({len(submission)} rows)")
print(submission.head())

# %% [markdown]
# ## Summary

# %%
print("\n" + "="*70)
print("EXPERIMENT 07_features_optimized — SUMMARY")
print("="*70)
print(f"  Reference (07_features):  RMSE {ref_mean:.4f} +/- {ref_std:.4f}")
print(f"  Best new config:          '{best_name}'")
print(f"                            RMSE {best_mean:.4f} +/- {best_std:.4f}  "
      f"(delta {best_mean - ref_mean:+.4f}, {(best_mean - ref_mean)/ref_mean*100:+.1f}%)")
print(f"  OFF-observed RMSE:        {best_obs:.4f}")
print(f"  OFF-missing RMSE:         {best_miss:.4f}")
print(f"  Submission:               submissions/07_features_optimized.csv ({len(submission)} rows)")
if _hub_available:
    print("  Hub: component-B reports only; see the actual URLs printed above.")
print("="*70)
