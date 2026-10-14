"""Optimized feature engineering for experiment 07_features_optimized.

Fixes and extensions over ``features.py``:

* Age-based interpolation instead of integer-index interpolation.
* Distance to nearest observed ON/OFF visit (in years).
* Patient-level counts of observed ON/OFF measurements.
* Huber-robust progression slopes and OLS slopes for both ON and OFF.
* ON/OFF differences and ratios at each visit.
* Medication-exposure proxy features (pharmacokinetic absorption-decay
  curves with several time constants).
* Missingness descriptors for OFF availability near each visit.

All features are derived **only from X**, never from the target.
GroupKFold on ``patient_id`` is safe: per-patient aggregates use
only the patient's own rows in whichever split they belong to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import siegelslopes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_features_optimized(X: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *X* with all optimised engineered features appended.

    Parameters
    ----------
    X:
        Raw feature matrix.  Must contain at minimum: ``patient_id``,
        ``age``, ``age_at_diagnosis``, ``off``, ``on``, ``ledd``,
        ``gene``, ``time_since_intake_on``, ``time_since_intake_off``.
        ``patient_id`` is used only for grouping and is *not* present in
        the returned DataFrame.

    Returns
    -------
    pd.DataFrame
        Copy of *X* with extra columns; ``patient_id`` dropped.
    """
    X = X.copy()

    # Sort once by patient then age; restore original order at the very end.
    original_order = X.index
    X = X.sort_values(["patient_id", "age"])

    # ------------------------------------------------------------------ #
    # Group A — temporal features                                         #
    # ------------------------------------------------------------------ #
    X["disease_duration"] = X["age"] - X["age_at_diagnosis"]

    X["years_since_first_visit"] = X["age"] - X.groupby("patient_id")["age"].transform(
        "min"
    )

    X["visit_rank"] = (
        X.groupby("patient_id")["age"].rank(method="first").astype(float) - 1.0
    )

    X["n_visits"] = X.groupby("patient_id")["age"].transform("count").astype(float)

    # ------------------------------------------------------------------ #
    # Group B — corrected age-based trajectory features                   #
    # ------------------------------------------------------------------ #

    # Per-patient aggregates of off and on
    for col in ("off", "on"):
        for agg in ("mean", "median", "min", "max"):
            X[f"{col}_{agg}"] = X.groupby("patient_id")[col].transform(agg)

    # Count of observed (non-NaN) measurements per patient
    X["off_n_obs"] = X.groupby("patient_id")["off"].transform(
        lambda s: s.notna().sum()
    ).astype(float)
    X["on_n_obs"] = X.groupby("patient_id")["on"].transform(
        lambda s: s.notna().sum()
    ).astype(float)

    # Lag-1 / lead-1 off and on within patient (already sorted by age)
    X["off_prev"] = X.groupby("patient_id")["off"].shift(1)
    X["off_next"] = X.groupby("patient_id")["off"].shift(-1)
    X["on_prev"] = X.groupby("patient_id")["on"].shift(1)
    X["on_next"] = X.groupby("patient_id")["on"].shift(-1)

    # Age-based linear interpolation of off and on within each patient.
    # FIX: use the actual ``age`` column as the numeric index for interpolation,
    # not the integer row index.  This correctly maps between age gaps.
    off_interp = X["off"].copy().astype(float)
    on_interp = X["on"].copy().astype(float)

    for pid, grp in X.groupby("patient_id"):
        # OFF interpolation
        if grp["off"].notna().sum() >= 2:
            s = grp["off"].copy()
            # Set age as the index so interpolate(method='index') uses years
            s.index = grp["age"]
            s_interp = s.interpolate(method="index")
            # Map back to original DataFrame index
            off_interp.loc[grp.index] = s_interp.values

        # ON interpolation
        if grp["on"].notna().sum() >= 2:
            s = grp["on"].copy()
            s.index = grp["age"]
            s_interp = s.interpolate(method="index")
            on_interp.loc[grp.index] = s_interp.values

    X["off_interp"] = off_interp
    X["on_interp"] = on_interp

    # Distance in years to the nearest visit where off is observed
    off_dist = pd.Series(np.nan, index=X.index)
    on_dist = pd.Series(np.nan, index=X.index)

    for pid, grp in X.groupby("patient_id"):
        off_ages = grp.loc[grp["off"].notna(), "age"].values
        on_ages = grp.loc[grp["on"].notna(), "age"].values

        if len(off_ages) > 0:
            dists = np.abs(grp["age"].values[:, None] - off_ages[None, :]).min(axis=1)
            off_dist.loc[grp.index] = dists
        if len(on_ages) > 0:
            dists = np.abs(grp["age"].values[:, None] - on_ages[None, :]).min(axis=1)
            on_dist.loc[grp.index] = dists

    X["off_dist_nearest_obs"] = off_dist
    X["on_dist_nearest_obs"] = on_dist

    # OLS slope of off ~ age and on ~ age per patient
    off_slope = pd.Series(np.nan, index=X.index)
    on_slope = pd.Series(np.nan, index=X.index)
    off_slope_robust = pd.Series(np.nan, index=X.index)

    for pid, grp in X.groupby("patient_id"):
        valid_off = grp[["age", "off"]].dropna()
        if len(valid_off) >= 2:
            slope = float(np.polyfit(valid_off["age"], valid_off["off"], 1)[0])
            off_slope.loc[grp.index] = slope
            # Siegel robust slope (resistant to outlier ratings)
            try:
                res = siegelslopes(valid_off["off"].values, valid_off["age"].values)
                off_slope_robust.loc[grp.index] = float(res.slope)
            except Exception:
                off_slope_robust.loc[grp.index] = slope

        valid_on = grp[["age", "on"]].dropna()
        if len(valid_on) >= 2:
            slope = float(np.polyfit(valid_on["age"], valid_on["on"], 1)[0])
            on_slope.loc[grp.index] = slope

    X["off_slope"] = off_slope
    X["on_slope"] = on_slope
    X["off_slope_robust"] = off_slope_robust

    # ON/OFF difference and ratio at visit level
    # Use interpolated values so NaN-off visits still get an estimate.
    X["on_off_diff"] = X["on_interp"] - X["off_interp"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            X["off_interp"].notna() & (X["off_interp"] > 0),
            X["on_interp"] / X["off_interp"],
            np.nan,
        )
    X["on_off_ratio"] = ratio

    # Per-patient mean of on_off_diff (improvement across visits)
    X["on_off_diff_mean"] = X.groupby("patient_id")["on_off_diff"].transform("mean")

    # ------------------------------------------------------------------ #
    # Group C — pharmacokinetic medication-exposure proxies               #
    # ------------------------------------------------------------------ #
    # In the generative model: fast absorption → peak, then exponential
    # decay. We approximate with:
    #   absorbed(t) = ledd × [1 − exp(−t / τ_abs)]   for t = time_since_intake_on
    #   residual(t) = ledd × exp(−t / τ_dec)           for t = time_since_intake_off
    #
    # We try three half-lives / time-constants per function and add them
    # all as features; the HGBR will pick the most informative ones.

    ledd = X["ledd"].fillna(0.0)  # 0 → no drug effect when LEDD missing
    ledd_flag = X["ledd"].isna().astype(float)

    t_on = X["time_since_intake_on"].values.astype(float)   # hours, may be NaN
    t_off = X["time_since_intake_off"].values.astype(float)  # hours, may be NaN

    # Absorption-decay (ON score context): drug effect at time of ON measurement
    # τ_abs controls rise to peak; typical levodopa Tmax 30–90 min.
    for tau_abs_h in [0.5, 1.0, 2.0]:
        col = f"ledd_absorbed_tau{tau_abs_h:.1f}"
        absorbed = ledd.values * (1 - np.exp(-t_on / tau_abs_h))
        absorbed = np.where(np.isnan(t_on), np.nan, absorbed)
        X[col] = absorbed

    # Exponential decay (OFF score context): residual effect at time of OFF
    # τ_dec controls half-life; levodopa plasma t½ ~60–90 min but clinical
    # motor effect lasts 3–5 h.
    for tau_dec_h in [1.5, 3.0, 5.0]:
        col = f"ledd_residual_tau{tau_dec_h:.1f}"
        residual = ledd.values * np.exp(-t_off / tau_dec_h)
        residual = np.where(np.isnan(t_off), np.nan, residual)
        X[col] = residual

    # Combined: residual at time of OFF assessment relative to LEDD
    # Gives a normalised "how much drug is still active" fraction
    for tau_dec_h in [1.5, 3.0, 5.0]:
        col = f"ledd_frac_remaining_tau{tau_dec_h:.1f}"
        frac = np.exp(-t_off / tau_dec_h)
        frac = np.where(np.isnan(t_off), np.nan, frac)
        X[col] = frac

    # Interaction: measured ON score × absorption fraction (corrected ON)
    for tau_abs_h in [0.5, 1.0, 2.0]:
        absorbed_col = f"ledd_absorbed_tau{tau_abs_h:.1f}"
        col = f"on_x_absorbed_tau{tau_abs_h:.1f}"
        X[col] = X["on"] * X[absorbed_col]

    # ------------------------------------------------------------------ #
    # Group D — missingness flags                                         #
    # ------------------------------------------------------------------ #
    X["ledd_missing"] = X["ledd"].isna().astype(np.int8)
    X["off_missing"] = X["off"].isna().astype(np.int8)
    X["on_missing"] = X["on"].isna().astype(np.int8)
    X["gene_missing"] = X["gene"].isna().astype(np.int8)
    X["both_measured"] = (~X["off"].isna() & ~X["on"].isna()).astype(np.int8)

    # Fraction of patient visits with OFF observed (availability proxy)
    X["off_obs_rate"] = X["off_n_obs"] / X["n_visits"]
    X["on_obs_rate"] = X["on_n_obs"] / X["n_visits"]

    # ------------------------------------------------------------------ #
    # Restore original row order; drop patient_id                         #
    # ------------------------------------------------------------------ #
    X = X.loc[original_order]
    X = X.drop(columns=["patient_id"])

    return X
