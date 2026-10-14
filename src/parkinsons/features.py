"""Feature engineering for experiment 07_features.

All features are derived from the input matrix X only — never from the
target.  Call ``add_features(X)`` identically on the training and test
sets.  The function is pure (does not modify the input) and safe to call
multiple times.

Per-patient aggregates use only the patient's own rows in X.  Because
GroupKFold groups entire patients, no aggregate can cross fold boundaries
(see journal/07_features.md §Leakage analysis for the formal argument).

This is batch prediction: aggregates and ``off_next`` use other supplied
visits, including later visits, of the same held-out patient. It is not a
prospective forecast using only information available at the current visit.
Keep complete patient groups together when calling this function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *X* with all engineered features appended.

    Parameters
    ----------
    X:
        Raw feature matrix.  Must contain at minimum: ``patient_id``,
        ``age``, ``age_at_diagnosis``, ``off``, ``on``, ``ledd``,
        ``gene``, ``time_since_intake_on``, ``time_since_intake_off``.
        ``patient_id`` is used only for grouping and is *not* added to
        the output.

    Returns
    -------
    pd.DataFrame
        Copy of *X* with extra columns; ``patient_id`` is dropped from
        the output (it is an identifier, not a feature).
    """
    X = X.copy()

    # ------------------------------------------------------------------ #
    # Group A — temporal features (per-patient, derived from age)         #
    # ------------------------------------------------------------------ #
    X["disease_duration"] = X["age"] - X["age_at_diagnosis"]

    # years_since_first_visit: age − min(age) per patient
    X["years_since_first_visit"] = X["age"] - X.groupby("patient_id")["age"].transform("min")

    # visit_rank: 0-based rank within patient sorted by age
    X["visit_rank"] = (
        X.groupby("patient_id")["age"]
        .rank(method="first")
        .astype(float) - 1.0
    )

    # n_visits: total number of visits for the patient in this X
    X["n_visits"] = X.groupby("patient_id")["age"].transform("count").astype(float)

    # ------------------------------------------------------------------ #
    # Group B — patient trajectory features                               #
    # ------------------------------------------------------------------ #
    # Per-patient aggregates of off and on
    for col in ("off", "on"):
        for agg in ("mean", "median", "min", "max"):
            X[f"{col}_{agg}"] = X.groupby("patient_id")[col].transform(agg)

    # OFF of the previous and next visit (sorted by age within patient).
    # We sort within each patient group to get lag/lead correctly, then
    # restore the original row order so that positional cv_splits stay valid.
    original_order = X.index
    X = X.sort_values(["patient_id", "age"])
    X["off_prev"] = X.groupby("patient_id")["off"].shift(1)
    X["off_next"] = X.groupby("patient_id")["off"].shift(-1)
    X = X.loc[original_order]

    # Historical implementation: interpolate OFF over the DataFrame index
    # (the CSV's Index), NOT age. Preserve this behavior for reproducibility;
    # changing to age-based interpolation would require a new CV evaluation.
    off_interp = X["off"].copy()
    for pid, grp in X.groupby("patient_id"):
        if grp["off"].notna().sum() >= 2:
            interp_vals = (
                grp["off"]
                .interpolate(method="index")  # numeric index labels, not positions
            )
            off_interp.loc[grp.index] = interp_vals
    X["off_interp"] = off_interp

    # per-patient slope of off vs age (OLS, requires >= 2 non-NaN off values)
    off_slope = pd.Series(np.nan, index=X.index)
    for pid, grp in X.groupby("patient_id"):
        valid = grp[["age", "off"]].dropna()
        if len(valid) >= 2:
            # OLS slope via np.polyfit
            slope = float(np.polyfit(valid["age"], valid["off"], 1)[0])
            off_slope.loc[grp.index] = slope
    X["off_slope"] = off_slope

    # ------------------------------------------------------------------ #
    # Group C — drug timing / LEDD features                               #
    # ------------------------------------------------------------------ #
    # ledd is already in X; just add the missingness flag
    X["ledd_missing"] = X["ledd"].isna().astype(np.int8)

    # ------------------------------------------------------------------ #
    # Group D — missingness flags                                         #
    # ------------------------------------------------------------------ #
    X["off_missing"] = X["off"].isna().astype(np.int8)
    X["on_missing"] = X["on"].isna().astype(np.int8)
    X["gene_missing"] = X["gene"].isna().astype(np.int8)
    X["both_measured"] = (~X["off"].isna() & ~X["on"].isna()).astype(np.int8)

    # ------------------------------------------------------------------ #
    # Drop patient_id — identifier, not a feature                         #
    # ------------------------------------------------------------------ #
    X = X.drop(columns=["patient_id"])

    return X
