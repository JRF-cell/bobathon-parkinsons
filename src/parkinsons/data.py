"""Data loading for the Parkinson's UPDRS prediction workspace.

Loads the merged (X_train + y_train) dataset for training and
X_test for Kaggle submission. Data paths are resolved via
PROJECT_ROOT so scripts run from any working directory.
"""

from __future__ import annotations

import pandas as pd

from parkinsons import PROJECT_ROOT

TARGET_COL = "target"
_DATA_DIR = PROJECT_ROOT / "data"


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` from the training CSV files.

    Reads ``data/X_train.csv`` and ``data/y_train.csv``, joins them on
    the shared index, and splits into features and target.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``X`` is the feature matrix (all columns except ``target``).
        ``y`` is the target series (UPDRS motor score).
    """
    X = pd.read_csv(_DATA_DIR / "X_train.csv", index_col="Index")
    y_df = pd.read_csv(_DATA_DIR / "y_train.csv", index_col="Index")
    y = X.join(y_df)[TARGET_COL]
    return X, y


def load_test_dataset() -> pd.DataFrame:
    """Return ``X_test`` for the Kaggle submission CSV.

    Reads ``data/X_test.csv`` and preserves the ``Index`` column needed
    by the competition. Unlike ``load_dataset``, this leaves ``Index`` as
    a column. Set it as the index before feature engineering so it cannot
    become a predictor and predictions retain the original test-row order.

    Returns
    -------
    pd.DataFrame
        Raw test features plus the submission identifier column ``Index``.
    """
    return pd.read_csv(_DATA_DIR / "X_test.csv")
