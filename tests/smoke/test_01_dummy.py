"""Smoke test for experiment 01_dummy.

Fits DummyRegressor on a training slice and asserts that the number
of predictions equals the number of rows in the test slice.
Because DummyRegressor uses no features, there is no cross-row history
to worry about — this test is trivially structural.
"""

from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyRegressor

from parkinsons import PROJECT_ROOT
from parkinsons.pipeline import build_learner

_DATA_DIR = PROJECT_ROOT / "data"


def test_01_dummy_prediction_shape() -> None:
    """Predictions count must equal test-slice row count."""
    X = pd.read_csv(_DATA_DIR / "X_train.csv", index_col="Index")
    y_df = pd.read_csv(_DATA_DIR / "y_train.csv", index_col="Index")
    y = X.join(y_df)["target"]

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train = y.iloc[:split]

    learner: DummyRegressor = build_learner()
    learner.fit(X_train, y_train)
    preds = learner.predict(X_test)

    assert len(preds) == len(X_test), (
        f"Expected {len(X_test)} predictions, got {len(preds)}"
    )
