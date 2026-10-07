"""Dummy mean-predictor pipeline for experiment 01_dummy.

Returns a bare sklearn DummyRegressor that always predicts the
training-set mean of the UPDRS target. No features are used.
This is the floor baseline every subsequent model must beat.
"""

from __future__ import annotations

from sklearn.dummy import DummyRegressor


def build_learner(data_dir_preview=None):  # noqa: ARG001
    """Return an unfit DummyRegressor (strategy='mean').

    Parameters
    ----------
    data_dir_preview :
        Unused; kept for interface consistency with later experiment
        scripts that pass a data directory path.

    Returns
    -------
    DummyRegressor
        Always predicts the mean of the training targets.
    """
    return DummyRegressor(strategy="mean")
