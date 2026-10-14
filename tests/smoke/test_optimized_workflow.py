"""Regression checks for optimized features and final-model selection.

Extract the selection function without running the experiment's tuning or Hub
operations. Model doubles expose wrong routing without an expensive refit.
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from parkinsons import PROJECT_ROOT
from parkinsons.features_optimized import add_features_optimized


def selection_namespace():
    tree = ast.parse((PROJECT_ROOT / 'experiments/07_features_optimized.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fit_final_model')

    class Model:
        def __init__(self, lr, leaves, l2):
            self.params = (lr, leaves, l2)

        def fit(self, X, y):
            self.training_features = X
            return self

    class Specialist(Model):
        pass

    ns = dict(make_pipe=Model, SpecialistRegressor=Specialist,
              REF_LR=.03, REF_LEAVES=63, REF_L2=1,
              best_lr=.05, best_leaves=63, best_l2=1,
              X_opt=object(), X_ref=object(), y=object(),
              best2=[('specialist', (4.28, None)), ('hgbr_opt', (4.30, None))])
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<selection>', 'exec'), ns)
    return ns, Specialist


def test_ensemble_selects_the_actual_candidate_models():
    ns, specialist_type = selection_namespace()
    fitted, features = ns['fit_final_model']('E: ensemble best2')
    assert fitted[0] == 'ensemble'
    assert isinstance(fitted[1][0], specialist_type)
    assert not isinstance(fitted[1][1], specialist_type)
    assert all(m.params == (.05, 63, 1) for m in fitted[1])
    assert features is ns['X_opt']


def test_reference_keeps_original_feature_matrix():
    ns, _ = selection_namespace()
    fitted, features = ns['fit_final_model']('07_features reference')
    assert features is ns['X_ref']
    assert fitted.training_features is ns['X_ref']
    assert fitted.params == (.03, 63, 1)
    with pytest.raises(ValueError, match='Unknown model'):
        ns['fit_final_model']('unexpected')


def test_age_interpolation_is_patient_local_and_preserves_row_order():
    X = pd.DataFrame({
        'patient_id': ['a', 'a', 'a', 'b', 'b', 'b'],
        'age': [50., 51., 54., 60., 61., 64.],
        'age_at_diagnosis': [49.] * 3 + [59.] * 3,
        'off': [10., np.nan, 30., 80., np.nan, 100.],
        'on': [5., 7., 15., 40., 45., 50.],
        'ledd': [100.] * 6, 'gene': ['GBA'] * 6,
        'time_since_intake_on': [1.] * 6,
        'time_since_intake_off': [12.] * 6,
    }, index=[100, 101, 900, 13, 97, 99]).iloc[[4, 1, 5, 0, 3, 2]]
    before = X.copy(deep=True)
    features = add_features_optimized(X)
    pd.testing.assert_frame_equal(X, before)
    assert features.index.equals(X.index)
    assert features.loc[101, 'off_interp'] == 15.  # Uses age gaps, not ID gaps.
    assert features.loc[97, 'off_interp'] == 85.
    assert 'patient_id' not in features
    a = X[X.patient_id == 'a']
    pd.testing.assert_frame_equal(features.loc[a.index], add_features_optimized(a))
    relabeled = X.copy()
    relabeled.index = np.arange(len(X)) * 1000
    pd.testing.assert_frame_equal(features.reset_index(drop=True),
                                  add_features_optimized(relabeled).reset_index(drop=True))
