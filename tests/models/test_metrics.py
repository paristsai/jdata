from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal
from sklearn.base import BaseEstimator
from sklearn.model_selection import PredefinedSplit, cross_validate

from src.models import monkey_patch
from src.models.metrics import jdata_fscorer, use_cols, JDataScore


class MockEstimatorWithPredefinedPrediction(BaseEstimator):
    """Dummy classifier to test the scores with predefined prediction"""

    def __init__(self, pred_map):
        self.pred_map = pred_map

    def fit(self, X, y):
        pass

    def _decision_function(self, X):
        pass

    def predict(self, X, threshold=0.5):
        pass

    def predict_proba(self, X):
        return self.pred_map.get(self.hash(X), np.zeros((X.shape[0], 2)))

    def set(self, X, pred):
        self.pred_map[self.hash(X)] = pred

    def hash(self, x):
        return hash(x.tostring())


def get_jdata_test_cases():
    df = pd.read_csv(Path(__file__).parent.joinpath("test_case.csv").resolve())
    data = df[use_cols].values
    target = df.label.values
    pred_pos_proba = df.pred_pos_proba.values.reshape(-1, 1)
    pred_proba = np.concatenate((1 - pred_pos_proba, pred_pos_proba), axis=1)
    test_fold = df.test_fold.values
    expected_group_scores = {"group_0": 0.24825397, "group_1": 0.57428571}
    expected_scores = np.array(list(expected_group_scores.values()))
    return df, data, target, pred_proba, test_fold, expected_scores


def test_jdata_fscorer():
    monkey_patch.run()

    user_sku_pair, data, target, pred_proba, test_fold, expected_scores = (
        get_jdata_test_cases()
    )

    pred_map = {}
    clf = MockEstimatorWithPredefinedPrediction(pred_map)
    ps = PredefinedSplit(test_fold)

    for train_index, test_index in ps.split():
        print("TRAIN:", train_index, "TEST:", test_index)
        clf.set(data[train_index, :], pred_proba[train_index])
        clf.set(data[test_index, :], pred_proba[test_index])

    scoring = {
        "custom_score_index": jdata_fscorer(),
        "custom_score_index_with_user_sku_pair": jdata_fscorer(user_sku_pair),
    }
    scores = cross_validate(
        clf, data, target, scoring=scoring, cv=ps, return_estimator=True
    )

    for name in scoring.keys():
        assert_almost_equal(scores[f"test_{name}"], expected_scores)


def test_jdata_fscorer_class():
    monkey_patch.run()

    user_sku_pair, data, target, pred_proba, test_fold, expected_scores = (
        get_jdata_test_cases()
    )

    pred_map = {}
    clf = MockEstimatorWithPredefinedPrediction(pred_map)
    ps = PredefinedSplit(test_fold)

    for train_index, test_index in ps.split():
        print("TRAIN:", train_index, "TEST:", test_index)
        clf.set(data[train_index, :], pred_proba[train_index])
        clf.set(data[test_index, :], pred_proba[test_index])

    scoring = {
        "custom_score_index": JDataScore(),
        "custom_score_index_with_user_sku_pair": JDataScore(user_sku_pair),
    }
    scores = cross_validate(
        clf, data, target, scoring=scoring, cv=ps, return_estimator=True
    )

    for name in scoring.keys():
        assert_almost_equal(scores[f"test_{name}"], expected_scores)
