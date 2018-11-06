from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from .metrics import get_jdata_scoring


def get_X_y(dataset):
    X = dataset.drop(columns=["user_id", "label"]).fillna(-1).values
    y = dataset.label.values
    return X, y


def merge_scoring_metrics(scores, scorer):
    df = pd.DataFrame(scores)
    custom_metrics = scorer.get(filter=None)
    for metric, scores in custom_metrics.items():
        df["test_{0}".format(metric)] = scores[::2]
        df["train_{0}".format(metric)] = scores[1::2]
    return df


def score_whole_dataset(clf, dataset, pre_train=True):
    if not ("label" in dataset):
        raise ValueError("dataset must include the label column")
    X, y = get_X_y(dataset)
    if not pre_train:
        clf.fit(X, y)
    scoring, scorer = get_jdata_scoring(dataset)
    scoring["custom_index"](clf, X, y, np.arange(X.shape[0]))

    metrics = {}
    for k, v in scorer.get(filter=None).items():
        metrics["test_{}".format(k)] = v

    return pd.DataFrame(metrics)


class Model(ABC):
    @property
    def name(self):
        raise NotImplementedError

    @abstractmethod
    def train(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, y, pred):
        raise NotImplementedError

    def dump(self):
        pass

    def scale_pos_weight(self):
        pass


class TreeModel(Model):
    @abstractmethod
    def get_tree_feature(self):
        raise NotImplementedError

    @abstractmethod
    def get_feature_importance(self):
        raise NotImplementedError

    @abstractmethod
    def plot_importance(self):
        raise NotImplementedError


# class XGBClassifier(TreeModel):
#     def __init__(self, params):

#     @property
#     def name(self):
#         return self.__class__.__name__

#     def train(self, X, y):
#         pass

#     def predict(self, X):
#         pass

#     def evaluate(self, y, pred):
#         pass

#     def get_tree_feature(self):
#         pass

#     def get_feature_importance(self):
#         return self.model.feature_importances_

#     def plot_importance(self):
#         pass


# class KFold(ABC):
#     def __init__(self, model, n_splits, suffle, random_state):
#         pass

#     def cross_val_score(self):
#         raise NotImplementedError

#     def predict(self):
#         raise NotImplementedError
