import numpy as np
import pandas as pd

use_cols = ["user_id", "sku_id"]


def division(x, y):
    if y:
        return x / y
    return 0


def multidim_intersect(arr1, arr2):
    set1 = set(map(tuple, arr1))
    set2 = set(map(tuple, arr2))
    return set1 & set2


def jdata_fscorer(user_sku_pair=None):
    def f11(pred_pair, y_pair):
        correct_user_num = np.isin(pred_pair[:, 0], y_pair[:, 0]).sum()
        precision_1 = division(correct_user_num, len(pred_pair))
        recall_1 = division(correct_user_num, len(y_pair))
        f11 = 6 * recall_1 * precision_1 / (5 * recall_1 + precision_1)
        print("correct_user_num", correct_user_num)
        print(
            "precision_1: {}, recall_1: {}, f11: {}".format(precision_1, recall_1, f11)
        )
        return f11

    def f12(pred_pair, y_pair):
        correct_sku_num = len(multidim_intersect(pred_pair, y_pair))
        precision_2 = division(correct_sku_num, len(pred_pair))
        recall_2 = division(correct_sku_num, len(y_pair))
        f12 = division(5 * recall_2 * precision_2, (2 * recall_2 + 3 * precision_2))
        print("correct_sku_num", correct_sku_num)
        print(
            "precision_2: {}, recall_2: {}, f12: {}".format(precision_2, recall_2, f12)
        )
        return f12

    def prepare_y_pair(df):
        return (
            df[df.label == 1]
            .groupby("user_id", as_index=False)
            .first()[use_cols]
            .drop_duplicates()
            .values
        )

    def prepare_pair(clf, X, y, indices, threshold=0.5):
        if isinstance(indices, np.ndarray) and isinstance(user_sku_pair, pd.DataFrame):
            df = user_sku_pair.iloc[indices, :][use_cols + ["label"]]
            if not np.array_equal(df.label, y):
                raise ValueError("indices between user_sku_pair and y are different.")
        else:
            print(
                "user_sku_pair type: {}; indices type: {}".format(
                    type(user_sku_pair), type(indices)
                )
            )
            # only for testing
            if X.shape[1] != len(use_cols):
                raise ValueError(
                    f"X shape {X.shape} not match use_cols length {len(use_cols)}."
                )
            df = pd.DataFrame(data=X, columns=use_cols)
            df["label"] = y

        y_pair = prepare_y_pair(df)
        y_pred = clf.predict_proba(X)
        df["prob"] = y_pred[:, -1]
        pred_pair = (
            df.sort_values("prob", ascending=False)
            .groupby("user_id", as_index=False)
            .first()
        )
        pred_positive_pair = pred_pair[pred_pair.prob >= threshold][use_cols].values

        return pred_positive_pair, y_pair

    def score(clf, X, y, indices=None):
        pred_pair, y_pair = prepare_pair(clf, X, y, indices=indices)
        f11_ratio = 0.4
        f12_ratio = 0.6
        return f11_ratio * f11(pred_pair, y_pair) + f12_ratio * f12(pred_pair, y_pair)

    return score


class JDataScore:
    def __init__(self, user_sku_pair=None, verbose=0):
        self.user_sku_pair = user_sku_pair
        self.verbose = verbose
        self.rs: dict = {
            "correct_user_num": [],
            "precision_1": [],
            "recall_1": [],
            "f11": [],
            "correct_sku_num": [],
            "precision_2": [],
            "recall_2": [],
            "f12": [],
            "score": [],
        }

    def __call__(self, clf, X, y, indices=None):
        return self.score(clf, X, y, indices)

    def save(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.rs:
                self.rs[k].append(v)

    def get(self, filter="test"):
        """The parameter return_train_score of GridSearchCV and cross_validate
        are both 'warn', so the elements of each list is followd by the ordering:
        [test, train, test, train, ...]
        """
        if filter == "test":
            new_obj = {}
            for k, v in self.rs.items():
                new_obj[k] = v[::2]
            return new_obj
        return self.rs.copy()

    def f11(self, pred_pair, y_pair):
        correct_user_num = np.isin(pred_pair[:, 0], y_pair[:, 0]).sum()
        precision_1 = division(correct_user_num, len(pred_pair))
        recall_1 = division(correct_user_num, len(y_pair))
        f11 = division(6 * recall_1 * precision_1, (5 * recall_1 + precision_1))
        self.save(
            correct_user_num=correct_user_num,
            precision_1=precision_1,
            recall_1=recall_1,
            f11=f11,
        )
        if self.verbose > 0:
            print(
                f"correct_user_num: {correct_user_num}, precision_1: {precision_1}"
                ", recall_1: {recall_1}, f11: {f11}"
            )
        return f11

    def f12(self, pred_pair, y_pair):
        correct_sku_num = len(multidim_intersect(pred_pair, y_pair))
        precision_2 = division(correct_sku_num, len(pred_pair))
        recall_2 = division(correct_sku_num, len(y_pair))
        f12 = division(5 * recall_2 * precision_2, (2 * recall_2 + 3 * precision_2))
        self.save(
            correct_sku_num=correct_sku_num,
            precision_2=precision_2,
            recall_2=recall_2,
            f12=f12,
        )
        if self.verbose > 0:
            print(
                f"correct_sku_num: {correct_sku_num}, precision_2: {precision_2}"
                ", recall_2: {recall_2}, f12: {f12}"
            )
        return f12

    def prepare_y_pair(self, df):
        return (
            df[df.label == 1]
            .groupby("user_id", as_index=False)
            .first()[use_cols]
            .drop_duplicates()
            .values
        )

    def prepare_pair(self, clf, X, y, indices, threshold=0.5):
        if isinstance(indices, np.ndarray) and isinstance(
            self.user_sku_pair, pd.DataFrame
        ):
            df = self.user_sku_pair.iloc[indices, :][use_cols + ["label"]]
            if not np.array_equal(df.label, y):
                raise ValueError("indices between user_sku_pair and y are different.")
        else:
            if self.verbose > 1:
                print(
                    "user_sku_pair type: {}; indices type: {}".format(
                        type(self.user_sku_pair), type(indices)
                    )
                )
            # only for testing
            if X.shape[1] != len(use_cols):
                raise ValueError(
                    f"X shape {X.shape} not match use_cols length {len(use_cols)}."
                )
            df = pd.DataFrame(data=X, columns=use_cols)
            df["label"] = y

        y_pair = self.prepare_y_pair(df)
        y_pred = clf.predict_proba(X)
        df["prob"] = y_pred[:, -1]
        pred_pair = (
            df.sort_values("prob", ascending=False)
            .groupby("user_id", as_index=False)
            .first()
        )
        pred_positive_pair = pred_pair[pred_pair.prob >= threshold][use_cols].values

        return pred_positive_pair, y_pair

    def score(self, clf, X, y, indices=None):
        pred_pair, y_pair = self.prepare_pair(clf, X, y, indices=indices)
        f11_ratio = 0.4
        f12_ratio = 0.6
        score = f11_ratio * self.f11(pred_pair, y_pair) + f12_ratio * self.f12(
            pred_pair, y_pair
        )
        self.save(score=score)
        return score


def get_jdata_scoring(user_sku_pair=None):
    scorer = JDataScore(user_sku_pair)
    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "neg_log_loss": "neg_log_loss",
        "roc_auc": "roc_auc",
        "custom_index": scorer,
    }
    return scoring, scorer
