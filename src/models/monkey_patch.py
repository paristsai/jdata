import numbers
import time
import warnings
from traceback import format_exception_only

import numpy as np
import sklearn
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection._validation import _index_param_value
from sklearn.utils._joblib import logger
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.validation import _num_samples


def is_use_index(key):
    return any(map(lambda word: word in key, ["index", "indice", "custom", "jdata"]))


def use_or_not(use, arr):
    if use is True and isinstance(arr, np.ndarray):
        return arr
    return None


def _fit_and_score(  # noqa
    estimator,
    X,
    y,
    scorer,
    train,
    test,
    verbose,
    parameters,
    fit_params,
    return_train_score=False,
    return_parameters=False,
    return_n_test_samples=False,
    return_times=False,
    return_estimator=False,
    error_score="raise-deprecating",
):
    if verbose > 1:
        if parameters is None:
            msg = ""
        else:
            msg = "%s" % (", ".join("%s=%s" % (k, v) for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * "."))

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict(
        [(k, _index_param_value(X, v, train)) for k, v in fit_params.items()]
    )

    train_scores = {}
    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    is_multimetric = not callable(scorer)
    n_scorers = len(scorer.keys()) if is_multimetric else 1

    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == "raise":
            raise
        elif error_score == "raise-deprecating":
            warnings.warn(
                "From version 0.22, errors during fit will result "
                "in a cross validation score of NaN by default. Use "
                "error_score='raise' if you want an exception "
                "raised or error_score=np.nan to adopt the "
                "behavior from version 0.22.",
                FutureWarning,
            )
            raise
        elif isinstance(error_score, numbers.Number):
            if is_multimetric:
                test_scores = dict(zip(scorer.keys(), [error_score] * n_scorers))
                if return_train_score:
                    train_scores = dict(zip(scorer.keys(), [error_score] * n_scorers))
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
            warnings.warn(
                "Estimator fit failed. The score on this train-test"
                " partition for these parameters will be set to %f. "
                "Details: \n%s" % (error_score, format_exception_only(type(e), e)[0]),
                FitFailedWarning,
            )
        else:
            raise ValueError(
                "error_score must be the string 'raise' or a"
                " numeric value. (Hint: if using 'raise', please"
                " make sure that it has been spelled correctly.)"
            )

    else:
        fit_time = time.time() - start_time
        # _score will return dict if is_multimetric is True
        # **PATCH** - send train / test indices when use_index is True
        use_index = any(is_use_index(key) for key in scorer.keys())

        test_scores = _score(
            estimator,
            X_test,
            y_test,
            scorer,
            is_multimetric,
            use_or_not(use_index, test),
        )
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = _score(
                estimator,
                X_train,
                y_train,
                scorer,
                is_multimetric,
                use_or_not(use_index, train),
            )

    if verbose > 2:
        if is_multimetric:
            for scorer_name, score in test_scores.items():
                msg += ", %s=%s" % (scorer_name, score)
        else:
            msg += ", score=%s" % test_scores
    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = "%s, total=%s" % (msg, logger.short_format_time(total_time))
        print("[CV] %s %s" % ((64 - len(end_msg)) * ".", end_msg))

    ret = [train_scores, test_scores] if return_train_score else [test_scores]

    if return_n_test_samples:
        ret.append(_num_samples(X_test))
    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    if return_estimator:
        ret.append(estimator)
    return ret


def _score(estimator, X_test, y_test, scorer, is_multimetric=False, indices=None):
    """Compute the score(s) of an estimator on a given test set.
    Will return a single float if is_multimetric is False and a dict of floats,
    if is_multimetric is True
    """
    if is_multimetric:
        return _multimetric_score(estimator, X_test, y_test, scorer, indices)
    else:
        if y_test is None:
            score = scorer(estimator, X_test)
        # **PATCH**
        elif indices is not None:
            score = scorer(estimator, X_test, y_test, indices)
        else:
            score = scorer(estimator, X_test, y_test)

        if hasattr(score, "item"):
            try:
                # e.g. unwrap memmapped scalars
                score = score.item()
            except ValueError:
                # non-scalar?
                pass

        if not isinstance(score, numbers.Number):
            raise ValueError(
                "scoring must return a number, got %s (%s) "
                "instead. (scorer=%r)" % (str(score), type(score), scorer)
            )
    return score


def _multimetric_score(estimator, X_test, y_test, scorers, indices=None):
    """Return a dict of score for multimetric scoring"""
    scores = {}
    for name, scorer in scorers.items():
        if y_test is None:
            score = scorer(estimator, X_test)
        # **PATCH**
        elif indices is not None:
            score = scorer(estimator, X_test, y_test, indices)
        else:
            score = scorer(estimator, X_test, y_test)

        if hasattr(score, "item"):
            try:
                # e.g. unwrap memmapped scalars
                score = score.item()
            except ValueError:
                # non-scalar?
                pass
        scores[name] = score

        if not isinstance(score, numbers.Number):
            raise ValueError(
                "scoring must return a number, got %s (%s) "
                "instead. (scorer=%s)" % (str(score), type(score), name)
            )
    return scores


def run():
    """
    Modify sklearn functions or methods using _fit_and_score such as cross_validate
    cross_val_score, GridSearchCV and etc, which make internal score functions to
    retrieve the cv split indices to put the original information together.
    In the case of the JData competition, we need the indices to check
    (user_id, sku_id) pair prediction correction, but there is no chance to use
    user_id to construct the pairs since it's not appropriate to keep user_id
    as a feature inside the X of the model training.

    Usage:

    1. Create a scoring function dictionary with the key contains any keyword
    of the list ["index", "indices", "custom"]
    >>> scoring = { "custom": your_custom_score_function }

    2. Send the scoring dictionary parameter.
    >>> from sklearn.model_selection import cross_validate, GridSearchCV
    >>> cross_validate(clf, X, y, cv=kfold, scoring=scoring, verbose=1)
    >>> GridSearchCV(clf, param_grid, scoring=scoring, cv=kfold, refit=refit)
    """
    version = "0.20.0"
    if sklearn.__version__ != version:
        message = (
            f"The patch is only for sklearn.__version__ == {version}"
            "but {sklearn.__version__}"
        )
        warnings.warn(message, DeprecationWarning)

    # Command for checking the modules using the function:
    # % find . -name '*.py' | xargs grep '_fit_and_score' | awk -F: '{print $1}' | uniq
    sklearn.model_selection._validation._fit_and_score = _fit_and_score
    sklearn.model_selection._validation._score = _score
    sklearn.model_selection._validation._multimetric_score = _multimetric_score
    sklearn.model_selection._search._fit_and_score = _fit_and_score
    print("Patched!")


def main():
    run()


if __name__ == "__main__":
    main()
