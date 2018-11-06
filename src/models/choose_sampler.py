from collections import Counter

import pandas as pd
from xgboost import XGBClassifier

from . import get_X_y, monkey_patch, score_whole_dataset
from .. import PROJECT_DIR, timer
from ..data import FeatherReadWriter
from .sampling import sampler


def main():
    monkey_patch.run()

    frw = FeatherReadWriter()
    train = frw.read(
        dir="processed", name=frw.extend_name("all_merged_1.0"), nthreads=4
    )
    label = frw.read(
        dir="processed", name=frw.extend_name("all_merged_1.0.label"), nthreads=4
    )
    train[label.columns] = label
    X, y = get_X_y(train)

    # load online dataset for submission
    online_train = frw.read(
        dir="processed", name=frw.extend_name("all_merged_online"), nthreads=4
    )
    online_label = frw.read(
        dir="processed", name=frw.extend_name("all_merged_online.label"), nthreads=4
    )
    online_train[online_label.columns] = online_label
    sampling_paras = [
        ("rus", 0.1),
        ("rus", 0.01),
        ("rus", 0.001),
        ("nm", 0.1),
        ("nm", 0.01),
        ("nm", 0.001),
        # ("tl", None),
        # ("renn", None),
        # ("allknn", None),
        # ("enn", None),
    ]
    # allknn 跑了 499m 跑不出來, tl 跑了 120m 也跑不出來
    fpath = str(PROJECT_DIR.joinpath("reports/metrics_by_samplers.csv"))
    whole_dataset_metrics = pd.DataFrame()
    # whole_dataset_metrics = pd.read_csv(fpath, index_col=0)
    for method, ratio in sampling_paras:
        with timer(f"method: {method}, ratio: {ratio}"):
            sampler_setting = {"name": method, "ratio": ratio, "n_jobs": 4}
            s = sampler(**sampler_setting)
            res_X, res_y, indices = s.fit_sample(X, y)
        print("Distribution of class labels after resampling {}".format(Counter(res_y)))

        clf = XGBClassifier(nthread=-1)
        with timer("start training"):
            clf.fit(res_X, res_y, verbose=3)

        score_df = score_whole_dataset(clf, online_train)
        score_df = score_df.set_index([["{0}-{1}".format(method, ratio)]])
        whole_dataset_metrics = pd.concat([whole_dataset_metrics, score_df])
        whole_dataset_metrics.to_csv(fpath)
        frw.write(
            pd.DataFrame({"index": indices}),
            "processed",
            frw.extend_name(f"all_merged_1.0_{method}_{ratio}_indices"),
        )


if __name__ == "__main__":
    main()
