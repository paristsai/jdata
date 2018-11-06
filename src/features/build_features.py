import gc

import fire
import pandas as pd

from . import (
    Feature,
    UserItemCrossTimeFeature,
    count_unique_ratio,
    drop_duplicated_columns,
)
from .. import DEFAULT_VERSION, timer, to_unix
from ..data import FeatherReadWriter, JDataTrainTestReadWriter, info
from .helper import create_group_interval_features, merge_time_window_agg

RULES = [
    # V1 - Basic Features
    # single column
    # user behavior total action counts
    {"groupby": ["user_id"], "select": "sku_id", "agg_func": "count"},
    # Unique Ratio
    {
        "groupby": ["user_id"],
        "select": "sku_id",
        "agg_func": count_unique_ratio,
        "agg_name": "AvgSkuPerDistinct",
    },
    # age total counts
    {"groupby": ["age"], "select": "sku_id", "agg_func": "count"},
    # sex total counts
    {"groupby": ["sex"], "select": "sku_id", "agg_func": "count"},
    # user_lv_cd total counts
    {"groupby": ["user_lv_cd"], "select": "sku_id", "agg_func": "count"},
    # sku_id interaction counts
    {"groupby": ["sku_id"], "select": "user_id", "agg_func": "count"},
    {
        "groupby": ["sku_id"],
        "select": "user_id",
        "agg_func": count_unique_ratio,
        "agg_name": "AvgUserPerDistinct",
    },
    # brand interaction counts
    {"groupby": ["brand"], "select": "user_id", "agg_func": "count"},
    # cate counts
    {"groupby": ["cate"], "select": "user_id", "agg_func": "count"},
    # multiple columns
    # user personal preference
    # user & sku_id
    {"groupby": ["user_id", "sku_id"], "select": "brand", "agg_func": "count"},
    # user & brand
    {"groupby": ["user_id", "brand"], "select": "sku_id", "agg_func": "count"},
    # user & cate
    {"groupby": ["user_id", "cate"], "select": "sku_id", "agg_func": "count"},
    # sex preference
    # sex & sku
    {"groupby": ["sex", "sku_id"], "select": "user_id", "agg_func": "count"},
    # sex & brand
    {"groupby": ["sex", "brand"], "select": "user_id", "agg_func": "count"},
    # age preference
    # age & sku
    {"groupby": ["age", "sku_id"], "select": "user_id", "agg_func": "count"},
    # age & brand
    {"groupby": ["brand", "sku_id"], "select": "user_id", "agg_func": "count"},
    # V2
    # cart conversion
    # ...
]


def prepare_train_pair(df: pd.DataFrame) -> pd.DataFrame:
    """
    drop duplicated pair to build unique train pair
    """

    used_cols = [
        "user_id",
        "age",
        "sex",
        "user_lv_cd",
        "year_user_reg",
        "month_user_reg",
        "day_user_reg",
        "reg_days",
        "sku_id",
        "cate",
        "brand",
        "a1",
        "a2",
        "a3",
    ]
    return df[used_cols].drop_duplicates()


def get_actions(df, actions=[]):
    def get_suffix(key):
        return "of_{}".format(key) if key != "all" else ""

    index_map = {
        "all": None,
        "browse": df.browse == 1,
        "click": df.click == 1,
        "follow": df.follow == 1,
        "purchase": df.purchase == 1,
        "addcart": df.add_cart == 1,
        "remcart": df.rem_cart == 1,
    }

    for action, indice in index_map.items():
        if not actions or action in actions:
            yield (action, df[indice] if indice is not None else df, get_suffix(action))


def main(name, version=DEFAULT_VERSION):
    frw = FeatherReadWriter()

    with timer(f"prepare train data: {name}-{version}"):
        rw = JDataTrainTestReadWriter()
        train, _ = rw.read(name, version)
        config = rw.get_config(name, version)
        print(config)
        user_column = "user_id"
        time_column = config["setting"]["time_col"]
        mid_unix_time = to_unix(config["mid"])

    with timer("prepare train pair"):
        train_pair = prepare_train_pair(train)
    # TODO: when preparing features we need to find existed file before create
    # prepare_feature: get created feature or start to create
    # featureA = prepare_feature("A")

    for action, sub_train, suffix in get_actions(train):
        with timer("create feature - UserItemCrossTimeFeature - {}".format(action)):
            ui = UserItemCrossTimeFeature(
                sub_train, user_column, "sku_id", time_column, mid_unix_time
            )
            train_pair = ui.create_and_merge(train_pair, how="left", suffix=suffix)

    for gp in [["user_id"], ["sku_id"], ["user_id", "sku_id"]]:
        for action, sub_train, suffix in get_actions(train, ["purchase", "browse"]):
            with timer(
                "create feature - GroupIntervalTimeFeature"
                " - group:{group} action:{action}".format(group=gp, action=action)
            ):
                train_pair = create_group_interval_features(
                    sub_train, train_pair, gp, time_column, suffix
                )

    with timer("save train pair with features to feather format file - stage1"):
        frw.write(train_pair, "processed", frw.extend_name(f"{name}_{version}.stage1"))

    with timer("reset train pair"):
        del train_pair
        train_pair = prepare_train_pair(train)
        gc.collect()

    rules = RULES
    windows = [1, 3, 7, 15, 30, 45]
    for action, sub_train, suffix in get_actions(train, ["all", "browse", "purchase"]):
        with timer(
            f"create feature - TimeWindowAggFeature - "
            "windows:{windows}, action:{action}, data_rows:{len(sub_train)}."
        ):
            part_of_suffix = suffix
            train_pair = merge_time_window_agg(
                sub_train,
                train_pair,
                rules,
                windows,
                time_column,
                mid_unix_time,
                part_of_suffix,
            )

    info(train_pair)
    train_pair = drop_duplicated_columns(train_pair)

    with timer("save train pair with features to feather format file - stage2"):
        frw.write(train_pair, "processed", frw.extend_name(f"{name}_{version}.stage2"))

    with timer("read staging features and merge"):
        del train
        del train_pair
        gc.collect()

    # TODO: add File method to find file with pattern
    train_pair = Feature.concat(
        [
            frw.read("processed", frw.extend_name(f"{name}_{version}.stage{i}"))
            for i in range(1, 3)
        ]
    )
    info(train_pair)

    with timer("save train pair with features to feather format file - final"):
        frw.write(train_pair, "processed", frw.extend_name(f"{name}_{version}"))


if __name__ == "__main__":
    fire.Fire(main)
