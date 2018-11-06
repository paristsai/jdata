import gc
from abc import ABC, abstractmethod

from typing import List, Dict

import pandas as pd

from ..data import optimize_numeric_values


class Feature(ABC):
    df: pd.DataFrame

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def create(self):
        raise NotImplementedError

    def create_and_merge(
        self, target: pd.DataFrame, how="left", suffix=None, *args, **kwargs
    ) -> pd.DataFrame:
        old_columns = target.columns
        feature_df = self.create()
        if suffix:
            mapper = Feature.get_rename_mapper(old_columns, feature_df.columns, suffix)
            feature_df = feature_df.rename(columns=mapper)
        print("start merging...")
        return target.merge(feature_df, how=how, *args, **kwargs)

    @staticmethod
    def concat(rslt: List, axis=1, join="inner", *args):
        df = pd.concat(rslt, axis=axis, join=join, *args)
        df = drop_duplicated_columns(df)
        return df

    @staticmethod
    def get_rename_mapper(old: List, new: List, suffix: str) -> Dict:
        mapper: Dict = {}
        if suffix:
            suffix = "_{}".format(suffix)
            mapper = dict([(c, "{}{}".format(c, suffix)) for c in new if c not in old])
        return mapper

    @staticmethod
    @abstractmethod
    def categorical_features():
        raise NotImplementedError


class UserItemCrossTimeFeature(Feature):
    def __init__(
        self,
        data: pd.DataFrame,
        user_column: str,
        item_column: str,
        time_column: str,
        end_time,
    ) -> None:
        super().__init__(data)
        self.user_column = user_column
        self.item_column = item_column
        self.time_column = time_column
        self.user_suffix = "_by_user"
        self.user_action_suffix = "_by_user_{item}".format(item=self.item_column)

        if isinstance(end_time, pd.DatetimeIndex):
            self.end_time_unix = end_time.value // 10 ** 2
        elif isinstance(end_time, (int, float)):
            self.end_time_unix = int(end_time)
        else:
            raise TypeError("unexpected end_time type: {}".format(type(end_time)))

    def create(self):
        user_action_df = (
            self.df.groupby([self.user_column, self.item_column])[self.time_column]
            .agg(["max", "min"])
            .reset_index()
            .rename(columns={"min": "first_ts", "max": "last_ts"})
        )

        user_df = (
            user_action_df.groupby(self.user_column)
            .agg({"first_ts": "min", "last_ts": "max"})
            .reset_index()
        )

        user_df = self.add_time_diff_features(user_df, self.end_time_unix)
        user_action_df = self.add_time_diff_features(user_action_df, self.end_time_unix)

        merged_df = user_action_df.merge(
            user_df,
            on=self.user_column,
            suffixes=[self.user_action_suffix, self.user_suffix],
        )

        d = self.get_columns_dict()

        merged_df[d["last_last"]] = (
            merged_df[d["last_item"]] - merged_df[d["last_item"]]
        )
        merged_df[d["first_first"]] = (
            merged_df[d["first_item"]] - merged_df[d["first_item"]]
        )
        # optimize
        self.df = optimize_numeric_values(merged_df, fillna=-1, as_int=True)

        del user_df, user_action_df, merged_df
        gc.collect()
        return self.df

    def add_time_diff_features(self, df, ts):
        df["first_diff"] = ts - df.first_ts
        df["last_diff"] = ts - df.last_ts
        df["first_last_diff"] = df.last_ts - df.first_ts
        return df

    def get_columns_dict(self) -> Dict:
        user_suffix, user_action_suffix = self.user_suffix, self.user_action_suffix
        # the last action timestamp
        last = f"last_ts{user_suffix}"
        # the last target action timestamp
        last_item = f"last_ts{user_action_suffix}"
        # the last action to the last target action timestamp
        last_last = f"last_last_diff{user_action_suffix}"

        # the first action timestamp
        first = f"first_ts{user_suffix}"
        # the first target action timestamp
        first_item = f"first_ts{user_action_suffix}"
        # first action to first target action
        first_first = f"first_first_diff{user_action_suffix}"

        return {
            "last": last,
            "last_item": last_item,
            "last_last": last_last,
            "first": first,
            "first_item": first_item,
            "first_first": first_first,
        }

    @staticmethod
    def categorical_features():
        return []


class GroupIntervalTimeFeature(Feature):
    def __init__(
        self, data: pd.DataFrame, group_column: List, time_column: str, shift: int = 1
    ) -> None:
        self.group_column = group_column
        self.time_column = time_column
        self.shift = shift
        super().__init__(data)

    def create(self):
        used_cols = self.group_column + [self.time_column]
        lag_column = "{0}_lag".format(self.time_column)
        user_action_interval = self.df[used_cols].sort_values(
            self.time_column, ascending=True
        )
        user_action_interval[lag_column] = user_action_interval.groupby(
            self.group_column
        )[self.time_column].shift(self.shift)
        user_action_interval["diff"] = (
            user_action_interval[self.time_column] - user_action_interval[lag_column]
        )
        user_action_interval_summary = (
            user_action_interval.groupby(self.group_column)
            .diff.describe()
            .reset_index()
        )

        self.df = user_action_interval_summary
        del user_action_interval
        gc.collect()
        return self.df

    @staticmethod
    def categorical_features():
        return []


class AggFeature(Feature):
    def __init__(
        self, data: pd.DataFrame, target: pd.DataFrame, rules: List[Dict]
    ) -> None:
        self.rules = rules
        self.target = target
        super().__init__(data)

    def create(self):
        rules = self.rules
        source = self.df
        target = self.target

        for rule in rules:
            agg_name = rule["agg_name"] if "agg_name" in rule else rule["agg_func"]
            new_feature = "{0}_{1}_{2}".format(
                "_".join(rule["groupby"]), agg_name, rule["select"]
            )

            print(
                "Grouping by {}, and aggregating {} with {}".format(
                    rule["groupby"], rule["select"], agg_name
                )
            )

            all_features = list(set(rule["groupby"] + [rule["select"]]))

            gp = (
                source[all_features]
                .groupby(rule["groupby"])[rule["select"]]
                .agg(rule["agg_func"])
                .reset_index()
                .rename(index=str, columns={rule["select"]: new_feature})
            )
            target = target.merge(gp, on=rule["groupby"], how="left")

            del gp
            gc.collect()

        self.df = target
        return self.df

    @staticmethod
    def categorical_features():
        return []


# agg by combination
class CombinationAggFeature(AggFeature):
    pass


# TODO: rewrite from helper.merge_time_window_agg
# class TimeWindowAggFeature(Feature):
#     def __init__(
#         self,
#         data: pd.DataFrame,
#         target: pd.DataFrame,
#         rules: List[Dict],
#         windows: List,
#         time_column: str,
#         end_time_unix: int,
#         suffix: str,
#     ) -> None:
#         self.rules = rules
#         self.target = target
#         self.windows = windows
#         self.time_column = time_column
#         self.end_time_unix = end_time_unix

#     def create(self):
#         rules = self.rules
#         data = self.df
#         for day in self.windows:
#             print("Window Size: {} day".format(day))
#             cond_index = data[self.time_column] < self.end_time_unix
#             if day == 0 or day is not None:
#                 start_time_unix = self.end_time_unix - 60 * 60 * 24 * day
#                 cond_index = (data[self.time_column] >= start_time_unix) & cond_index

#             self.df = self.df[cond_index]
#             self.rules = rules
#             self.target = super().create()
#         return self.target

#     @staticmethod
#     def categorical_features():
#         return []


def drop_duplicated_columns(df):
    distinct_cols = ~df.columns.duplicated()
    df = df.iloc[:, distinct_cols]
    return df


def count_unique_ratio(x):
    return len(x) / x.nunique()
