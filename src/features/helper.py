import gc

from . import AggFeature, GroupIntervalTimeFeature
from ..data import optimize_numeric_values


def create_group_interval_features(train, train_pair, gp, tc, suffix):
    it = GroupIntervalTimeFeature(train[gp + [tc]], gp, tc)
    gp_suffix = "X".join([e.replace("_id", "") for e in gp])
    mix_suffix = "{0}_{1}".format(gp_suffix, suffix) if suffix else gp_suffix
    train_pair = it.create_and_merge(train_pair, suffix=mix_suffix)
    train_pair = optimize_numeric_values(train_pair, fillna=-1, as_int=True)
    return train_pair


def merge_time_window_agg(
    source, target, rules, windows, time_column, end_time_unix, suffix
):
    for day in windows:
        print("Window Size: {0} day".format(day))
        cond_index = source[time_column] < end_time_unix
        if day is not None and day > 0:
            start_time_unix = end_time_unix - 60 * 60 * 24 * day
            cond_index = (source[time_column] >= start_time_unix) & cond_index

        agg = AggFeature(source[cond_index], target, rules)

        day_suffix = "{0}d".format(day) if day else ""
        suffixes = [day_suffix, suffix]
        mix_suffix = "_".join([e for e in suffixes if e])

        target = agg.create_and_merge(target, suffix=mix_suffix)
        print("optimizing...")
        target = optimize_numeric_values(target, fillna=0, as_int=True)
        del agg
        gc.collect()

    # print("optimizing...")
    # target = optimize_numeric_values(target, fillna=0, as_int=True)
    return target
