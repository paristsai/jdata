from pprint import pprint

import numpy as np
import pandas as pd


def info(df):
    print(df.info(max_cols=10000, memory_usage=True, null_counts=True))


def mem_usage(pd_obj):
    if isinstance(pd_obj, pd.DataFrame):
        usage_bytes = pd_obj.memory_usage(deep=True).sum()
    else:
        usage_bytes = pd_obj.memory_usage(deep=True)
    usage_mb = usage_bytes / 1024 ** 2
    return "{:03.2f} MB".format(usage_mb)


def compare_mem_usage(b, a):
    print("{0} -> {1}".format(mem_usage(b), mem_usage(a)))

    compare = pd.concat([b.dtypes, a.dtypes], axis=1)
    compare.columns = ["before", "after"]
    print(compare.apply(pd.Series.value_counts))


def get_column_types(dtypes):
    dtypes_col = dtypes.index
    dtypes_type = [i.name for i in dtypes.values]
    return dict(zip(dtypes_col, dtypes_type))


def show_dtypes(df):
    dt_dtpyes = df.select_dtypes(
        include=[np.datetime64, "datetime", "datetime64"]
    ).dtypes
    print("datetime_cols:")
    pprint(get_column_types(dt_dtpyes))
    g_dtypes = df.drop(dt_dtpyes.index, axis=1).dtypes
    print("other_cols:")
    pprint(get_column_types(g_dtypes))


def optimize_numeric_values(df, fillna=None, as_int=None):
    optimized_df = df.head(0)
    df_int = df.select_dtypes(include="int")
    # TODO: need to handle unsigned dtype
    converted_int = df_int.apply(pd.to_numeric, downcast="integer")

    # if mem_usage(df_int) > ...

    df_float = df.select_dtypes(include="float")
    if fillna is None or not as_int:
        converted_float = df_float.apply(pd.to_numeric, downcast="float")
    else:
        converted_float = (
            df_float.fillna(fillna).astype(int).apply(pd.to_numeric, downcast="integer")
        )

    optimized_cols = converted_int.columns.tolist() + converted_float.columns.tolist()
    not_optimized_cols = df.columns[~df.columns.isin(optimized_cols)]

    # merge
    optimized_df[df_int.columns] = converted_int
    optimized_df[df_float.columns] = converted_float
    optimized_df[not_optimized_cols] = df[not_optimized_cols]

    return optimized_df
