import pandas as pd

from .. import DATA_DIR
from .optimizer import optimize_numeric_values
from .readwriter import CSVReadWriter


def load_files(input_dir, output_dir):

    column_types = {
        "user_id": "int32",
        "sku_id": "int32",
        "model_id": "int16",
        "type": "int8",
        "cate": "int8",
        "brand": "int16",
    }
    optimized_action: pd.DataFrame = pd.read_csv(
        "{}/Dedup_All_Action.csv".format(output_dir),
        dtype=column_types,
        parse_dates=["time"],
        infer_datetime_format=True,
    ).rename(index=str, columns={"time": "dt"})

    column_types = {
        "user_id": "int32",
        "age": "int8",
        "sex": "int8",
        "user_lv_cd": "int8",
    }
    optimized_user = pd.read_csv(
        "{}/user.csv".format(output_dir),
        dtype=column_types,
        parse_dates=["user_reg_tm"],
        infer_datetime_format=True,
    ).rename(index=str, columns={"user_reg_tm": "dt"})

    column_types = {
        "sku_id": "int32",
        "a1": "int8",
        "a2": "int8",
        "a3": "int8",
        "cate": "int8",
        "brand": "int16",
    }
    optimized_product = pd.read_csv(
        "{}/JData_Product.csv".format(input_dir),
        dtype=column_types,
        usecols=["sku_id", "a1", "a2", "a3"],
    )

    column_types = {
        "sku_id": "int32",
        "comment_num": "int8",
        "has_bad_comment": "int8",
        "bad_comment_rate": "float32",
    }
    optimized_comment = pd.read_csv(
        "{}/JData_Comment.csv".format(input_dir),
        dtype=column_types,
        parse_dates=["dt"],
        infer_datetime_format=True,
    )

    return (optimized_action, optimized_user, optimized_product, optimized_comment)


def add_dt_features(df: pd.DataFrame, dt_col, features) -> pd.DataFrame:
    """Add DataTime Features to DataFrame.

    Args:
        df: Pandas DataFrame
        dt_col: The column with dtype datetime64[ns]
        features: unix and other datetime properties

    Return:
        DataFrame
    """
    dts = df[dt_col]
    dts_prop = dts.dt

    properties = [
        "year",
        "month",
        "day",
        "week",
        "hour",
        "miniute",
        "second",
        "date",
        "time",
    ]

    m = [f for f in features if f in properties]
    for f in m:
        df[f] = pd.to_numeric(getattr(dts_prop, f), downcast="unsigned")

    if "unix" in features:
        df["unix"] = dts.astype(int) // 10 ** 9

    return df


def main(input_dir=DATA_DIR.joinpath("raw"), output_dir=DATA_DIR.joinpath("interim")):
    action, user, product, comment = load_files(input_dir, output_dir)

    # add_dt_features
    action = add_dt_features(
        action, "dt", ["unix", "year", "month", "day", "week", "hour"]
    )
    user = add_dt_features(user, "dt", ["unix", "year", "month", "day"])
    comment = add_dt_features(comment, "dt", ["unix", "year", "month", "day", "week"])

    # drop original dt string
    action = action.drop("dt", axis=1)
    user = user.drop("dt", axis=1)
    comment = comment.drop("dt", axis=1)

    # process datetime columns
    merged = (
        action.merge(user, how="left", on="user_id", suffixes=["", "_user_reg"])
        .fillna(-1)
        .merge(product, how="left", on=["sku_id"])
        .fillna(-1)
        .merge(
            comment,
            how="left",
            on=["sku_id", "year", "month", "day"],
            suffixes=["_action", "_comment"],
        )
        .fillna(0)
    )

    merged = optimize_numeric_values(merged)

    print("total: {}".format(len(merged)))
    print(merged.info())
    print(merged.dtypes)

    rw = CSVReadWriter()
    rw.write(merged, dir="interim", name="all_merged", version=1.0, save_dtypes=True)
    print("complete!")


if __name__ == "__main__":
    main()
