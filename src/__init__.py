import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

p = Path(__file__).resolve()
PROJECT_DIR = p.parents[1]
DATA_DIR = PROJECT_DIR.joinpath("data")
DEFAULT_VERSION = 1.0


def to_unix(dt) -> int:
    if isinstance(dt, pd.Timestamp):
        return dt.value / 10 ** 9
    return int(dt)


@contextmanager
def timer(context):
    start_time = time.time()
    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        print("{}: {:.3f} sec".format(context, elapsed_time))


def drop_duplicated_columns(df):
    distinct_cols = ~df.columns.duplicated()
    return df.iloc[:, distinct_cols]
