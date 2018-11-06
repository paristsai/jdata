import gc
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from .. import DATA_DIR, DEFAULT_VERSION
from .readwriter import TrainTestReadWriter


class BaseSpliter(object):
    """
    Base Train/Test Spliter
    """

    def parse_dates(self, *args):
        return pd.to_datetime(args, format="%Y-%m-%d", errors="raise")

    def parse_sme_dates(self, start, mid, end):
        if any([isinstance(a, str) for a in (start, mid, end)]) is False:
            raise ValueError("time format error. correct: %Y-%m-%d")

        start, mid, end = self.parse_dates(start, mid, end)
        if not (start < mid and mid < end):
            raise ValueError(
                "Rule: start < mid and mid < end\n"
                "But start: {0}, mid: {1}, end: {2}".format(start, mid, end)
            )
        return start, mid, end

    def make_train(self, df):
        return df

    def make_test(self, df):
        return df

    def run(
        self,
        file_p,
        start,
        mid,
        end,
        time_col,
        unix_time=False,
        keep_cols=[],
        only_keep_train_index=True,
        version=DEFAULT_VERSION,
    ):

        p = Path(file_p)
        fixed_output_dir = DATA_DIR.joinpath("interim")
        if not p.is_file():
            raise FileNotFoundError("cannot file the file: {}".format(file_p))
        usecols = keep_cols + [time_col] if keep_cols else None

        start, mid, end = self.parse_sme_dates(start, mid, end)
        print("start: {0}, mid: {1}, end: {2}".format(start, mid, end))

        abs_path = p.resolve()
        if unix_time:
            df = pd.read_csv(abs_path, usecols=usecols)
            df[time_col] = pd.to_datetime(df[time_col], unit="s")
        else:
            df = pd.read_csv(abs_path, parse_dates=[time_col], usecols=usecols)

        train = self.make_train(df[(df[time_col] >= start) & (df[time_col] < mid)])
        test = self.make_test(df[(df[time_col] >= mid) & (df[time_col] < end)])

        del df
        gc.collect()

        train_suffix = ".npz" if only_keep_train_index else ".csv"
        fpath = fixed_output_dir.joinpath(p.name).resolve()
        fversion = f"{fpath.stem}_{version}"
        train_name = "{0}{1}{2}".format(fversion, "_train", train_suffix)
        test_name = "{0}{1}{2}".format(fversion, "_test", ".csv")
        train_path = str(fixed_output_dir.joinpath(train_name))
        test_path = str(fixed_output_dir.joinpath(test_name))

        setting_key = p.stem
        setting_value = {
            "input": {"folder": p.parent.name, "name": p.name},
            "output": {
                "folder": fixed_output_dir.name,
                "train": train_name,
                "train_only_index": only_keep_train_index,
                "test": test_name,
            },
            "setting": {
                "time_col": time_col,
                "start": str(start),
                "mid": str(mid),
                "end": str(end),
            },
            "version": version,
        }
        h = TrainTestReadWriter()
        h.write(setting_key, setting_value, version, force=False)

        print("config saved")
        print("start saving train...lenght={}".format(len(train)))
        if only_keep_train_index:
            with open(train_path, "wb") as outfile:
                np.savez_compressed(outfile, index=train.index.values)
        else:
            train.to_csv(train_path, columns=usecols, index=False)
        print("start saving test...lenght={}".format(len(test)))
        test.to_csv(test_path, index=False)


class JDataSpliter(BaseSpliter):
    def make_test(self, df):
        return df[df.type == 4][["user_id", "sku_id"]].drop_duplicates()

    def run(
        self,
        file_p,
        start,
        mid,
        end,
        time_col,
        unix_time=False,
        keep_cols=[],
        only_keep_train_index=True,
        version=DEFAULT_VERSION,
    ):
        return super().run(
            file_p,
            start,
            mid,
            end,
            time_col,
            unix_time,
            keep_cols,
            only_keep_train_index,
            version,
        )


if __name__ == "__main__":
    fire.Fire(JDataSpliter)
