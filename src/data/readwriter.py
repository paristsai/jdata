import gc
from abc import ABC

import numpy as np
import pandas as pd
import yaml
from pyarrow.feather import read_feather

from .. import PROJECT_DIR, DATA_DIR, DEFAULT_VERSION, to_unix


class ValueExistError(Exception):
    """Raised when the value existed in the original list/dict/file"""

    pass


class DataReadWriter(ABC):
    """
    Base Data Reader/Writer.
    """

    def read(self, *args):
        raise NotImplementedError

    def write(self, *args):
        raise NotImplementedError


class EmptyReadWriter(DataReadWriter):
    """
    Do Nothing Reader/Writer.
    """

    def read(self):
        pass

    def write(self):
        pass


class FileReadWriter(DataReadWriter):
    """
    Base File Reader/Writer.
    """

    project_dir = PROJECT_DIR
    data_dir = DATA_DIR
    config_dir = PROJECT_DIR / "src" / "data" / "config"
    config_file = None
    path_dict: dict = {}

    @property
    def extension(self):
        return None

    def __init__(self):
        self.set_path_dict()
        self.create_folder_if_not_exist()
        self.create_file_if_not_exist()

    def read(self, dir, name):
        raise NotImplementedError

    def write(self, df, dir, name):
        raise NotImplementedError

    def extend_name(self, name: str):
        if self.extension and not name.endswith(self.extension):
            return "{name}.{extension}".format(name=name, extension=self.extension)

    def fpath(self, dir, name):
        p = self.path_dict[dir] / name
        return p.resolve()

    def set_path_dict(self):
        for dir in ("raw", "interim", "processed", "external"):
            self.path_dict[dir] = self.data_dir.joinpath(dir)

    def create_folder_if_not_exist(self):
        if not self.config_dir.is_dir():
            self.config_dir.mkdir()

    def create_file_if_not_exist(self):
        if self.config_file and not self.config_file.is_file():
            self.config_file.touch()


class CSVReadWriter(FileReadWriter):
    """
    CSV Format Files Helper
    """

    @property
    def extension(self):
        return "csv"

    def __init__(self):
        self.schema = Schema()
        super().__init__()

    def read(self, dir, name, version, *args):
        schema = self.schema.get(name, version)
        return pd.read_csv(self.fpath(dir, name), dtype=schema, *args)

    def write(self, df, dir, name, version, save_dtypes=False, *args):
        df.to_csv(self.fpath(dir, name), index=False, *args)
        if save_dtypes:
            self.save_dtypes(name, df.dtypes, version)

    def save_dtypes(self, key, value, version):
        self.schema.save(key, value, version)


class FeatherReadWriter(FileReadWriter):
    """
    Feather Format Files Helper
    """

    @property
    def extension(self):
        return "feather"

    def __init__(self):
        super().__init__()

    def read(self, dir, name, nthreads=4):
        # TODO: https://github.com/pandas-dev/pandas/issues/23053
        # return pd.read_feather(self.fpath(dir, name), nthreads=nthreads)
        return read_feather(self.fpath(dir, name), use_threads=True)

    def write(self, df, dir, name):
        df.to_feather(self.fpath(dir, name))


class NpyReadWriter(FileReadWriter):
    pass


class PickleReadWriter(FileReadWriter):
    pass


class HDF5ReadWriter(FileReadWriter):
    pass


class DaskReadWriter(FileReadWriter):
    pass


class CompositeReadWriter(DataReadWriter):
    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer

    def read(self, *args):
        self.reader.read(*args)

    def write(self, df, *args):
        self.writer.write(df, *args)


class Schema(FileReadWriter):
    def __init__(self):
        self.config_file = self.config_dir / "schema.yaml"
        try:
            with self.config_file.open() as f:
                self.config = yaml.load(f)
        except yaml.YAMLError as exc:
            print("Error in configuration file:", exc)

    def get(self, name, version=DEFAULT_VERSION):
        version = f"-{version}"
        return self.config.get("{0}{1}".format(name, version))

    def save(self, key, value, version=DEFAULT_VERSION):
        self.config.update({key: value})
        with self.config_file.open("w") as outfile:
            yaml.dump(self.config, outfile, default_flow_style=False)


class TrainTestReadWriter(CSVReadWriter):
    config: dict = {}
    config_template = {
        "input": {"folder": "raw", "name": "example_file.csv"},
        "output": {
            "folder": "interim",
            "train": "example_file_train_1.0.npy",
            "train_only_index": True,
            "test": "example_file_test_1.0.csv",
        },
        "setting": {
            "time_col": "time",
            "start": "2018-01-01",
            "mid": "2018-01-03",
            "end": "2018-01-05",
            "unix_time": False,
        },
        "version": "1.0",
    }

    def __init__(self, *args):
        super().__init__(*args)

        self.config_file = self.config_dir / "train_test.yaml"
        q = self.config_file

        if not q.exists():
            self.write("example", self.config_template)

        try:
            with q.open() as f:
                self.config = yaml.load(f)
        except yaml.YAMLError as exc:
            print("Error in configuration file:", exc)

    def read(self, key, version=DEFAULT_VERSION):
        vkey = "{}-{}".format(key, version)
        c = self.config.get(vkey, {}).copy()
        if not c:
            raise KeyError(
                "Config doesn't contain the key {} with version {}".format(key, version)
            )
        infolder = c["input"]["folder"]
        outfolder = c["output"]["folder"]
        outdir = self.path_dict[outfolder]

        # load train
        if c["output"]["train_only_index"]:
            train_index_p = outdir / c["output"]["train"]
            with np.load(train_index_p) as data:
                train_index = data["index"]
            df = super().read(infolder, c["input"]["name"], version)
            train = df.iloc[train_index, :]
            del df
            gc.collect()
        else:
            train = super().read(outdir, c["output"]["train"], version)

        # load test
        test_path = outdir.joinpath(c["output"]["test"])
        test = pd.read_csv(test_path)

        return train, test

    def write(self, key, value, version=DEFAULT_VERSION, force=False):
        if key in self.config and force is False:
            raise ValueExistError(
                "Dictionary key: {0} existed. You could use force=True"
                " to overwrite original value".format(key)
            )
        vkey = "{}-{}".format(key, version)
        self.config.update({vkey: value})
        with self.config_file.open("w") as outfile:
            yaml.dump(self.config, outfile, default_flow_style=False)

    def get_config(self, key=None, version=DEFAULT_VERSION):
        config = self.config.copy()
        if key:
            vkey = "{}-{}".format(key, version)
            config = config.get(vkey)
            config["start"] = pd.to_datetime(config["setting"]["start"])
            config["mid"] = pd.to_datetime(config["setting"]["mid"])
            config["end"] = pd.to_datetime(config["setting"]["end"])
        return config

    def show(self, key=None, version=DEFAULT_VERSION):
        c = self.get_config(key, version)
        print(yaml.dump(c, default_flow_style=False))


class JDataTrainTestReadWriter(TrainTestReadWriter):
    def __init__(self, *args):
        super().__init__(*args)

    def read(self, key, version=DEFAULT_VERSION, encode_type=True, add_feature=True):
        train, test = super().read(key, version)
        if encode_type:
            types = pd.get_dummies(data=train.type)
            types.columns = [
                "browse",
                "add_cart",
                "rem_cart",
                "purchase",
                "follow",
                "click",
            ]
            train[types.columns] = types
            del types
        if add_feature:
            end_time: pd.Timestamp = pd.to_datetime(
                super().get_config(key)["setting"]["end"], format="%Y-%m-%d %H:%M:%S"
            )
            reg_secs = to_unix(end_time) - train.unix_user_reg
            train["reg_days"] = reg_secs // (60 * 60 * 24)
        return train, test
