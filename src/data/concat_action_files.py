"""
Concat JData Action Files And Save
"""

import pandas as pd

from .. import DATA_DIR


def main(input_dir=DATA_DIR.joinpath("raw"), output_dir=DATA_DIR.joinpath("interim")):
    file_list = list(input_dir.glob("JData_Action*.csv"))
    df_list = [pd.read_csv(p) for p in file_list]
    actions = pd.concat(df_list)
    actions.to_csv("{}/All_Action.csv".format(output_dir), index=False)


if __name__ == "__main__":
    main()
