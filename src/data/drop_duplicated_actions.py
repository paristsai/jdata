import pandas as pd

from .. import DATA_DIR


def main():
    interim_path = DATA_DIR.joinpath("interim")
    action = pd.read_csv(
        "{}/All_Action.csv".format(interim_path),
        parse_dates=["time"],
        infer_datetime_format=True,
    )
    dedup_action = action[~action.duplicated()]
    dedup_action.to_csv("{}/Dedup_All_Action.csv".format(interim_path), index=False)


if __name__ == "__main__":
    main()
