import numpy as np
import pandas as pd

from .. import DATA_DIR


def main():
    interim_path = DATA_DIR.joinpath("interim")

    df = pd.read_csv("{0}/All_Action.csv".format(interim_path))
    np.save("{0}/All_Action.npy".format(interim_path), df.values)
    print("{0}/All_Action.npy".format(interim_path))


if __name__ == "__main__":
    main()
