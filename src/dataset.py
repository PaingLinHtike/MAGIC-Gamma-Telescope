from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_DIR / "data" / "raw" / "magic04.data"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

COLUMNS = [
    "fLength",
    "fWidth",
    "fSize",
    "fConc",
    "fConc1",
    "fAsym",
    "fM3Long",
    "fM3Trans",
    "fAlpha",
    "fDist",
    "class",
]


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH, names=COLUMNS)
    df["class"] = (df["class"] == "g").astype(int)
    return df


def plot_feature_distributions(df: pd.DataFrame) -> None:
    for label in COLUMNS[:-1]:
        plt.hist(df[df["class"] == 1][label], density=True, color="blue", alpha=0.5, label="gamma")
        plt.hist(df[df["class"] == 0][label], density=True, color="red", alpha=0.5, label="hadron")
        plt.xlabel(label)
        plt.ylabel("Probability")
        plt.title(label)
        plt.legend()
        plt.show()


def scale_dataset(dataframe: pd.DataFrame, over_sample: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if over_sample:
        sampler = RandomOverSampler(random_state=42)
        X, y = sampler.fit_resample(X, y)

    data = np.hstack((X, y.reshape(-1, 1)))
    return data, X, y


def build_processed_datasets(show_plots: bool = False) -> None:
    df = load_raw_data()

    if show_plots:
        plot_feature_distributions(df)

    shuffled = df.sample(frac=1, random_state=42)
    train_end = int(0.6 * len(shuffled))
    valid_end = int(0.8 * len(shuffled))
    train = shuffled.iloc[:train_end]
    valid = shuffled.iloc[train_end:valid_end]
    test = shuffled.iloc[valid_end:]

    train, _, _ = scale_dataset(train, over_sample=True)
    valid, _, _ = scale_dataset(valid, over_sample=False)
    test, _, _ = scale_dataset(test, over_sample=False)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / "train.npy", train)
    np.save(PROCESSED_DIR / "valid.npy", valid)
    np.save(PROCESSED_DIR / "test.npy", test)

    print(f"Saved processed datasets to {PROCESSED_DIR}")


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Prepare the MAGIC Gamma Telescope dataset for modeling.")
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display feature distribution plots while preparing the data.",
    )
    return parser


if __name__ == "__main__":
    args = parse_args().parse_args()
    build_processed_datasets(show_plots=args.show_plots)
