from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"


def predict(limit: int) -> None:
    model = tf.keras.models.load_model(MODELS_DIR / "neural_network_model.keras")

    test = np.load(PROCESSED_DIR / "test.npy")
    X_test, y_test = test[:, :-1], test[:, -1]

    probabilities = model.predict(X_test).flatten()
    y_pred = np.where(probabilities > 0.5, "gamma", "hadron")
    y_true = np.where(y_test == 1, "gamma", "hadron")

    print("Predicted labels:", y_pred[:limit])
    print("True labels:", y_true[:limit])


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Predict MAGIC labels with the trained neural network.")
    parser.add_argument("--limit", type=int, default=20, help="Number of predictions to print.")
    return parser


if __name__ == "__main__":
    args = parse_args().parse_args()
    predict(limit=args.limit)
