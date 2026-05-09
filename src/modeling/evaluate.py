from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import itertools

import joblib
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"
TARGET_NAMES = ["hadron", "gamma"]


def load_test_data() -> tuple[np.ndarray, np.ndarray]:
    test = np.load(PROCESSED_DIR / "test.npy")
    return test[:, :-1], test[:, -1]


def plot_confusion_matrix(cm: np.ndarray, title: str, cmap=plt.cm.Blues) -> None:
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(TARGET_NAMES))
    plt.xticks(tick_marks, TARGET_NAMES, rotation=45)
    plt.yticks(tick_marks, TARGET_NAMES)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def evaluate_models(show_plots: bool = False) -> None:
    X_test, y_test = load_test_data()

    models = {
        "KNN": joblib.load(MODELS_DIR / "knn_model.pkl"),
        "Naive Bayes": joblib.load(MODELS_DIR / "naive_bayes_model.pkl"),
        "Logistic Regression": joblib.load(MODELS_DIR / "logistic_regression_model.pkl"),
        "SVM": joblib.load(MODELS_DIR / "svm_model.pkl"),
    }

    neural_network = tf.keras.models.load_model(MODELS_DIR / "neural_network_model.keras")

    predictions = {name: model.predict(X_test) for name, model in models.items()}
    predictions["Neural Network"] = (neural_network.predict(X_test) > 0.5).astype("int32").flatten()

    for name, y_pred in predictions.items():
        print(f"Classification Report ({name}):")
        print(classification_report(y_test, y_pred, target_names=TARGET_NAMES))

        if show_plots:
            plot_confusion_matrix(
                confusion_matrix(y_test, y_pred),
                title=f"Confusion Matrix ({name})",
            )


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Evaluate trained MAGIC gamma/hadron classifiers.")
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display confusion matrix plots for each model.",
    )
    return parser


if __name__ == "__main__":
    args = parse_args().parse_args()
    evaluate_models(show_plots=args.show_plots)
