from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"


def load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(PROCESSED_DIR / f"{name}.npy")
    return data[:, :-1], data[:, -1]


def train_neural_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_nodes: int,
    dropout_prob: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(num_nodes, activation="relu", input_shape=(10,)),
            tf.keras.layers.Dropout(dropout_prob),
            tf.keras.layers.Dense(num_nodes, activation="relu"),
            tf.keras.layers.Dropout(dropout_prob),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0,
    )

    return model, history


def plot_history(history: tf.keras.callbacks.History) -> None:
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history.history["loss"], label="loss")
    ax1.plot(history.history["val_loss"], label="val_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary crossentropy")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(history.history["accuracy"], label="accuracy")
    ax2.plot(history.history["val_accuracy"], label="val_accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True)
    ax2.legend()

    plt.show()


def train_models(epochs: int, show_plots: bool) -> None:
    X_train, y_train = load_split("train")
    X_valid, y_valid = load_split("valid")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    knn_grid = GridSearchCV(
        KNeighborsClassifier(),
        {"n_neighbors": [1, 2, 5, 10]},
        cv=5,
        scoring="accuracy",
    )
    knn_grid.fit(X_train, y_train)

    models = {
        "knn_model.pkl": KNeighborsClassifier(n_neighbors=knn_grid.best_params_["n_neighbors"]),
        "naive_bayes_model.pkl": GaussianNB(),
        "logistic_regression_model.pkl": LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
        ),
        "svm_model.pkl": SVC(),
    }

    for filename, model in models.items():
        print(f"Training {filename}...")
        model.fit(X_train, y_train)
        joblib.dump(model, MODELS_DIR / filename)

    best_val_loss = float("inf")
    best_model = None

    for num_nodes in [16, 32, 64]:
        for dropout_prob in [0, 0.2]:
            for learning_rate in [0.01, 0.005, 0.001]:
                for batch_size in [32, 64, 128]:
                    print(
                        "Training neural network: "
                        f"{num_nodes} nodes, dropout {dropout_prob}, "
                        f"learning rate {learning_rate}, batch size {batch_size}"
                    )
                    model, history = train_neural_network(
                        X_train,
                        y_train,
                        num_nodes,
                        dropout_prob,
                        learning_rate,
                        batch_size,
                        epochs,
                    )

                    if show_plots:
                        plot_history(history)

                    val_loss = model.evaluate(X_valid, y_valid, verbose=0)[0]
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = model

    if best_model is None:
        raise RuntimeError("Neural network training did not produce a model.")

    best_model.save(MODELS_DIR / "neural_network_model.keras")
    print(f"Saved trained models to {MODELS_DIR}")


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Train models for MAGIC gamma/hadron classification.")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs per neural-network trial.")
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display neural-network training history plots.",
    )
    return parser


if __name__ == "__main__":
    args = parse_args().parse_args()
    train_models(epochs=args.epochs, show_plots=args.show_plots)
