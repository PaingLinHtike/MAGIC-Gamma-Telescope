# MAGIC Gamma Telescope Classification

This project trains machine learning models to classify events from the MAGIC Gamma Telescope dataset as either gamma signal events or hadron background events.

The workflow starts from the raw `magic04.data` file, creates processed NumPy datasets, trains several classifiers, and evaluates their performance on a held-out test set.

## Dataset

The dataset contains simulated high-energy particle events observed by the MAGIC imaging atmospheric Cherenkov telescope. Each row has 10 continuous features and one class label:

- `g`: gamma event, encoded as `1`
- `h`: hadron event, encoded as `0`

The repository includes both the raw dataset and the generated processed splits:

- `data/raw/magic04.data`
- `data/processed/train.npy`
- `data/processed/valid.npy`
- `data/processed/test.npy`

## Models

The training script builds and saves these models:

- K-Nearest Neighbors
- Gaussian Naive Bayes
- Logistic Regression
- Support Vector Machine
- TensorFlow/Keras neural network

Classical scikit-learn models are saved as `.pkl` files in `models/`. The neural network is saved as `models/neural_network_model.keras`.

## Setup

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate fcc-MAGIC-example
```

If the environment already exists, update it:

```bash
conda env update --file environment.yml --prune
conda activate fcc-MAGIC-example
```

## Usage

Run commands from the project root.

Prepare the dataset:

```bash
python src/dataset.py
```

To also display feature distribution plots:

```bash
python src/dataset.py --show-plots
```

Train all models:

```bash
python src/modeling/train.py
```

For a faster neural-network training pass while experimenting:

```bash
python src/modeling/train.py --epochs 10
```

Evaluate trained models:

```bash
python src/modeling/evaluate.py
```

To show confusion matrix plots:

```bash
python src/modeling/evaluate.py --show-plots
```

Run neural-network predictions on the test split:

```bash
python src/modeling/predict.py --limit 20
```

## Project Structure

```text
.
|-- data
|   |-- external
|   |-- interim
|   |-- processed
|   |   |-- test.npy
|   |   |-- train.npy
|   |   `-- valid.npy
|   `-- raw
|       `-- magic04.data
|-- models
|-- notebooks
|-- references
|-- reports
|   `-- figures
|-- src
|   |-- dataset.py
|   |-- features.py
|   |-- modeling
|   |   |-- evaluate.py
|   |   |-- predict.py
|   |   `-- train.py
|   |-- plots.py
|   `-- services
|-- environment.yml
|-- pyproject.toml
`-- README.md
```

## Notes

- `models/` is ignored by Git, so trained model artifacts are generated locally.
- Raw and processed data files are committed so the project can be inspected and run without an extra download step.
- `node_modules/` is included in this repository because it was part of the requested publish scope.
