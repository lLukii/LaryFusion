"""
SVM baseline for throat cancer classification, adapted from Rehman et al.,
"Voice disorder detection using machine learning algorithms: An application
in speech and language pathology", Engineering Applications of Artificial
Intelligence 133 (2024) 108047 (https://doi.org/10.1016/j.engappai.2024.108047),
whose SVM was the best-performing classifier they tested (up to 99.97%
accuracy with PCA-selected features).

Reproduces their 11-dimensional feature set (Table 3 / Section 3.3): patient
age plus ten Praat-derived acoustic perturbation measures - fundamental
frequency (F0), periodicity (harmonics-to-noise ratio), four jitter variants
(absolute, local, RAP, PPQ5), and four shimmer variants (absolute/local_dB,
local, APQ3, APQ5) - computed per recording via `parselmouth` (the Praat
bindings), then fits an SVM (RBF/polynomial kernel, tuned via GridSearchCV).
`class_weight="balanced"` is used to counter the benign/normal vs. malignant
imbalance, per project convention. Positive class (1) = malignant, negative
class (0) = benign/normal, matching the rest of this project's label
convention.

Run from `src/`:
    python svm.py
"""

import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.preprocessing import StandardScaler

from config import Config
from dataset import load_patients

import warnings
warnings.filterwarnings("ignore")

config = Config()


def build_dataset(data):
    """Stack per-patient [age, ...acoustic] feature vectors into (X, y)."""
    X = np.stack([np.concatenate(([float(p["background"]["Age"])], p["signal"])) for p in data])
    y = np.array([p["label"] for p in data])
    return X, y


def visualize(all_labels, all_preds, file_name="svm_results"):
    """Plot a confusion matrix, saved to `graphs/{file_name}.png`."""
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["benign/normal", "malignant"])
    disp.plot(colorbar=False)
    plt.title("SVM Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"../graphs/{file_name}.png", dpi=150)
    plt.show()


def parse_args():
    parser = ArgumentParser(description="Train SVM baseline (Praat jitter/shimmer/F0/periodicity + age)")
    parser.add_argument("--results_name", type=str, default="svm_results", help="What to name the result diagram")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading dataset and extracting Praat acoustic features...")
    patients = load_patients(feature_type="praat", include_demographics=True)
    patient_list = list(patients.values())
    train_data, test_data = train_test_split(patient_list, test_size=0.2, random_state=config.seed)

    X_train, y_train = build_dataset(train_data)
    X_test, y_test = build_dataset(test_data)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.4, 0.8, 1.6, 3.2],
        "kernel": ["rbf", "poly"],
    }
    grid = GridSearchCV(SVC(class_weight="balanced"), param_grid, scoring="roc_auc", cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"Best params: {grid.best_params_}")
    model = grid.best_estimator_

    preds = model.predict(X_test)
    scores = model.decision_function(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auroc = roc_auc_score(y_test, scores)

    print(f"Test sensitivity: {sensitivity:.3f} specificity: {specificity:.3f} auroc: {auroc:.3f}")

    visualize(y_test, preds, args.results_name)


if __name__ == '__main__':
    main()
