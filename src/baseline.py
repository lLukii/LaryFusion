import numpy as np
import matplotlib.pyplot as plt
import librosa

from argparse import ArgumentParser
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from dataset import load_wavs, cross_validation
from config import Config

import warnings
warnings.filterwarnings("ignore")

config = Config()


def extract_features(patient):
    signal = patient["signal"].squeeze(0).numpy()
    mfcc = librosa.feature.mfcc(y=signal, sr=config.sampling_rate)
    return mfcc.flatten()


def build_dataset(data):
    X = np.stack([extract_features(p) for p in data])
    y = np.array([p["label"] for p in data])
    return X, y


def visualize(all_labels, all_preds):
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["benign/normal", "malignant"])
    disp.plot(colorbar=False)
    plt.title("SVM Confusion Matrix")
    plt.tight_layout()
    plt.savefig("graphs/svm_results.png", dpi=150)
    plt.show()


def parse_args():
    parser = ArgumentParser(description="Train baseline SVM classifier for throat cancer classification")
    parser.add_argument("--use_demographics", action="store_true", help="Whether to include demographic background features")
    return parser.parse_args()


if __name__ == '__main__':
    dataset = load_wavs()
    train_data, test_data = cross_validation(dataset)

    X_train, y_train = build_dataset(train_data)
    X_test, y_test = build_dataset(test_data)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto"],
        "kernel": ["rbf", "linear"],
    }
    grid = GridSearchCV(SVC(class_weight="balanced"), param_grid, scoring="roc_auc", cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"Best params: {grid.best_params_}")
    model = grid.best_estimator_

    preds = model.predict(X_test)
    scores = model.decision_function(X_test)
    acc = (preds == y_test).mean()
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auroc = roc_auc_score(y_test, scores)

    print(f"Test acc: {acc:.3f} sensitivity: {sensitivity:.3f} specificity: {specificity:.3f} auroc: {auroc:.3f}")

    visualize(y_test, preds)
