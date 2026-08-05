"""
Logistic regression baseline for throat cancer classification.

Extracts OpenSMILE eGeMAPSv02 functionals (88-dim paralinguistic voice
descriptors) per recording, concatenates them with each patient's
demographic/symptom background, and fits a logistic regression classifier
on the fused feature vector. Evaluated on sensitivity, specificity, and
AUROC, matching the other baselines in this project.

Baseline from Paterson et al., (2025): https://arxiv.org/abs/2412.16267v2

Run from `src/benchmarks/`:
    python logisticreg.py --use_demographics
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
import opensmile
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import Config

import warnings
warnings.filterwarnings("ignore")

config = Config()

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)


def load_patients():
    """
    Load every .wav under `config.dataset_path/{benign,malignant,normal}`,
    extract OpenSMILE eGeMAPSv02 functionals, and attach each patient's
    demographic/symptom row from that class's `medicalhistory.xlsx`.

    Returns:
        dict mapping patient ID -> {"audio", "background", "label"}.
    """
    patients = {}
    for voice_type in ["benign", "malignant", "normal"]:
        folder = os.path.join(config.dataset_path, voice_type)
        for sample in os.listdir(folder):
            if sample.split(".")[-1] != "wav":
                continue
            user_id = sample.split(".")[0]
            signal, _ = librosa.load(os.path.join(folder, sample), sr=config.sampling_rate)
            functionals = smile.process_signal(signal, config.sampling_rate)
            patients[user_id] = {
                "audio": functionals.values.flatten(),
                "label": 1 if voice_type == "malignant" else 0,
                "background": None,
            }

        history = pd.read_excel(os.path.join(folder, "medicalhistory.xlsx")).fillna(0)
        for _, row in history.iterrows():
            patients[row["ID"]]["background"] = row.drop(["ID", "Disease category"])

    return patients


def cross_validation(patients):
    """
    Split patients into train/test and z-score normalize non-binary
    demographic columns, fitting the scaler on the train split only to
    avoid test-set leakage.
    """
    patient_list = list(patients.values())
    train_data, test_data = train_test_split(patient_list, test_size=0.2, random_state=config.seed)

    bg_cols = train_data[0]["background"].index
    quant_cols = [col for col in bg_cols
                  if pd.Series([p["background"][col] for p in train_data]).nunique() > 2]

    train_bg = pd.DataFrame([p["background"] for p in train_data])
    scaler = StandardScaler()
    scaler.fit(train_bg[quant_cols])
    for split in [train_data, test_data]:
        for p in split:
            bg = p["background"].copy()
            bg[quant_cols] = scaler.transform(bg[quant_cols].values.reshape(1, -1).astype(float))[0]
            p["background"] = bg

    return train_data, test_data


def build_dataset(data, use_demographics):
    """Stack per-patient audio (+ optionally demographic) features into (X, y)."""
    X_audio = np.stack([p["audio"] for p in data])
    if use_demographics:
        X_demo = np.stack([p["background"].values.astype(float) for p in data])
        X = np.concatenate([X_audio, X_demo], axis=1)
    else:
        X = X_audio
    y = np.array([p["label"] for p in data])
    return X, y


def visualize(all_labels, all_preds, file_name="logisticreg_results"):
    """Plot a confusion matrix, saved to `graphs/{file_name}.png`."""
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["benign/normal", "malignant"])
    disp.plot(colorbar=False)
    plt.title("Logistic Regression Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"../graphs/{file_name}.png", dpi=150)
    plt.show()


def parse_args():
    parser = ArgumentParser(description="Train logistic regression baseline (OpenSMILE audio + demographics)")
    parser.add_argument("--use_demographics", action="store_true",
                         help="Concatenate demographic/symptom features alongside OpenSMILE audio features")
    parser.add_argument("--results_name", type=str, default="logisticreg_results",
                         help="What to name the result diagram")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading dataset and extracting OpenSMILE features...")
    patients = load_patients()
    train_data, test_data = cross_validation(patients)

    X_train, y_train = build_dataset(train_data, args.use_demographics)
    X_test, y_test = build_dataset(test_data, args.use_demographics)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
    }
    grid = GridSearchCV(
        LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=5000),
        param_grid, scoring="roc_auc", cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"Best params: {grid.best_params_}")
    model = grid.best_estimator_

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auroc = roc_auc_score(y_test, probs)

    print(f"Test sensitivity: {sensitivity:.3f} specificity: {specificity:.3f} auroc: {auroc:.3f}")

    visualize(y_test, preds, args.results_name)


if __name__ == '__main__':
    main()