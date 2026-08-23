"""
Logistic regression baseline for throat cancer classification, reproducing
the best-performing configuration from Paterson et al. (2025), "A
Classification Benchmark for Artificial Intelligence Detection of Laryngeal
Cancer from Patient Voice" (https://arxiv.org/abs/2412.16267v2, code at
https://github.com/mary-paterson/LaryngealCancerClassificationBenchmark):
on their FEMH holdout set). Their FEMH dataset splits demographics and
symptoms into separate tables; this project's `medicalhistory.xlsx` already
bundles both (age/sex alongside symptom columns), so including "background"
here reproduces their combined "Voice + Demographics + Symptoms" setup.

Preprocessing follows their Figure 4 pipeline:
    - Audio features: mean-imputation (their eGeMAPS F0-based descriptors
      can be missing for very hoarse voices) -> z-score normalization ->
      decision-tree-based feature selection (`SelectFromModel`).
    - Demographic/symptom features: missing values filled with 0 (already
      done in `load_patients` below) -> z-score normalization of the
      non-binary columns (in `stratified_kfold`).
The two are concatenated and fed into a logistic regression tuned via
GridSearchCV over their exact Table 3 hyperparameter grid (penalty, C,
solver, max_iter, l1_ratio), with `class_weight="balanced"` and
`scoring="balanced_accuracy"`, both per their Section 3.3.

Evaluated via stratified k-fold cross-validation rather than a single train/
test split, given how few malignant examples exist to hold out.

Run from `src/`:
    python logisticreg.py
"""

import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
)
from sklearn.preprocessing import StandardScaler

from config import Config
from dataset import load_patients, stratified_kfold

import warnings
warnings.filterwarnings("ignore")

config = Config()


def build_matrices(data):
    """Stack per-patient audio feature vectors, demographic vectors, and labels."""
    X_audio = np.stack([p["signal"] for p in data])
    X_demo = np.stack([p["background"].values.astype(float) for p in data])
    y = np.array([p["label"] for p in data])
    return X_audio, X_demo, y


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
    parser = ArgumentParser(description="Train logistic regression baseline (OpenSMILE audio + demographics/symptoms)")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of stratified k-fold splits")
    parser.add_argument("--results_name", type=str, default="logisticreg_results",
                         help="What to name the result diagram")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading dataset and extracting OpenSMILE features...")
    patients = load_patients(feature_type="opensmile", include_demographics=True)

    fold_metrics = []
    for fold, (train_data, test_data) in enumerate(stratified_kfold(patients, n_splits=args.n_splits), start=1):
        print(f"\n=== Fold {fold}/{args.n_splits} ===")
        X_train_audio, X_train_demo, y_train = build_matrices(train_data)
        X_test_audio, X_test_demo, y_test = build_matrices(test_data)

        # audio branch: mean-impute -> z-score -> decision-tree feature selection
        imputer = SimpleImputer(strategy="mean")
        X_train_audio = imputer.fit_transform(X_train_audio)
        X_test_audio = imputer.transform(X_test_audio)

        audio_scaler = StandardScaler()
        X_train_audio = audio_scaler.fit_transform(X_train_audio)
        X_test_audio = audio_scaler.transform(X_test_audio)

        selector = SelectFromModel(DecisionTreeClassifier(random_state=config.seed))
        X_train_audio = selector.fit_transform(X_train_audio, y_train)
        X_test_audio = selector.transform(X_test_audio)
        print(f"Selected {X_train_audio.shape[1]}/{len(imputer.statistics_)} audio features")

        # demographic/symptom branch was already z-scored (non-binary cols) in stratified_kfold
        X_train = np.concatenate([X_train_audio, X_train_demo], axis=1)
        X_test = np.concatenate([X_test_audio, X_test_demo], axis=1)
        param_grid = {
            "C": [0.01], 
            "max_iter": [100], 
            "penalty": ["l2"], 
            "solver": ["liblinear"]
        }

        grid = GridSearchCV(
            LogisticRegression(class_weight="balanced"),
            param_grid, scoring="balanced_accuracy", cv=5, n_jobs=-1, verbose=2)
        grid.fit(X_train, y_train)

        print(f"Best params: {grid.best_params_}")
        model = grid.best_estimator_

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balanced_acc = (sensitivity + specificity) / 2

        print(f"Fold {fold} test sensitivity: {sensitivity:.3f} "
              f"specificity: {specificity:.3f} ba: {balanced_acc:.3f}")

        visualize(y_test, preds, f"{args.results_name}_fold{fold}")
        fold_metrics.append((sensitivity, specificity, balanced_acc))

    senss, specs, aurocs = zip(*fold_metrics)
    print("\n=== K-Fold Summary ===")
    print(f"Sensitivity: {np.mean(senss):.3f} +/- {np.std(senss):.3f}")
    print(f"Specificity: {np.mean(specs):.3f} +/- {np.std(specs):.3f}")
    print(f"AUROC:       {np.mean(aurocs):.3f} +/- {np.std(aurocs):.3f}")

if __name__ == '__main__':
    main()
