"""
1D-CNN baseline for throat cancer classification, adapted from Kim et al., https://doi.org/10.3390/jcm9113415.

Operates directly on the raw waveform (no hand-crafted features). Six
convolution blocks with channel counts 16/32/64/128/256/256 (matching the
paper), each followed by BatchNorm + ReLU; the last five blocks are also
max-pooled (paper: pool sizes 8, 8, 8, 8, 4). The paper tuned its fixed
110,000-sample input and final pooled length of 6 to its own 22050 Hz
recordings; ours are 3s at 16kHz (`config.padding` samples), so the final
block uses an AdaptiveMaxPool1d(6) instead of a fixed-size pool to still
land on the paper's 6 x 256 = 1536 flattened features feeding the paper's
dense head (1536 -> 100 -> 50 -> 2). Audio-only, matching the paper.

Run from `src/`:
    python cnn1d.py --name cnn1d_v1
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, average_precision_score

from config import Config
from dataset import load_patients, stratified_kfold, batch_inputs
from models import CNN1D

import warnings
warnings.filterwarnings("ignore")

config = Config()
torch.manual_seed(1337)


class FocalLoss(nn.Module):
    """
    Focal loss (Lin et al., 2017) for the class-imbalanced benign/normal vs.
    malignant split: down-weights easy, well-classified examples (via
    `(1-pt)**gamma`) and reweights classes by `alpha`/`1-alpha`.
    """
    def __init__(self, gamma=2.0, alpha=0.969):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = torch.where(labels == 1, self.alpha, 1 - self.alpha)
        return (alpha_t * (1 - pt) ** self.gamma * ce).mean()


def train_epoch(loader, model, optimizer, loss_fn):
    """Run one training pass over `loader`, returning (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    for signals, demographics, labels in tqdm(loader, desc="Train", leave=False):
        signals = signals.to(config.device)
        labels = labels.to(config.device).long()
        optimizer.zero_grad()
        logits = model(signals, demographics)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += len(labels)

    return total_loss / len(loader), correct / total


def eval_epoch(loader, model, loss_fn):
    """
    Run one no-grad evaluation pass over `loader`.

    Returns:
        (avg_loss, accuracy, sensitivity, specificity, auprc, preds, labels)
    """
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for signals, demographics, labels in tqdm(loader, desc="Eval", leave=False):
            signals = signals.to(config.device)
            labels = labels.to(config.device).long()

            logits = model(signals, demographics)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            probs = F.softmax(logits, dim=-1)[:, 1]
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auprc = average_precision_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return total_loss / len(loader), correct / total, sensitivity, specificity, auprc, all_preds, all_labels


def visualize(train_losses, val_losses, train_accs, val_accs, all_preds, all_labels, file_name="cnn1d_results"):
    """Plot loss/accuracy curves and a confusion matrix, saved to `graphs/{file_name}.png`."""
    epochs = range(1, len(train_losses) + 1)
    _, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs, train_losses, label="Train")
    axes[0].plot(epochs, val_losses, label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_accs, label="Train")
    axes[1].plot(epochs, val_accs, label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["benign/normal", "malignant"])
    disp.plot(ax=axes[2], colorbar=False)
    axes[2].set_title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig(f"../graphs/{file_name}.png", dpi=150)
    plt.show()


def parse_args():
    parser = ArgumentParser(description="Train 1D-CNN baseline (raw waveform, Kim et al. 2020 architecture)")
    parser.add_argument("--name", type=str, required=True, help="Name of your model")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of stratified k-fold splits")
    parser.add_argument("--results_name", type=str, default="cnn1d_results", help="What to name the result diagram")
    return parser.parse_args()


def main():
    """
    Train one CNN1D per stratified k-fold split, checkpointing each fold's
    best epoch (by validation loss) to `checkpoints/{name}_fold{k}_best.pt`,
    with early stopping after `config.patience` epochs of no improvement.
    Reports mean +/- std metrics across folds.
    """
    args = parse_args()

    print("Loading dataset...")
    patients = load_patients(feature_type="raw")

    fold_metrics = []
    for fold, (train_data, test_data) in enumerate(stratified_kfold(patients, n_splits=args.n_splits), start=1):
        print(f"\n=== Fold {fold}/{args.n_splits} ===")
        train_loader = batch_inputs(train_data, stratify=True)
        test_loader = batch_inputs(test_data)

        model = CNN1D().to(config.device)
        loss_fn = FocalLoss().to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        if fold == 1:
            print(f"Model layout: {model}")
            print("Trainable parameters:", sum(p.numel() for p in model.parameters()))

        best_loss = float('inf')
        best_metrics = None
        no_improv = 0
        for epoch in range(1, config.num_epochs + 1):
            tr_loss, tr_acc = train_epoch(train_loader, model, optimizer, loss_fn)
            val_loss, val_acc, val_sens, val_spec, val_auprc, preds, labels = eval_epoch(test_loader, model, loss_fn)

            train_losses.append(tr_loss)
            val_losses.append(val_loss)
            train_accs.append(tr_acc)
            val_accs.append(val_acc)

            print(f"Epoch {epoch:02d} | "
                f"Train loss: {tr_loss:.4f} acc: {tr_acc:.3f} | "
                f"Val loss: {val_loss:.4f} acc: {val_acc:.3f} sensitivity: {val_sens:.3f} specificity: {val_spec:.3f} auprc: {val_auprc:.3f}")

            if val_loss < best_loss:
                print("Updating best model...")
                best_loss = val_loss
                no_improv = 0
                best_metrics = (val_acc, val_sens, val_spec, val_auprc)
                torch.save(model.state_dict(), f"../checkpoints/{args.name}_fold{fold}_best.pt")
                visualize(train_losses, val_losses, train_accs, val_accs, preds, labels, f"{args.results_name}_fold{fold}")
            else:
                no_improv += 1
                if no_improv >= config.patience:
                    print(f"No improvement in val loss for {config.patience} epochs, stopping fold {fold} early.")
                    break

        fold_metrics.append(best_metrics)

    accs, senss, specs, auprcs = zip(*fold_metrics)
    print("\n=== K-Fold Summary ===")
    print(f"Accuracy:    {np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    print(f"Sensitivity: {np.mean(senss):.3f} +/- {np.std(senss):.3f}")
    print(f"Specificity: {np.mean(specs):.3f} +/- {np.std(specs):.3f}")
    print(f"AUPRC:       {np.mean(auprcs):.3f} +/- {np.std(auprcs):.3f}")


if __name__ == '__main__':
    main()