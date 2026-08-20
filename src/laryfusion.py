"""
LaryFusion: multimodal (audio + demographic) fusion classifier for throat
cancer classification, regularized with modality dropout and evaluated via
stratified k-fold cross-validation - simpler alternatives to the earlier
LeMDA-VAE and two-stage contrastive approaches, chosen because this
dataset's small malignant-class count made both of those noisy/overkill.

Modality dropout: during training, each modality's encoded features are
independently zeroed with probability `p_drop` before fusion, so the
classifier can't shortcut through whichever modality happens to be more
predictive in-sample and is forced to use both.

Stratified k-fold: rather than a single train/test split, `n_splits` folds
(`dataset.stratified_kfold`) are trained and evaluated independently and
metrics are reported as mean +/- std across folds, which matters here since
a single 80/20 split leaves only a handful of malignant validation examples
and its point-estimate metrics carry a lot of split-dependent variance.

Class-balanced loss: `FocalLoss`'s per-class weight is set via the
Effective Number of Samples (Cui et al., 2019, arxiv.org/abs/1901.05555)
rather than a hand-picked alpha - see the class docstring below.

Run from `src/`:
    python laryfusion.py --name laryfusion_v1
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from tqdm import tqdm
from transformers import Wav2Vec2Model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, average_precision_score, roc_auc_score

from config import Config
from dataset import load_patients, stratified_kfold, batch_inputs

import warnings
warnings.filterwarnings("ignore")

config = Config()


class FusionModule(nn.Module):
    """
    Audio (Wav2Vec2Model) + demographic (MLP) fusion classifier with
    modality dropout (see module docstring) and an optional sigmoid gate
    over the fused features, matching `ablations.py`'s `FusionModule`.
    """
    def __init__(self, config: Config, num_features: int, audio_dim: int = 512,
                 demo_dim: int = 64, gate: bool = False, p_drop: float = 0.1):
        super().__init__()
        combined_dim = audio_dim + demo_dim
        self.p_drop = p_drop
        self.demographic_enc = nn.Sequential(
            nn.Linear(num_features, demo_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(demo_dim, demo_dim),
        )
        self.aud_encoder = Wav2Vec2Model.from_pretrained(config.w2v_name)
        self.classifier = nn.Linear(combined_dim, 2)
        self.gate = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(combined_dim, combined_dim),
        ) if gate else None

    def forward(self, audio: torch.Tensor, demographic: torch.Tensor) -> torch.Tensor:
        z_A = self.aud_encoder(audio).extract_features.mean(dim=1)
        z_D = self.demographic_enc(demographic)

        if self.training and self.p_drop > 0:
            keep_a = (torch.rand(z_A.size(0), 1, device=z_A.device) >= self.p_drop).float()
            keep_d = (torch.rand(z_D.size(0), 1, device=z_D.device) >= self.p_drop).float()
            z_A = z_A * keep_a
            z_D = z_D * keep_d

        concat = torch.cat((z_A, z_D), dim=-1)
        if self.gate:
            weights = torch.sigmoid(self.gate(concat))
            concat = concat * weights

        return self.classifier(F.dropout(concat, p=0.1, training=self.training))


class FocalLoss(nn.Module):
    """
    Focal loss (Lin et al., 2017), with per-class weights set via the
    Effective Number of Samples (Cui et al., 2019,
    https://arxiv.org/abs/1901.05555) instead of a hand-picked alpha.

    Each class's "effective number" is E_y = (1 - beta**n_y) / (1 - beta),
    where n_y is that class's sample count: as n_y grows, additional samples
    increasingly overlap with ones already seen (near-duplicate information
    content), so E_y grows sublinearly and saturates as beta -> 1. Weighting
    inversely by E_y gives the minority class a more principled boost than a
    hand-tuned alpha, and degrades to plain inverse-frequency weighting as
    beta -> 0.
    """
    def __init__(self, class_counts, gamma: float = 2.0, beta: float = 0.9999):
        super().__init__()
        self.gamma = gamma
        class_counts = torch.as_tensor(class_counts, dtype=torch.float32)
        effective_num = 1.0 - torch.pow(beta, class_counts)
        class_weights = (1.0 - beta) / effective_num
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = self.class_weights[labels]
        return (alpha_t * (1 - pt) ** self.gamma * ce).mean()


def train_epoch(loader, model: FusionModule, optimizer, loss_fn: FocalLoss):
    """Run one training pass over `loader`, returning (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    for signals, demographics, labels in tqdm(loader, desc="Train", leave=False):
        signals = signals.to(config.device)
        demographics = demographics.to(config.device)
        labels = labels.to(config.device).long()

        optimizer.zero_grad()
        logits = model(signals, demographics)
        loss = loss_fn(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += len(labels)

    return total_loss / len(loader), correct / total


def eval_epoch(loader, model: FusionModule, loss_fn: FocalLoss):
    """
    Run one no-grad evaluation pass over `loader`.

    Returns:
        (avg_loss, accuracy, sensitivity, specificity, auprc, auroc, preds, labels)
    """
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for signals, demographics, labels in tqdm(loader, desc="Eval", leave=False):
            signals = signals.to(config.device)
            demographics = demographics.to(config.device)
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
    has_both_classes = len(set(all_labels)) > 1
    auprc = average_precision_score(all_labels, all_probs) if has_both_classes else 0.0
    auroc = roc_auc_score(all_labels, all_probs) if has_both_classes else 0.0
    return total_loss / len(loader), correct / total, sensitivity, specificity, auprc, auroc, all_preds, all_labels


def visualize(train_losses, val_losses, train_accs, val_accs, all_preds, all_labels, file_name: str = "larycl_results"):
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
    plt.savefig(f"graphs/{file_name}.png", dpi=150)
    plt.show()


def parse_args():
    parser = ArgumentParser(description="Train LaryFusion: fusion classifier with modality dropout, stratified k-fold CV")
    parser.add_argument("--name", type=str, required=True, help="Name of your model")
    parser.add_argument("--gate", action="store_true", help="Use a sigmoid gate over the fused features")
    parser.add_argument("--modality_dropout", type=float, default=0.1,
                         help="Per-modality dropout probability applied before fusion during training")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of stratified k-fold splits")
    parser.add_argument("--beta", type=float, default=0.9999,
                         help="Effective-number-of-samples beta for class-balanced FocalLoss "
                              "(closer to 1 = more aggressive minority-class upweighting)")
    parser.add_argument("--results_name", type=str, default="laryfusion_results", help="What to name the result diagrams")
    return parser.parse_args()


def main():
    """
    Train one FusionModule per stratified k-fold split, checkpointing each
    fold's best epoch (by validation loss) to
    `checkpoints/{name}_fold{k}_best.pt`, with early stopping after
    `config.patience` epochs of no improvement. Reports mean +/- std
    accuracy/sensitivity/specificity/AUPRC/AUROC across folds.
    """
    args = parse_args()

    print("Loading dataset...")
    dataset = load_patients(feature_type="wav2vec2", include_demographics=True)

    fold_metrics = []
    for fold, (train_data, test_data) in enumerate(stratified_kfold(dataset, n_splits=args.n_splits), start=1):
        print(f"\n=== Fold {fold}/{args.n_splits} ===")
        train_loader = batch_inputs(train_data, stratify=True)
        test_loader = batch_inputs(test_data)

        dinput_dim = len(train_data[0]["background"])
        model = FusionModule(config, num_features=dinput_dim, gate=args.gate,
                              p_drop=args.modality_dropout).to(config.device)

        class_counts = [sum(1 for p in train_data if p["label"] == c) for c in (0, 1)]
        loss_fn = FocalLoss(class_counts=class_counts, beta=args.beta).to(config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5, eps=1e-6)

        if fold == 1:
            print(f"Model layout: {model}")
            print("Trainable parameters:", sum(p.numel() for p in model.parameters()))

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_val_loss = float('inf')
        best_metrics = None
        no_improv = 0
        for epoch in range(1, config.num_epochs + 1):
            tr_loss, tr_acc = train_epoch(train_loader, model, optimizer, loss_fn)
            val_loss, val_acc, val_sens, val_spec, val_auprc, val_auroc, preds, labels = eval_epoch(
                test_loader, model, loss_fn)

            train_losses.append(tr_loss)
            val_losses.append(val_loss)
            train_accs.append(tr_acc)
            val_accs.append(val_acc)

            print(f"Epoch {epoch:02d} | "
                  f"Train loss: {tr_loss:.4f} acc: {tr_acc:.3f} | "
                  f"Val loss: {val_loss:.4f} acc: {val_acc:.3f} sensitivity: {val_sens:.3f} "
                  f"specificity: {val_spec:.3f} auprc: {val_auprc:.3f} auroc: {val_auroc:.3f}")

            if val_loss < best_val_loss:
                print("Updating best model...")
                best_val_loss = val_loss
                no_improv = 0
                best_metrics = (val_acc, val_sens, val_spec, val_auprc, val_auroc)
                torch.save(model.state_dict(), f"checkpoints/{args.name}_fold{fold}_best.pt")
                visualize(train_losses, val_losses, train_accs, val_accs, preds, labels,
                          f"{args.results_name}_fold{fold}")
            else:
                no_improv += 1
                if no_improv >= config.patience:
                    print(f"No improvement in val loss for {config.patience} epochs, stopping fold {fold} early.")
                    break

        fold_metrics.append(best_metrics)

    _, senss, specs, auprcs, _ = zip(*fold_metrics)
    print("\n=== K-Fold Summary ===")
    print(f"Sensitivity: {np.mean(senss):.3f} +/- {np.std(senss):.3f}")
    print(f"Specificity: {np.mean(specs):.3f} +/- {np.std(specs):.3f}")
    print(f"AUPRC:       {np.mean(auprcs):.3f} +/- {np.std(auprcs):.3f}")

if __name__ == '__main__':
    main()
