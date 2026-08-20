"""
Baseline models and training script for throat cancer classification, without
LeMDA-style feature-space augmentation (see laryfusion.py for that).

Run directly to train one of four baselines, selected with `--model_type`:
    0 - FusionModule (wav2vec2 + demographic FFN, concatenated)
    1 - FusionModule with a sigmoid gate over the fused features
    2 - Wav2VecBase (audio-only wav2vec2 classifier)

Example: `python baselines.py --name fusion_gated --model_type 1`
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    average_precision_score
)
from tqdm import tqdm
from argparse import ArgumentParser
from config import Config
from dataset import load_patients, cross_validation, batch_inputs

import warnings
warnings.filterwarnings("ignore")

config = Config()

class FusionModule(nn.Module):
    """
    Multimodal Fusion Model for Audio and Demographic Data
    """
    def __init__(self, config, num_features, audio_dim=512, demo_dim=64, gate=False):
        super().__init__()
        combined_dim = audio_dim + demo_dim
        self.demographic_enc = nn.Sequential(
            nn.Linear(num_features, demo_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(demo_dim, demo_dim)
        )
        self.aud_encoder = Wav2Vec2Model.from_pretrained(config.w2v_name)
        self.classifier = nn.Linear(combined_dim, 2)
        self.gate = nn.Sequential(
            nn.Linear(combined_dim, combined_dim), 
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(combined_dim, combined_dim)
        ) if gate else None

    def forward(self, audio, demographic):
        z_A = self.aud_encoder(audio).extract_features
        z_A = torch.mean(z_A, dim=1)
        z_D = self.demographic_enc(demographic)
        concat = torch.cat((z_A, z_D), dim=-1)
        if self.gate: 
            B = F.sigmoid(self.gate(concat)) # [B, 2 * dim]
            concat = concat * B # [B, 2 * dim]

        outputs = self.classifier(F.dropout(concat, p=0.1))
        return outputs

class Wav2VecBase(nn.Module):
    """
    Audio-only classifier: wav2vec2 features mean-pooled over time, then a
    single linear classifier head. No demographic input.
    """
    def __init__(self, config, output_dim=512):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(config.w2v_name)
        self.classifier = nn.Linear(output_dim, 2)
    
    def forward(self, signals, demographics=None): 
        encoded = self.encoder(signals).extract_features
        encoded = torch.mean(encoded, dim=1)
        encoded = self.classifier(F.dropout(encoded, p=0.1))
        return encoded


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
        demographics = demographics.to(config.device)
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
    auprc = average_precision_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return total_loss / len(loader), correct / total, sensitivity, specificity, auprc, all_preds, all_labels


def visualize(train_losses, val_losses, train_accs, val_accs, all_preds, all_labels, file_name="results"):
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
    parser = ArgumentParser(description="Train multimodal throat cancer classifier")
    parser.add_argument("--name", type=str, required=True, help="Name of your model")
    parser.add_argument("--model_type", type=int, default=0, help="Which model to use for training")
    parser.add_argument("--loss", type=str, default="cross_entropy", help="Loss function to use for training")
    parser.add_argument("--results_name", type=str, default="results", help="What to name the result diagram")
    return parser.parse_args()

def main():
    """
    Train one of the four baseline models (picked via `--model_type`, see
    module docstring) and checkpoint the best epoch by validation AUPRC to
    `checkpoints/{name}_best.pt`, with early stopping after `config.patience`
    epochs of no improvement.
    """
    args = parse_args()
    dataset = load_patients(feature_type="wav2vec2", include_demographics=True)
    train_data, test_data = cross_validation(dataset)
    train_loader = batch_inputs(train_data, stratify=True)
    test_loader = batch_inputs(test_data)

    dinput_dim = len(train_data[0]["background"])
    model = None
    if args.model_type == 0:
        model = FusionModule(config, num_features=dinput_dim).to(config.device)
    elif args.model_type == 1:
        model = FusionModule(config, num_features=dinput_dim, gate=True).to(config.device)
    elif args.model_type == 2: 
        model = Wav2VecBase(config).to(config.device)
    
    loss_fn = FocalLoss().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"Model layout: {model}")
    print("Trainable parameters:", sum(p.numel() for p in model.parameters()))

    best_test_loss = float('inf')
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

        if val_loss < best_test_loss:
            print("Updating best model...")
            best_test_loss = val_loss
            no_improv = 0
            torch.save(model.state_dict(), f"checkpoints/{args.name}_best.pt")
            visualize(train_losses, val_losses, train_accs, val_accs, preds, labels, args.results_name)

        else:
            no_improv += 1
            if no_improv >= config.patience:
                print(f"No improvement in val loss for {config.patience} epochs, stopping early.")
                break

if __name__ == '__main__':
    main()