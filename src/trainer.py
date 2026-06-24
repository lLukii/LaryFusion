import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from tqdm import tqdm
from argparse import ArgumentParser
from config import Config
from dataset import load_wavs, batch_inputs, cross_validation
from models import MultimodalFusion
import warnings
warnings.filterwarnings("ignore")

config = Config()


def train_epoch(loader, model, optimizer, loss_fn):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for signals, masks, demographics, labels in tqdm(loader, desc="Train", leave=False):
        signals = signals.to(config.device)
        masks = masks.to(config.device)
        demographics = demographics.to(config.device)
        labels = labels.to(config.device).long()

        optimizer.zero_grad()
        logits = model(signals, masks, demographics)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += len(labels)

    return total_loss / len(loader), correct / total


def eval_epoch(loader, model, loss_fn):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for signals, masks, demographics, labels in tqdm(loader, desc="Eval", leave=False):
            signals = signals.to(config.device)
            masks = masks.to(config.device)
            demographics = demographics.to(config.device)
            labels = labels.to(config.device).long()

            logits = model(signals, masks, demographics)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), correct / total, f1, all_preds, all_labels


def visualize(train_losses, val_losses, train_accs, val_accs, all_preds, all_labels):
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
    plt.savefig("results.png", dpi=150)
    plt.show()


def parse_args():
    parser = ArgumentParser(description="Train multimodal throat cancer classifier")
    parser.add_argument("--use_lora", type=bool, default=False)
    parser.add_argument("--multimodal", type=bool, default=True)
    parser.add_argument("--model_path", type=str, default="models")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dataset = load_wavs(include_demographics=True)
    train_data, test_data = cross_validation(dataset)
    train_loader = batch_inputs(train_data, oversample=True)
    test_loader = batch_inputs(test_data)

    dinput_dim = len(train_data[0]["background"])
    model = MultimodalFusion(config, dinput_dim=dinput_dim).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, config.num_epochs + 1):
        tr_loss, tr_acc = train_epoch(train_loader, model, optimizer, loss_fn)
        val_loss, val_acc, val_f1, preds, labels = eval_epoch(test_loader, model, loss_fn)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:02d} | "
              f"Train loss: {tr_loss:.4f} acc: {tr_acc:.3f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.3f} f1: {val_f1:.3f}")

    visualize(train_losses, val_losses, train_accs, val_accs, preds, labels)
    torch.save(model.state_dict(), "multimodal_fusion.pt")
