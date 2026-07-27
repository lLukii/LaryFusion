import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    f1_score
)
from tqdm import tqdm
from argparse import ArgumentParser
from config import Config
from dataset import load_wavs, batch_inputs, cross_validation
from models import (
    FusionModule,
    Wav2VecBase,
    AugmentedFusion,
    ResNet
)

import warnings
warnings.filterwarnings("ignore")

config = Config()
torch.manual_seed(1337)

def train_epoch(args, loader, model, optimizer, loss_fn):
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


def eval_epoch(args, loader, model, loss_fn):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for signals, demographics, labels in tqdm(loader, desc="Eval", leave=False):
            signals = signals.to(config.device)
            demographics = demographics.to(config.device)
            labels = labels.to(config.device).long()

            logits = model(signals, demographics)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), correct / total, f1, all_preds, all_labels

def train_augmented_epoch(args, loader, model, task_optimizer, aug_optimizer, task_fn, consist_fn,
                           alpha=0.5, w1=1e-4, w2=0.1, w3=0.1):
    """
    LeMDA training step (Liu et al., 2022)
    """
    model.train()
    total_loss, correct, total = 0, 0, 0
    for signals, demographics, labels in tqdm(loader, desc="Train (LeMDA)", leave=False):
        signals = signals.to(config.device)
        demographics = demographics.to(config.device)
        labels = labels.to(config.device).long()

        # Task Network update
        model.freeze_grad(generate=False)
        task_optimizer.zero_grad()
        output, r_output, _, _ = model(signals, demographics, reconstruct=True)
        loss_task = task_fn(output, labels) + task_fn(r_output, labels)
        loss_task.backward()
        task_optimizer.step()

        # Augmentation Network update
        model.freeze_grad(generate=True)
        aug_optimizer.zero_grad()
        output_g, r_output_g, mu, std = model(signals, demographics, reconstruct=True)

        with torch.no_grad():
            confident = F.softmax(output_g, dim=-1).amax(dim=-1) > alpha

        adv_loss = task_fn(r_output_g, labels)
        if confident.any():
            log_p_aug = F.log_softmax(r_output_g[confident], dim=-1)
            p_orig = F.softmax(output_g[confident], dim=-1).detach()
            consist_loss = consist_fn(log_p_aug, p_orig)
        else:
            consist_loss = torch.zeros((), device=config.device)
        kl_loss = model.generator.kl_loss(mu, std)

        loss_aug = -w1 * adv_loss + w2 * consist_loss + w3 * kl_loss
        loss_aug.backward()
        aug_optimizer.step()

        model.freeze_grad(generate=False)  # leave task net trainable/unfrozen by default

        total_loss += loss_task.item()
        correct += (output.argmax(dim=-1) == labels).sum().item()
        total += len(labels)

    return total_loss / len(loader), correct / total

def visualize(train_losses, val_losses, train_accs, val_accs, all_preds, all_labels, file_name="results"):
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
    parser.add_argument("--use_lora", type=bool, default=False, help="Whether to use Low Rank Adaption (LoRA) fine-tuning")
    parser.add_argument("--model_type", type=int, default=0, help="Which model to use for training")
    parser.add_argument("--loss", type=str, default="cross_entropy", help="Loss function to use for training")
    parser.add_argument("--results_name", type=str, default="results", help="What to name the result diagram")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dataset = load_wavs(include_demographics=True, as_spectrogram=args.model_type==3)
    train_data, test_data = cross_validation(dataset)
    train_loader = batch_inputs(train_data, oversample=True, shuffle=True)
    test_loader = batch_inputs(test_data)

    dinput_dim = len(train_data[0]["background"])
    model = None
    if args.model_type == 0:
        model = FusionModule(config, num_features=dinput_dim).to(config.device)
    elif args.model_type == 1:
        model = FusionModule(config, num_features=dinput_dim, gate=True).to(config.device)
    elif args.model_type == 2: 
        model = Wav2VecBase(config).to(config.device)
    elif args.model_type == 3: 
        model = ResNet().to(config.device)
    elif args.model_type == 4:
        model = AugmentedFusion(config, num_features=dinput_dim).to(config.device)

    loss_fn, gen_fn = nn.CrossEntropyLoss(), nn.KLDivLoss(reduction="batchmean")

    aug_optimizer = None
    if args.model_type == 4:
        task_params = [p for n, p in model.named_parameters() if not n.startswith("generator.")]
        optimizer = torch.optim.Adam(task_params, lr=config.lr)
        aug_optimizer = torch.optim.Adam(model.generator.parameters(), lr=config.aug_lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"Model layout: {model}")
    print("Trainable parameters:", sum(p.numel() for p in model.parameters()))

    best_f1 = 0
    no_improv = 0
    for epoch in range(1, config.num_epochs + 1):
        tr_loss, tr_acc = train_epoch(args, train_loader, model, optimizer, loss_fn)
        val_loss, val_acc, val_f1, preds, labels = eval_epoch(args, test_loader, model, loss_fn)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:02d} | "
            f"Train loss: {tr_loss:.4f} acc: {tr_acc:.3f} | "
            f"Val loss: {val_loss:.4f} acc: {val_acc:.3f} f1: {val_f1:.3f}")

        if val_f1 > best_f1:
            print("Updating best model...")
            best_f1 = val_f1
            no_improv = 0
            torch.save(model.state_dict(), f"checkpoints/{args.name}_best.pt")
            visualize(train_losses, val_losses, train_accs, val_accs, preds, labels, args.results_name)

        else:
            no_improv += 1
            if no_improv >= config.patience:
                print(f"No improvement in val f1 for {config.patience} epochs, stopping early.")
                break