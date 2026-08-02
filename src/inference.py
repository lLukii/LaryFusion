import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve
)
from argparse import ArgumentParser
import dataset as dataset_module
from dataset import load_wavs, batch_inputs
from config import Config
from laryfusion import MultimodalFusion


CLASS_NAMES = ["benign/normal", "malignant"]


def parse_args():
    parser = ArgumentParser(description="Inference tool for throat cancer classifier")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to saved model state dict (.pt file)")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Override dataset path from config")
    parser.add_argument("--output", type=str, default="inference_results.png",
                        help="Path to save the output figure")
    return parser.parse_args()


def run_inference(loader, model, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for signals, masks, demographics, labels in loader:
            signals = signals.to(device)
            masks = masks.to(device)
            demographics = demographics.to(device)
            labels = labels.to(device)

            logits = model(signals, masks, demographics)
            probs = F.softmax(logits, dim=-1)[:, 1]
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    return all_labels, all_preds, all_probs


def plot_results(labels, preds, probs, save_path):
    _, axes = plt.subplots(1, 3, figsize=(16, 4))

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(ax=axes[0], colorbar=False)
    axes[0].set_title("Confusion Matrix")

    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    axes[1].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend()

    report = classification_report(labels, preds, target_names=CLASS_NAMES)
    axes[2].axis("off")
    axes[2].text(0.05, 0.5, report, transform=axes[2].transAxes,
                 fontsize=9, verticalalignment="center", fontfamily="monospace")
    axes[2].set_title("Classification Report")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"\nAUC-ROC: {auc:.4f}")
    print(f"\n{report}")
    print(f"Saved figure to {save_path}")


if __name__ == "__main__":
    args = parse_args()
    config = Config()

    if args.dataset_path:
        config.dataset_path = args.dataset_path
        dataset_module.config.dataset_path = args.dataset_path

    dataset = load_wavs(include_demographics=True)
    data = list(dataset.values())
    loader = batch_inputs(data)

    dinput_dim = len(data[0]["background"])
    model = MultimodalFusion(config, dinput_dim=dinput_dim).to(config.device)
    model.load_state_dict(torch.load(args.model_path, map_location=config.device))

    labels, preds, probs = run_inference(loader, model, config.device)
    plot_results(labels, preds, probs, save_path=args.output)
