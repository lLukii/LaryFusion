import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Model
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)
from tqdm import tqdm
from argparse import ArgumentParser
from config import Config
from dataset import load_wavs, batch_inputs, cross_validation

import warnings
warnings.filterwarnings("ignore")

config = Config()
torch.manual_seed(1337)

class FusionModule(nn.Module):
    """
    Multimodal Fusion Model for Audio and Demographic Data
    """
    def __init__(self, config, num_features, hidden_dim=512, gate=False):
        super().__init__()
        self.demographic_enc = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.aud_encoder = Wav2Vec2Model.from_pretrained(config.w2v_name)
        self.classifier = nn.Linear(2 * hidden_dim, 2)
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim), 
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim)
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
    Base classifier model using Wav2Vec2 for audio feature extraction.
    Conv + Linear layer 
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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                            stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())
        self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 
                            kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    """
    ResNet-18 implementation for audio encoding.
    Expects spectrogram input with dimension (B, 80, 188) 
    """
    def __init__(self, block=ResidualBlock, layers=[2, 2, 2, 2], hidden_dim=512):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, demographics=None):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 80, 188) -> (B, 1, 80, 188)

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.classifier(F.dropout(x, p=0.1))

        return x

class FocalLoss(nn.Module): 
    def __init__(self, gamma=2.0, alpha=0.969): 
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = torch.where(labels == 1, self.alpha, 1 - self.alpha)
        return (alpha_t * (1 - pt) ** self.gamma * ce).mean()

def train_epoch(loader, model, optimizer, loss_fn):
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
    auroc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return total_loss / len(loader), correct / total, sensitivity, specificity, auroc, all_preds, all_labels


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

def main(): 
    args = parse_args()
    dataset = load_wavs(include_demographics=True, as_spectrogram=args.model_type==3)
    train_data, test_data = cross_validation(dataset)
    train_loader = batch_inputs(train_data, shuffle=True)
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
    
    loss_fn = FocalLoss().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"Model layout: {model}")
    print("Trainable parameters:", sum(p.numel() for p in model.parameters()))

    best_auroc = 0
    no_improv = 0
    for epoch in range(1, config.num_epochs + 1):
        tr_loss, tr_acc = train_epoch(train_loader, model, optimizer, loss_fn)
        val_loss, val_acc, val_sens, val_spec, val_auroc, preds, labels = eval_epoch(test_loader, model, loss_fn)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:02d} | "
            f"Train loss: {tr_loss:.4f} acc: {tr_acc:.3f} | "
            f"Val loss: {val_loss:.4f} acc: {val_acc:.3f} sensitivity: {val_sens:.3f} specificity: {val_spec:.3f} auroc: {val_auroc:.3f}")

        if val_auroc > best_auroc:
            print("Updating best model...")
            best_auroc = val_auroc
            no_improv = 0
            torch.save(model.state_dict(), f"checkpoints/{args.name}_best.pt")
            visualize(train_losses, val_losses, train_accs, val_accs, preds, labels, args.results_name)

        else:
            no_improv += 1
            if no_improv >= config.patience:
                print(f"No improvement in val auroc for {config.patience} epochs, stopping early.")
                break

if __name__ == '__main__':
    main()