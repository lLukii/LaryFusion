"""
LaryFusion: the gated wav2vec2 + demographic fusion model, trained with a
LeMDA-style (Liu et al., 2022) feature-space augmentation network.

A VAE (`VAE`) learns to generate adversarial-but-label-preserving latent
vectors for the fused audio + demographic embedding. Each training step
(`train_augmented_epoch`) alternates between: (1) updating the task network
(encoders + classifier) on real and VAE-reconstructed features, and (2)
updating the VAE to maximize task loss on its reconstructions while a
consistency regularizer keeps its output distribution close to the original,
label-preserving prediction (see `AugmentedFusion.freeze_grad` for how the two
networks are alternately frozen). Run this file directly to train the model
end to end.

Example: `python laryfusion.py --name laryfusion_v1`
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

from librosa import load
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import Config
from dataset import load_wavs, batch_inputs, cross_validation
from argparse import ArgumentParser

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

config = Config()
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.w2v_name)

def load_wavs(include_demographics=False): 
    patients = {}
    for voice_type in ["benign", "malignant", "normal"]:
        for sample in os.listdir(os.path.join(config.dataset_path, voice_type)):
            if sample.split(".")[-1] != "wav":
                continue
            user_id = sample.split(".")[0]
            signal, _ = load(os.path.join(config.dataset_path, voice_type, sample), sr=config.sampling_rate)
            if len(signal) < config.padding: 
                signal = np.pad(signal, (0, config.padding - len(signal)), mode="constant", constant_values=0)
            else: 
                signal = signal[:config.padding]
            
            processed = feature_extractor(signal, sampling_rate=config.sampling_rate,  
                return_tensors="pt")
            signal = processed.input_values
            patients[user_id] = {
                    "signal" : signal,
                    "label" : 1 if voice_type in ["malignant", "synthetic"] else 0,
                    "background" : None
            }
    
    if include_demographics:
        benign_history = pd.read_excel(os.path.join(config.dataset_path, "benign", "medicalhistory.xlsx")).fillna(0)
        malignant_history = pd.read_excel(os.path.join(config.dataset_path, "malignant", "medicalhistory.xlsx")).fillna(0)
        normal_history = pd.read_excel(os.path.join(config.dataset_path, "normal", "medicalhistory.xlsx")).fillna(0)

        for medical_history in [benign_history, malignant_history, normal_history]:
            for _, row in medical_history.iterrows():
                user_id = row["ID"]
                patients[user_id]["background"] = row.drop(["ID", "Disease category"])

    return patients

def cross_validation(dataset):
    patient_list = list(dataset.values())
    train_data, test_data = train_test_split(patient_list, test_size=0.2, random_state=config.seed)

    # normalize demographic features based on statistics in each split
    if train_data[0]["background"] is not None:
        bg_cols = train_data[0]["background"].index
        quant_cols = [col for col in bg_cols
                      if pd.Series([p["background"][col] for p in train_data]).nunique() > 2] 
        # only normalize non-binary rows

        train_bg = pd.DataFrame([p["background"] for p in train_data])
        scaler = StandardScaler()
        scaler.fit(train_bg[quant_cols])

        for split in [train_data, test_data]:
            for p in split:
                bg = p["background"].copy()
                bg[quant_cols] = scaler.transform(bg[quant_cols].values.reshape(1, -1).astype(float))[0]
                p["background"] = bg

    return train_data, test_data

class ThroatCancerDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        p = self.data[idx]
        demo = (torch.tensor(p["background"].values.astype(float), dtype=torch.float32)
                if p["background"] is not None else None)
        return p["signal"].squeeze(0), demo, self.labels[idx]


def collate_fn(batch):
    signals, demos, labels = zip(*batch)
    demographics = torch.stack(demos) if demos[0] is not None else None
    return torch.stack(signals), demographics, torch.stack(list(labels))

def batch_inputs(data, shuffle=False):
    labels = torch.tensor([patient["label"] for patient in data])
    return DataLoader(ThroatCancerDataset(data, labels),
                      batch_size=config.batch_size, shuffle=shuffle
                      collate_fn=collate_fn)

class VAE(nn.Module):
    '''
    Variational autoencoder implementation for generating latent samples
    '''
    def __init__(self, input_dim=1024, hidden_dim=512, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_std = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        # Softplus + epsilon for stable std deviation
        std = F.softplus(self.fc_std(h)) + 1e-6
        return mu, std

    def reparameterize(self, mu, std):
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, kl_weight=1.0):
        mu, std = self.encode(x)
        z = self.reparameterize(mu, std)
        x_recon = self.decode(z)
        return x_recon

    def kl_loss(self, mu, std):
        """KL divergence of N(mu, std^2) against the standard normal prior."""
        var = std.pow(2)
        kl = -0.5 * torch.sum(1 + torch.log(var + 1e-8) - mu.pow(2) - var, dim=-1)
        return kl.mean()


class AugmentedFusion(nn.Module):
    """
    LeMDA-style feature-space augmentation: a VAE learns to generate adversarial
    but label-preserving latent vectors for the fused audio + demographic features.
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
        ) 
        self.generator = VAE(input_dim=2 * hidden_dim)

    def forward(self, audio, demographic, reconstruct=False):
        z_A = self.aud_encoder(audio).extract_features
        z_A = torch.mean(z_A, dim=1)
        z_D = self.demographic_enc(demographic)
        concat = torch.cat((z_A, z_D), dim=-1)
        if self.gate:
            B = F.sigmoid(self.gate(concat))
            concat = concat * B

        output = self.classifier(F.dropout(concat, p=0.1))
        # Reconstruction only necessary for training. Just pass in data during inference.
        if reconstruct:
            mu, std = self.generator.encode(concat)
            z_aug = self.generator.reparameterize(mu, std)
            recon = self.generator.decode(z_aug)
            r_output = self.classifier(F.dropout(recon, p=0.1))
            return output, r_output, mu, std

        return output

    def freeze_grad(self, generate=False):
        """
        generate=False -> train the task network (encoders + classifier), freeze the VAE.
        generate=True  -> train the VAE generator, freeze the task network.
        """
        for module in self.children():
            requires_grad = generate if isinstance(module, VAE) else not generate
            for p in module.parameters():
                p.requires_grad = requires_grad

class FocalLoss(nn.Module): 
    def __init__(self, gamma=2.0, alpha=0.969): 
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce)
        alpha_t = torch.where(labels == 1, self.alpha, 1 - self.alpha)
        return (alpha_t * (1 - pt) ** self.gamma * ce).mean()

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

        # confidence masking, as mentioned in the paper to avoid the model learning from low-confidence predictions
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

def parse_args():
    parser = ArgumentParser(description="Train multimodal throat cancer classifier")
    parser.add_argument("--name", type=str, required=True, help="Name of your model")
    parser.add_argument("--use_lora", type=bool, default=False, help="Whether to use Low Rank Adaption (LoRA) fine-tuning")
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
    model = AugmentedFusion(config, num_features=dinput_dim, gate=True).to(config.device)
    
    loss_fn = FocalLoss().to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"Model layout: {model}")
    print("Trainable parameters:", sum(p.numel() for p in model.parameters()))

    best_auroc = 0
    no_improv = 0
    for epoch in range(1, config.num_epochs + 1):
        tr_loss, tr_acc = train_augmented_epoch(train_loader, model, optimizer, loss_fn)
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