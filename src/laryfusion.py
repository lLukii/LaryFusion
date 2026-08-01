from argparse import ArgumentParser
import math
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from config import Config
from transformers import Wav2Vec2Model
from dataset import load_wavs, batch_inputs, cross_validation

config = Config()

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class UNet1D(nn.Module):
    """
    Time-conditioned residual MLP noise predictor eps_theta(x_t, t) for flat
    (B, dim) embeddings. Named UNet1D to keep the "denoiser" role explicit, but
    a single 1024-d vector has no spatial structure for convs to exploit, so
    this is a small conditioned MLP with residual blocks instead.
    """
    def __init__(self, dim=1024, hidden_dim=512, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_proj = nn.Linear(dim, hidden_dim)
        self.block1 = nn.Sequential(nn.Linear(hidden_dim + time_dim, hidden_dim), nn.SiLU())
        self.block2 = nn.Sequential(nn.Linear(hidden_dim + time_dim, hidden_dim), nn.SiLU())
        self.out_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = self.in_proj(x)
        h = self.block1(torch.cat([h, t_emb], dim=-1)) + h
        h = self.block2(torch.cat([h, t_emb], dim=-1)) + h
        return self.out_proj(h)


class DDPM(nn.Module):
    """
    Minimal DDPM (Ho et al., 2020) over flat feature vectors, standing in for
    the VAE as LeMDA's augmentation network G. `augment()` noises a real
    embedding to a random step t and recovers a one-step estimate of x0 from
    the predicted noise
    """
    def __init__(self, input_dim=1024, hidden_dim=512, time_dim=128,
                 timesteps=200, beta_start=1e-4, beta_end=2e-2):
        super().__init__()
        self.timesteps = timesteps
        self.denoiser = UNet1D(dim=input_dim, hidden_dim=hidden_dim, time_dim=time_dim)

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.alpha_bars[t].sqrt().unsqueeze(-1)
        sqrt_1m_ab = (1 - self.alpha_bars[t]).sqrt().unsqueeze(-1)
        return sqrt_ab * x0 + sqrt_1m_ab * noise, noise

    def augment(self, x0, t=None):
        """One-step differentiable noise -> predict -> reconstruct x0_hat."""
        b = x0.size(0)
        if t is None:
            t = torch.randint(0, self.timesteps, (b,), device=x0.device)
        x_t, noise = self.q_sample(x0, t)
        pred_noise = self.denoiser(x_t, t)

        sqrt_ab = self.alpha_bars[t].sqrt().unsqueeze(-1)
        sqrt_1m_ab = (1 - self.alpha_bars[t]).sqrt().unsqueeze(-1)
        x0_hat = (x_t - sqrt_1m_ab * pred_noise) / sqrt_ab
        return x0_hat, pred_noise, noise

    @torch.no_grad()
    def generate(self, x0, start_frac=0.3):
        """Optional full reverse-diffusion sampling (SDEdit-style partial noising)."""
        b = x0.size(0)
        start_t = max(1, int(self.timesteps * start_frac))
        t0 = torch.full((b,), start_t - 1, device=x0.device, dtype=torch.long)
        x_t, _ = self.q_sample(x0, t0)

        for step in reversed(range(start_t)):
            t = torch.full((b,), step, device=x0.device, dtype=torch.long)
            pred_noise = self.denoiser(x_t, t)
            alpha_t = self.alphas[t].unsqueeze(-1)
            alpha_bar_t = self.alpha_bars[t].unsqueeze(-1)
            beta_t = self.betas[t].unsqueeze(-1)

            mean = (x_t - beta_t / (1 - alpha_bar_t).sqrt() * pred_noise) / alpha_t.sqrt()
            noise = torch.randn_like(x_t) if step > 0 else torch.zeros_like(x_t)
            x_t = mean + beta_t.sqrt() * noise
        return x_t


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
        self.generator = DDPM(input_dim=2 * hidden_dim)

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
            recon, pred_noise, noise = self.generator.augment(concat)
            r_output = self.classifier(F.dropout(recon, p=0.1))
            return output, r_output, pred_noise, noise

        return output

    def freeze_grad(self, generate=False):
        """
        generate=False -> train the task network (encoders + classifier), freeze the VAE.
        generate=True  -> train the VAE generator, freeze the task network.
        """
        for module in self.children():
            requires_grad = generate if isinstance(module, DDPM) else not generate
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
        output_g, r_output_g, pred_noise, noise = model(signals, demographics, reconstruct=True)

        with torch.no_grad(): # only
            confident = F.softmax(output_g, dim=-1).amax(dim=-1) > alpha

        adv_loss = task_fn(r_output_g, labels)
        if confident.any():
            log_p_aug = F.log_softmax(r_output_g[confident], dim=-1)
            p_orig = F.softmax(output_g[confident], dim=-1).detach()
            consist_loss = consist_fn(log_p_aug, p_orig)
        else:
            consist_loss = torch.zeros((), device=config.device)
        diffusion_loss = F.mse_loss(pred_noise, noise)

        loss_aug = -w1 * adv_loss + w2 * consist_loss + w3 * diffusion_loss
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