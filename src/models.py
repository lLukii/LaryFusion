"""All PyTorch model architectures used across the baseline/fusion scripts."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model


class AudioEncoder(nn.Module):
    """
    Wav2Vec2Model wrapper, mean-pooled transformer output. The transformer
    is fully frozen (even LoRA still overfit); the CNN feature extractor and
    its projection into the transformer stay trainable.
    """
    def __init__(self, w2v_name, latent_dim=256):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(w2v_name)
        self.output_dim = latent_dim
        self.proj = nn.Linear(self.encoder.config.hidden_size, latent_dim)
        for param in self.encoder.encoder.parameters():
            param.requires_grad = False

    def forward(self, audio):
        return self.proj(self.encoder(audio).last_hidden_state.mean(dim=1))


class Wav2VecBase(nn.Module):
    """Audio-only classifier: pooled wav2vec2 features -> linear head."""
    def __init__(self, config):
        super().__init__()
        self.aud_encoder = AudioEncoder(config.w2v_name)
        self.classifier = nn.Linear(self.aud_encoder.output_dim, 2)

    def forward(self, signals: torch.Tensor, demographics: torch.Tensor = None) -> torch.Tensor:
        z_A = self.aud_encoder(signals)
        return self.classifier(F.dropout(z_A, p=0.1, training=self.training))


class FusionModule(nn.Module):
    """
    Audio (wav2vec2) + demographic (MLP) fusion classifier. `gate` adds a
    sigmoid gate over the fused features; `p_drop` > 0 adds modality dropout
    (each modality independently zeroed with that probability during
    training).
    """
    def __init__(self, config, num_features: int, demo_dim: int = 64,
                 gate: bool = False, p_drop: float = 0.0):
        super().__init__()
        self.p_drop = p_drop
        self.aud_encoder = AudioEncoder(config.w2v_name)
        combined_dim = self.aud_encoder.output_dim + demo_dim
        self.demographic_enc = nn.Sequential(
            nn.Linear(num_features, demo_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(demo_dim, demo_dim),
        )
        self.classifier = nn.Linear(combined_dim, 2)
        self.gate = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(combined_dim, combined_dim),
        ) if gate else None

    def forward(self, audio: torch.Tensor, demographic: torch.Tensor) -> torch.Tensor:
        z_A = self.aud_encoder(audio)
        z_D = self.demographic_enc(demographic)

        if self.training and self.p_drop > 0:
            keep_a = (torch.rand(z_A.size(0), 1, device=z_A.device) >= self.p_drop).float()
            keep_d = (torch.rand(z_D.size(0), 1, device=z_D.device) >= self.p_drop).float()
            z_A = z_A * keep_a
            z_D = z_D * keep_d

        concat = torch.cat((z_A, z_D), dim=-1)
        if self.gate:
            concat = concat * torch.sigmoid(self.gate(concat))

        return self.classifier(F.dropout(concat, p=0.1, training=self.training))


class ConvBlock1D(nn.Module):
    """Conv1d -> BatchNorm -> ReLU, optionally max-pooled."""
    def __init__(self, in_channels: int, out_channels: int, pool_size: int = None, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pool_size) if pool_size else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        return self.pool(x) if self.pool else x


class CNN1D(nn.Module):
    """Six-block 1D-CNN over raw waveform (Kim et al., 2020). Audio-only."""
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock1D(1, 16),
            ConvBlock1D(16, 32, pool_size=8),
            ConvBlock1D(32, 64, pool_size=8),
            ConvBlock1D(64, 128, pool_size=8),
            ConvBlock1D(128, 256, pool_size=8),
            ConvBlock1D(256, 256),
        )
        self.adaptive_pool = nn.AdaptiveMaxPool1d(6)
        self.classifier = nn.Sequential(
            nn.Linear(6 * 256, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, num_classes),
        )

    def forward(self, signals: torch.Tensor, demographics: torch.Tensor = None) -> torch.Tensor:
        x = signals.unsqueeze(1)  # (B, T) -> (B, 1, T)
        x = self.blocks(x)
        x = self.adaptive_pool(x)
        x = x.flatten(1)
        return self.classifier(x)
