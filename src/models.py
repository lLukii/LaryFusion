import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

class AFT(nn.Module):
    def __init__(self, config, hidden_size=512, embedding_size=256): 
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(config.w2v_name)
        self.label_classifier = nn.Sequential(
            nn.Linear(hidden_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, 2)
        )
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, 2)
        )
    
    def forward(self, audio_signals):
        encoded = self.encoder(audio_signals).last_hidden_state
        encoded = torch.mean(encoded, dim=1)
        label_probs = self.label_classifier(encoded)
        disc_probs = self.discriminator(encoded)
        return label_probs, disc_probs

class MultimodalFusion(nn.Module):
    """
    Cross-attention Multimodal Fusion Model for Audio and Demographic Data
    """
    def __init__(self, config, audio_dim=512, num_features=64, demo_dim=256, dH=128):
        super().__init__()
        self.demographic_enc = nn.Sequential(
            nn.Linear(num_features, demo_dim),
            nn.ReLU(),
            nn.Linear(demo_dim, demo_dim)
            nn.ReLU(),
            nn.Linear(demo_dim, demo_dim)
        )
        self.aud_encoder = Wav2Vec2Model.from_pretrained(config.w2v_name)
        self.WQ = nn.Linear(audio_dim, dH, bias=False)
        self.WK = nn.Linear(demo_dim, dH, bias=False)
        self.WV = nn.Linear(demo_dim, dH, bias=False)
        self.classifier = nn.Linear(dH, 2)

        self.attn = nn.MultiheadAttention(embed_dim=dH, num_heads=1, batch_first=True)
    
    def forward(self, audio_signals, demographics):
        z_a = self.aud_encoder(audio_signals).extract_features # [B, T, dAudio]
        z_d = self.demographic_enc(demographics).unsqueeze(1) # [B, 1, dDemo]
        Q = self.WQ(z_a) # [B, T, dH]
        K = self.WK(z_d) # [B, T, dH]
        V = self.WV(z_d) # [B, T, dH]

        attn_output, _ = self.attn(Q, K, V) # [B, T, dH]
        pooled = torch.mean(attn_output, dim=1)
        logits = self.classifier(pooled)
        return logits
        
class Wav2VecBase(nn.Module):
    """
    Base classifier model using Wav2Vec2 for audio feature extraction.
    Conv + Linear layer 
    """
    def __init__(self, config, output_dim=512):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(config.w2v_name)
        self.classifier = nn.Linear(output_dim, 2)
    
    def forward(self, signals, masks, demographics=None): 
        encoded = self.encoder(signals, attention_mask=masks).extract_features
        encoded = torch.mean(encoded, dim=1)
        encoded = self.classifier(F.dropout(encoded, p=0.1))
        return encoded

class FocalLoss(nn.Module):
    """
    Focal loss to handle class imbalance.
    alpha: weight for the class
    gamma: focusing parameter to reduce the loss contribution from easy examples
    logits: whether the inputs are logits or probabilities
    """
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss