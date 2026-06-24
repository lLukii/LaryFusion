import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

class AFT(nn.Module):
    def __init__(self, config, hidden_size=512, embedding_size=256): 
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained(
            config.w2v_name
        )
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
    def __init__(self, config, dinput_dim, dhidden_dim=128, doutput_dim=64,
                 aoutput_dim=768, dH=128):
        super().__init__()
        self.demographic_enc = nn.Sequential(
            nn.Linear(dinput_dim, dhidden_dim),
            nn.ReLU(),
            nn.Linear(dhidden_dim, dhidden_dim),
            nn.ReLU(),
            nn.Linear(dhidden_dim, doutput_dim)
        )
        self.aud_encoder = Wav2Vec2Model.from_pretrained(config.w2v_name)
        self.WQ = nn.Linear(aoutput_dim, dH, bias=False)
        self.WK = nn.Linear(doutput_dim, dH, bias=False)
        self.WV = nn.Linear(doutput_dim, dH, bias=False)
        self.classifier = nn.Linear(dH, 2)

    def forward(self, audio_data, attention_mask, demographic_data):
        a_encoded = self.aud_encoder(audio_data, attention_mask=attention_mask).last_hidden_state

        a_encoded = torch.mean(a_encoded, dim=1)          # [B, 768]
        d_encoded = self.demographic_enc(demographic_data) # [B, doutput_dim]
        Q = self.WQ(a_encoded)  # [B, dH]
        K = self.WK(d_encoded)  # [B, dH]
        V = self.WV(d_encoded)  # [B, dH]
        
        scale = torch.sqrt(torch.tensor(K.shape[-1], dtype=torch.float32, device=K.device))
        attn = F.softmax(Q @ K / scale, dim=-1)  # per-feature attention [B, dH]
        return self.classifier(attn @ V)          # [B, 2]
        
