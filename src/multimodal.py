import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

class Multimodal(nn.Module): 
    def __init__(self, config, dinput_dim, dhidden_dim, doutput_dim, 
                 aoutput_dim, dH):
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

    def forward(self, audio_data, demographic_data):
        a_encoded = self.aud_encoder(audio_data).last_hidden_state
        a_encoded = torch.mean(a_encoded, dim=1)
        d_encoded = self.demographic_enc(demographic_data)
        matQ = self.WQ(a_encoded)
        matK = self.WK(d_encoded)
        matV = self.WV(d_encoded)

        attn_scores = F.softmax(matQ @ matK.T / torch.sqrt(matK.shape[1]))
        attn_scores = attn_scores @ matV
        return attn_scores # [B, dH]
        
