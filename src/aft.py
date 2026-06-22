import torch
import torch.nn as nn
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
    
    