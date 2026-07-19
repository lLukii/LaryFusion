import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

class AFT(nn.Module):
    '''
    (Deprecated)
    '''
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

class FusionModule(nn.Module):
    """
    Multimodal Fusion Model for Audio and Demographic Data
    """
    def __init__(self, config, num_features, hidden_dim=512):
        super().__init__()
        self.demographic_enc = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.aud_encoder = Wav2Vec2Model.from_pretrained(config.w2v_name)
        self.classifier = nn.Linear(2 * hidden_dim, 2)

    def forward(self, audio, demographic):
        z_A = self.aud_encoder(audio)
        encoded = torch.mean(encoded, dim=1)
        z_D = self.demographic_enc(demographic)
        concat = torch.cat(z_A, z_D)
        outputs = self.classifier(F.dropout(concat, p=0.1))
        return outputs

class GatedNetwork(nn.Module):
    """
    Multimodal Fusion Model for Audio and Demographic Data
    """
    def __init__(self, config, num_features, mlp_dim=128, audio_dim=512, gated_dim=128):
        super().__init__()
        self.demographic_cls = nn.Sequential(
            nn.Linear(num_features, mlp_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(mlp_dim, 2)
        )
        self.aud_encoder = Wav2Vec2Model.from_pretrained(config.w2v_name)
        self.classifier = nn.Linear(audio_dim, 2)
        self.gated_network = nn.Sequential(
            nn.Linear(4, gated_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(gated_dim, 2)
        )

    def forward(self, audio, demographic):
        z_A = self.aud_encoder(audio).extract_features
        z_A = torch.mean(z_A, dim=1)
        logits_A = self.classifier(z_A)
        logits_D = self.demographic_cls(demographic)
        combined = torch.cat((logits_A, logits_D), dim=-1) # [B, 4]
        weights = F.softmax(self.gated_network(combined), dim=-1) # [B, 2]
        return logits_A * weights[:, 0:1] + logits_D * weights[:, 1:2]
        
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