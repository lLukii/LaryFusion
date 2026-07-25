import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

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
        self.gate_a = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim), 
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        ) if gate else None
        self.gate_b = nn.Sequential(
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
        if self.gate_a: 
            A = self.gate_a(concat)
            B = F.sigmoid(self.gate_b(concat)) # [B, 2 * dim]
            concat = A * B # [B, 2 * dim]

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

class AugmentedFusion(FusionModule): 
    def __init__(self, config):
        super().__init__()
        self.generator = VAE()

    def forward(self, audio, demographic): 
        z_A = self.aud_encoder(audio).extract_features
        z_A = torch.mean(z_A, dim=1)
        z_D = self.demographic_enc(demographic)

        concat = torch.cat((z_A, z_D), dim=-1)
        recons = self.generator(concat)
        output = self.classifier(F.dropout(concat, p=0.1))
        r_output = self.classifier(F.dropout(recons, p=0.1))
        return output, r_output

    def freeze_grad(self, generate=False): 
        for module in self.children(): 
            if isinstance(module, VAE): 
                for p in self.parameters(): 
                    p.requires_grad = generate
            else: 
                for p in self.parameters(): 
                    p.requires_grad = not generate