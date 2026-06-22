import torch
from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    sampling_rate: int = 16000
    padding: int = 48000
    num_epochs: int = 10
    batch_size: int = 32
    dataset_path: str = "/Users/llukii/Desktop/School/Programming/Multimodal throat cancer/dataset"
    model_path: str = ""
    w2v_name: str = "facebook/wav2vec2-base-960h"
    pretrained: bool = True
    lr: float = 1e-4
    