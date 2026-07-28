import torch
from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    sampling_rate: int = 16000
    padding: int = 48000 # 3 seconds
    num_epochs: int = 15
    patience: int = 5
    batch_size: int = 64
    dataset_path: str = "/Users/llukii/Desktop/School/Programming/Multimodal throat cancer/dataset"
    model_path: str = ""
    w2v_name: str = "facebook/wav2vec2-base"
    pretrained: bool = True

   