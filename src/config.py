import torch
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    """
    Central hyperparameters and paths shared by every script in this project
    (dataset.py, baselines.py, laryfusion.py, svm.py, inference.py). Instantiate
    once per script (`config = Config()`) and edit the defaults below to tune
    training rather than passing extra CLI flags around.
    """
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    sampling_rate: int = 16000
    padding: int = 48000 # 3 seconds
    num_epochs: int = 15
    patience: int = 5
    batch_size: int = 64
    dataset_path: Path = Path("/Users/llukii/Desktop/School/Programming/Multimodal throat cancer/dataset")
    model_path: str = ""
    w2v_name: str = "facebook/wav2vec2-base"
    pretrained: bool = True

   