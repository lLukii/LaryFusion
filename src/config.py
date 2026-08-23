from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class Config:
    # general
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    sampling_rate: int = 16000
    padding: int = 48000 # 3 seconds
    dataset_path: Path = Path("/teamspace/studios/this_studio/dataset")
    model_path: str = Path("/teamspace/studios/this_studio/checkpoints")
    w2v_name: str = "facebook/wav2vec2-base"
    pretrained: bool = True

    # model training
    num_epochs: int = 20
    warmup_steps: int = 2
    patience: int = 5
    batch_size: int = 64
    lr: float = 1e-4
    max_grad_norm: float = 1.0