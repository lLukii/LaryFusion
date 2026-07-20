from config import Config
import librosa
import os
import pandas as pd
import numpy as np
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

config = Config()
spec_args = {
    'sample_rate': 16000,
    'win_length': 256 * 4,
    'hop_length': 256,
    'n_fft': 1024,
    'f_min': 20.0,
    'f_max': 16000 / 2.0,
    'n_mels': 80,
    'power': 1.0,
    'normalized': True
}

def load_wavs(include_demographics=False): 
    patients = {}
    spec_transform = T.MelSpectrogram(**spec_args)
    for voice_type in ["benign", "malignant", "normal", "synthetic"]:
        for sample in os.listdir(os.path.join(config.dataset_path, voice_type)):
            if sample.split(".")[-1] != "wav":
                continue
            user_id = sample.split(".")[0]
            signal, _ = librosa.load(os.path.join(config.dataset_path, voice_type, sample), sr=config.sampling_rate)
            if len(signal) < config.padding: 
                signal = np.pad(signal, (0, config.padding - len(signal)), mode="constant", constant_values=0)
            else: 
                signal = signal[:config.padding]
            
            mel_spectrogram = spec_transform(torch.tensor(signal, dtype=torch.float32).unsqueeze(0))
            patients[user_id] = {
                    "spec" : mel_spectrogram,
                    "signal" : signal,
                    "label" : 1 if voice_type in ["malignant", "synthetic"] else 0,
                    "background" : None
            }
    
    if include_demographics:
        benign_history = pd.read_excel(os.path.join(config.dataset_path, "benign", "medicalhistory.xlsx")).fillna(0)
        malignant_history = pd.read_excel(os.path.join(config.dataset_path, "malignant", "medicalhistory.xlsx")).fillna(0)
        normal_history = pd.read_excel(os.path.join(config.dataset_path, "normal", "medicalhistory.xlsx")).fillna(0)

        for medical_history in [benign_history, malignant_history, normal_history]:
            for _, row in medical_history.iterrows():
                user_id = row["ID"]
                patients[user_id]["background"] = row.drop(["ID", "Disease category"])

    return patients

def cross_validation(dataset):
    patient_list = list(dataset.values())
    train_data, test_data = train_test_split(patient_list, test_size=0.2, random_state=config.seed)

    # normalize demographic features based on statistics in each split
    if train_data[0]["background"] is not None:
        bg_cols = train_data[0]["background"].index
        quant_cols = [col for col in bg_cols
                      if pd.Series([p["background"][col] for p in train_data]).nunique() > 2] 
        
        # only normalize non-binary rows
        train_bg = pd.DataFrame([p["background"] for p in train_data])
        scaler = StandardScaler()
        scaler.fit(train_bg[quant_cols])

        for split in [train_data, test_data]:
            for p in split:
                bg = p["background"].copy()
                bg[quant_cols] = scaler.transform(bg[quant_cols].values.reshape(1, -1).astype(float))[0]
                p["background"] = bg

    return train_data, test_data

class ThroatCancerDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        p = self.data[idx]
        demo = (torch.tensor(p["background"].values.astype(float), dtype=torch.float32)
                if p["background"] is not None else None)
        return p["spec"], demo, self.labels[idx]


def collate_fn(batch):
    specs, demos, labels = zip(*batch)
    demographics = torch.stack(demos) if demos[0] is not None else None
    return torch.stack(specs), demographics, torch.stack(list(labels))


def batch_inputs(data, oversample=False, shuffle=False):
    labels = torch.tensor([patient["label"] for patient in data])
    sampler = None
    if oversample:
        class_weights = 1. / torch.bincount(labels)
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(sample_weights, len(data), replacement=True)

    return DataLoader(ThroatCancerDataset(data, labels),
                      batch_size=config.batch_size, sampler=sampler if shuffle else None, 
                      collate_fn=collate_fn)