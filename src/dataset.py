from config import Config
import librosa
import os
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2FeatureExtractor

config = Config()
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.w2v_name)


def load_wavs(include_demographics=False): 
    patients = {}
    for voice_type in ["benign", "malignant", "normal"]:
        for sample in os.listdir(os.path.join(config.dataset_path, voice_type)):
            if sample.split(".")[1] != "wav":
                continue
            user_id = sample.split(".")[0]
            signal, _ = librosa.load(os.path.join(config.dataset_path, voice_type, sample),
                                    sr=config.sampling_rate)
            processed = feature_extractor(signal, sampling_rate=config.sampling_rate,
                                  return_tensors="pt", padding=True, return_attention_mask=True)
            signal = processed.input_values
            attn_mask = processed.attention_mask
            patients[user_id] = {
                    "signal" : signal,
                    "attention_mask" : attn_mask,
                    "label" : 1 if voice_type == "malignant" else 0,
                    "background" : None
            }
    
    if include_demographics: 
        benign_history = pd.read_excel(os.path.join(config.dataset_path, "benign", "medicalhistory.xlsx")).fillna(0)
        malignant_history = pd.read_excel(os.path.join(config.dataset_path, "malignant", "medicalhistory.xlsx")).fillna(0)
        normal_history = pd.read_excel(os.path.join(config.dataset_path, "normal", "medicalhistory.xlsx")).fillna(0)

        scaler = StandardScaler()
        # standardize only non-binary quantitative columns
        for col in benign_history.columns:
            if col not in ["ID", "Disease category"] and benign_history[col].nunique() > 2:
                benign_history[col] = scaler.fit_transform(benign_history[col].values.reshape(-1, 1))
                malignant_history[col] = scaler.fit_transform(malignant_history[col].values.reshape(-1, 1))
                normal_history[col] = scaler.fit_transform(normal_history[col].values.reshape(-1, 1))        
    
        for medical_history in [benign_history, malignant_history, normal_history]:
            for idx, row in medical_history.iterrows():
                user_id = row["ID"]
                patients[user_id]["background"] = row.drop(["ID", "Disease category"])

    return patients

def cross_validation(dataset): 
    patient_list = [patient for patient in dataset]
    train_data, test_data = train_test_split(patient_list, test_size=0.2, random_state=config.seed)
    train_data, test_data = torch.tensor(train_data), torch.tensor(test_data)

    return train_data, test_data

def batch_inputs(data): 
    batches, batch_labels = [], []
    labels = torch.tensor([patient["label"] for patient in data]).reshape(-1, 1)
    for idx in range(0, len(data), config.batch_size):
        if idx + config.batch_size <= len(data):
            batch = data[idx:idx+config.batch_size]
            batch_label = labels[idx:idx+config.batch_size]
        else:
            batch = data[idx:]
            batch_label = labels[idx:]

        batches.append(batch)
        batch_labels.append(batch_label)
    
    return torch.tensor(batches), torch.tensor(batch_labels)

