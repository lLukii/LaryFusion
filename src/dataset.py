"""
Shared data-loading pipeline for every training script in this project.
Centralizes: walking the raw `.wav` dataset + `medicalhistory.xlsx` files,
the four supported audio feature representations, the train/test split +
demographic normalization, and the PyTorch Dataset/DataLoader/batch-sampler
plumbing used by the neural-net baselines.

Audio feature backends (`feature_type` passed to `load_patients`):
    "wav2vec2"  - Wav2Vec2FeatureExtractor-processed raw waveform (larycl.py, ablations.py)
    "raw"       - padded/truncated raw waveform, no processing (cnn1d.py)
    "opensmile" - OpenSMILE eGeMAPSv02 functionals (logisticreg.py)
    "praat"     - Praat/parselmouth jitter/shimmer/F0/HNR features (svm.py)
Each backend's third-party dependency (transformers/opensmile/parselmouth)
is imported lazily inside its extraction function, so a script that only
uses one backend doesn't need the others installed.
"""

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from config import Config

config = Config()

VOICE_TYPES = ["benign", "malignant", "normal"]

_wav2vec2_extractor = None
_opensmile_extractor = None


def _get_wav2vec2_extractor():
    global _wav2vec2_extractor
    if _wav2vec2_extractor is None:
        from transformers import Wav2Vec2FeatureExtractor
        _wav2vec2_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.w2v_name)
    return _wav2vec2_extractor


def _get_opensmile_extractor():
    global _opensmile_extractor
    if _opensmile_extractor is None:
        import opensmile
        _opensmile_extractor = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return _opensmile_extractor


def _pad_or_truncate(signal: np.ndarray) -> np.ndarray:
    if len(signal) < config.padding:
        return np.pad(signal, (0, config.padding - len(signal)), mode="constant")
    return signal[:config.padding]


def _extract_praat_features(wav_path) -> np.ndarray:
    """
    Ten Praat-derived perturbation features (Rehman et al., 2024, Table 3):
    F0, periodicity (HNR), four jitter variants, and four shimmer variants.
    """
    import parselmouth
    from parselmouth.praat import call
    PITCH_FLOOR, PITCH_CEILING = 75.0, 500.0

    sound = parselmouth.Sound(str(wav_path))
    pitch = sound.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    f0 = call(pitch, "Get mean", 0, 0, "Hertz")

    harmonicity = sound.to_harmonicity()
    periodicity = call(harmonicity, "Get mean", 0, 0)

    point_process = call(sound, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)
    jitter_absolute = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)

    shimmer_absolute = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq3 = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq5 = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    features = [f0, periodicity, jitter_absolute, jitter_local, jitter_rap, jitter_ppq5,
                shimmer_absolute, shimmer_local, shimmer_apq3, shimmer_apq5]
    return np.nan_to_num(np.array(features, dtype=float), nan=0.0)


def load_patients(feature_type: str = "wav2vec2", include_demographics: bool = False) -> dict:
    """
    Load every .wav under `config.dataset_path/{benign,malignant,normal}`
    and extract the requested audio feature representation (see module
    docstring for the four supported `feature_type`s).

    Args:
        feature_type: "wav2vec2", "raw", "opensmile", or "praat".
        include_demographics: also attach each patient's demographic/symptom
            row (read from `medicalhistory.xlsx`) under the "background" key.

    Returns:
        dict mapping patient ID -> {"signal", "label", "background"}.
    """
    patients = {}
    for voice_type in VOICE_TYPES:
        folder = config.dataset_path / voice_type
        for wav_path in folder.glob("*.wav"):
            user_id = wav_path.stem

            if feature_type == "praat":
                signal = _extract_praat_features(wav_path)
            else:
                raw, _ = librosa.load(wav_path, sr=config.sampling_rate)
                if feature_type == "wav2vec2":
                    raw = _pad_or_truncate(raw)
                    signal = _get_wav2vec2_extractor()(raw, sampling_rate=config.sampling_rate,
                                                         return_tensors="pt").input_values
                elif feature_type == "raw":
                    raw = _pad_or_truncate(raw)
                    signal = torch.from_numpy(raw.astype(np.float32))
                elif feature_type == "opensmile":
                    signal = _get_opensmile_extractor().process_signal(raw, config.sampling_rate).values.flatten()
                else:
                    raise ValueError(f"Unknown feature_type: {feature_type!r}")

            patients[user_id] = {
                "signal": signal,
                "label": 1 if voice_type == "malignant" else 0,
                "background": None,
            }

        if include_demographics:
            history = pd.read_excel(folder / "medicalhistory.xlsx").fillna(0)
            for _, row in history.iterrows():
                patients[row["ID"]]["background"] = row.drop(["ID", "Disease category"])

    return patients


def cross_validation(patients: dict) -> tuple[list, list]:
    """
    Split a `load_patients` dataset into train/test patient lists and
    z-score normalize non-binary demographic/symptom columns, fitting the
    scaler on the train split only to avoid test-set leakage.
    """
    patient_list = list(patients.values())
    train_data, test_data = train_test_split(patient_list, test_size=0.2, random_state=config.seed)

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


def stratified_kfold(patients: dict, n_splits: int = 5):
    """
    Yield `n_splits` stratified (train_data, test_data) folds from a
    `load_patients` dataset (class-balance preserved in every fold, unlike
    `cross_validation`'s single unstratified split - useful given how few
    malignant examples there are).

    Each fold's demographic/symptom columns are z-score normalized
    independently, fit on that fold's train split only. Fold-local shallow
    copies of each patient dict are used so a patient appearing in multiple
    folds' train splits (unavoidable with k-fold) never gets normalized
    on top of a previous fold's already-normalized values.
    """
    patient_list = list(patients.values())
    labels = [p["label"] for p in patient_list]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
    for train_idx, test_idx in skf.split(patient_list, labels):
        train_data = [dict(patient_list[i]) for i in train_idx]
        test_data = [dict(patient_list[i]) for i in test_idx]

        if train_data[0]["background"] is not None:
            bg_cols = train_data[0]["background"].index
            quant_cols = [col for col in bg_cols
                          if pd.Series([p["background"][col] for p in train_data]).nunique() > 2]

            train_bg = pd.DataFrame([p["background"] for p in train_data])
            scaler = StandardScaler()
            scaler.fit(train_bg[quant_cols])

            for split in [train_data, test_data]:
                for p in split:
                    bg = p["background"].copy()
                    bg[quant_cols] = scaler.transform(bg[quant_cols].values.reshape(1, -1).astype(float))[0]
                    p["background"] = bg

        yield train_data, test_data


class ThroatCancerDataset(Dataset):
    """Wraps a `load_patients`/`cross_validation` split for `DataLoader`."""
    def __init__(self, data: list, labels: torch.Tensor):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        p = self.data[idx]
        demo = (torch.tensor(p["background"].values.astype(float), dtype=torch.float32)
                if p["background"] is not None else None)
        return p["signal"].squeeze(0), demo, self.labels[idx]


def collate_fn(batch):
    signals, demos, labels = zip(*batch)
    demographics = torch.stack(demos) if demos[0] is not None else None
    return torch.stack(signals), demographics, torch.stack(list(labels))


def batch_inputs(data: list, shuffle: bool = False) -> DataLoader:
    """
    Wrap a patient list (from `cross_validation`) in a DataLoader.

    Args:
        shuffle: whether to shuffle the data (ignored if `stratify=True`).
        stratify: use `StratifiedBatchSampler` so every batch carries a
            fixed malignant/benign-normal ratio; typically only for train.
    """
    labels = torch.tensor([patient["label"] for patient in data])
    dataset = ThroatCancerDataset(data, labels)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, collate_fn=collate_fn)
