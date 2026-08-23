"""
Shared data-loading pipeline for every training script in this project.
Centralizes: walking the raw `.wav` dataset + `medicalhistory.xlsx` files,
the four supported audio feature representations, the train/test split +
demographic normalization, and the PyTorch Dataset/DataLoader/batch-sampler
plumbing used by the neural-net baselines. 
"""

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio.functional as AF
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from config import Config
from tqdm import tqdm

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


def _extract_from_raw(raw: np.ndarray, feature_type: str):
    """Feature extraction shared by `load_patients`'s clean and augmented-raw paths."""
    if feature_type == "wav2vec2":
        raw = _pad_or_truncate(raw)
        return _get_wav2vec2_extractor()(raw, sampling_rate=config.sampling_rate,
                                          return_tensors="pt").input_values
    elif feature_type == "raw":
        raw = _pad_or_truncate(raw)
        return torch.from_numpy(raw.astype(np.float32))
    elif feature_type == "opensmile":
        return _get_opensmile_extractor().process_signal(raw, config.sampling_rate).values.flatten()
    else:
        raise ValueError(f"Unknown feature_type: {feature_type!r}")


def band_stop_augment(signal: torch.Tensor, sr: int) -> torch.Tensor:
    central_freq = np.random.uniform(200, 4000)
    bandwidth_fraction = np.random.uniform(0.05, 2.0)
    signal = signal.to(config.device)
    return AF.bandreject_biquad(signal, sr, central_freq, Q=1.0 / bandwidth_fraction)


def gaussian_noise_augment(signal: torch.Tensor, sr: int = None) -> torch.Tensor:
    signal = signal.to(config.device)
    noise_scale = torch.empty_like(signal).uniform_(0.001, 0.03)
    return signal + signal * noise_scale


def pitch_shift_augment(signal: torch.Tensor, sr: int) -> torch.Tensor:
    n_steps = int(np.random.randint(-6, 7))
    signal = signal.to(config.device)
    shifter = T.PitchShift(sample_rate=sr, n_steps=n_steps).to(config.device)
    with torch.no_grad():
        return shifter(signal)

AUGMENTATIONS = {
    "band_stop": band_stop_augment,
    "gaussian_noise": gaussian_noise_augment,
    "pitch_shift": pitch_shift_augment,
}


def apply_augmentations(signal: torch.Tensor, sr: int, methods=tuple(AUGMENTATIONS)) -> torch.Tensor:
    for method in methods:
        signal = AUGMENTATIONS[method](signal, sr)
    return signal


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


def load_patients(feature_type: str = "wav2vec2", include_demographics: bool = False, augment: bool = False) -> dict:
    """
    Load every .wav under `config.dataset_path/{benign,malignant,normal}`
    and extract the requested audio feature representation (see module
    docstring for the four supported `feature_type`s).

    Args:
        feature_type: "wav2vec2", "raw", "opensmile", or "praat".
        include_demographics: also attach each patient's demographic/symptom
            row (read from `medicalhistory.xlsx`) under the "background" key.
        augment: also run the raw waveform through all three Section 4
            augmentations (combined) and extract the same feature
            representation from it, stored under "augmented_signal" -
            computed once here at load time rather than repeatedly at train
            time. Not supported for feature_type="praat" (stays None).

    Returns:
        dict mapping patient ID -> {"signal", "augmented_signal", "label", "background"}.
    """
    patients = {}
    for voice_type in VOICE_TYPES:
        folder = config.dataset_path / voice_type
        for wav_path in tqdm(list(folder.glob("*.wav")), desc=f"Loading {voice_type}"):
            user_id = wav_path.stem

            augmented_signal = None
            if feature_type == "praat":
                signal = _extract_praat_features(wav_path)
            else:
                raw, _ = librosa.load(wav_path, sr=config.sampling_rate)
                signal = _extract_from_raw(raw, feature_type)
                if augment:
                    aug_raw = apply_augmentations(torch.from_numpy(raw.astype(np.float32)), config.sampling_rate)
                    augmented_signal = _extract_from_raw(aug_raw.detach().cpu().numpy(), feature_type)

            patients[user_id] = {
                "signal": signal,
                "augmented_signal": augmented_signal,
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
    """
    patient_list = list(patients.values())
    labels = [p["label"] for p in patient_list]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
    for train_idx, test_idx in skf.split(patient_list, labels):
        train_data = [dict(patient_list[i]) for i in train_idx]
        test_data = [dict(patient_list[i]) for i in test_idx]
        np.random.shuffle(train_data)
        
        # normalize non-binary colums in tabular data
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
    """Wrap a patient list (from `cross_validation`/`stratified_kfold`) in a DataLoader."""
    labels = torch.tensor([patient["label"] for patient in data])
    dataset = ThroatCancerDataset(data, labels)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, collate_fn=collate_fn)
