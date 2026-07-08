import os, shutil
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    data_path = "/Users/llukii/Desktop/School/Programming/Multimodal throat cancer/dataset"
    os.makedirs(f"{data_path}/augmented", exist_ok=False) 
    id_list = [
        name.split(".")[0] for name in os.listdir(f"{data_path}/malignant") if name.endswith(".wav")
    ]
    train_split, _ = train_test_split(id_list, test_size=0.2, random_state=42)
    for id in train_split: 
        shutil.copy(f"{data_path}/malignant/{id}.wav", f"{data_path}/augmented/{id}.wav")