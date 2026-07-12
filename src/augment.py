from dataset import load_wavs, cross_validation
import os, shutil
from config import Config

if __name__ == "__main__":
    dataset = load_wavs(include_demographics=False)
    config = Config()
    train_data, _ = cross_validation(dataset)
    positive_ids = [p["id"] for p in train_data if p["label"] == 1]

    try: 
        os.mkdir(f"{config.dataset_path}/augmented")
        for user_id in positive_ids:
            shutil.copyfile(f"{config.dataset_path}/malignant/{user_id}.wav", f"{config.dataset_path}/augmented/{user_id}.wav")
    
    except FileExistsError:
        print("Augmented folder already exists. Please delete it before running this script.")

        