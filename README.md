# LaryFusion: Addressing data scarcity in non-invasive laryngeal cancer detection with multimodal feature learning. 
### ⚠️ Work in progress ⚠️
### Data Science Distinction, 2026 S.T. Yau Science Competition / 2027 Regeneron STS Submission
### Abstract: 
Laryngeal cancer is traditionally diagnosed using advanced clinical methods, which are both uncomfortable and inaccessible to patients with limited access to healthcare. Recently, using Machine Learning (ML) based methods to classify voice data has been considered a friendly and economically sustainable alternative for laryngeal cancer diagnosis. However, existing models often suffer from scarce training data, making them hard to train and validate in practice. To tackle this issue, we propose a custom open-source multi-modal deep learning architecture that addresses the issue of data scarcity. Our method uses a fine-tuned Wav2Vec 2.0 encoder to create rich audio representations, which are then combined with demographic/symptom data using a linear gating mechanism to minimize data ambiguity. Furthermore, we train a Variational Autoencoder (VAE) based on that creates reconstructed multimodal data to further improve the model's robustness to unseen data. Each model was trained on focal loss to prevent class bias, and the generator was trained adversarially against the rest of the model. Applying this architecture to the Far Eastern Memorial Hospital (FEMH) dataset, our proposed method outperforms existing methods in Sensitivity, Specificity, and AUROC scores.

### Usage

#### Setup
```bash
pip install -r requirements.txt
```

Expects a dataset directory (path set via `dataset_path` in `src/config.py`) laid out as:
```
dataset/
  benign/       *.wav + medicalhistory.xlsx
  malignant/    *.wav + medicalhistory.xlsx
  normal/       *.wav + medicalhistory.xlsx
```