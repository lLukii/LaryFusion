# LaryFusion: Addressing data scarcity in non-invasive laryngeal cancer detection with multimodal feature learning. 
### Data Science Distinction, 2026 S.T. Yau Science Competition/2027 Regeneron STS Submission
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
Other hyperparameters (epochs, batch size, learning rate, wav2vec2 checkpoint, etc.) are also set in `src/config.py` rather than passed as CLI flags.

All commands below are run from `src/`.

#### Train LaryFusion (gated fusion + LeMDA-style VAE augmentation)
```bash
python laryfusion.py --name laryfusion_v1
```
| Flag | Default | Description |
| --- | --- | --- |
| `--name` | *(required)* | Run name; best checkpoint is saved to `checkpoints/{name}_best.pt` |
| `--use_lora` | `False` | Fine-tune the wav2vec2 encoder with LoRA |
| `--loss` | `cross_entropy` | Loss function to use for training |
| `--results_name` | `results` | Filename (under `graphs/`) for the loss/accuracy/confusion-matrix plot |

#### Train a baseline model
```bash
python baselines.py --name fusion_gated --model_type 1
```
| Flag | Default | Description |
| --- | --- | --- |
| `--name` | *(required)* | Run name; best checkpoint is saved to `checkpoints/{name}_best.pt` |
| `--model_type` | `0` | `0` fusion, `1` gated fusion, `2` audio-only wav2vec2, `3` ResNet-18 on mel-spectrograms |
| `--use_lora` | `False` | Fine-tune the wav2vec2 encoder with LoRA |
| `--loss` | `cross_entropy` | Loss function to use for training |
| `--results_name` | `results` | Filename (under `graphs/`) for the loss/accuracy/confusion-matrix plot |

#### Train the SVM baseline
```bash
python svm.py --use_demographics
```
| Flag | Default | Description |
| --- | --- | --- |
| `--use_demographics` | off | Include demographic/symptom background features alongside MFCCs |

#### Run inference with a trained checkpoint
```bash
python inference.py --model_path checkpoints/laryfusion_v1_best.pt
```
| Flag | Default | Description |
| --- | --- | --- |
| `--model_path` | *(required)* | Path to a saved model state dict (`.pt`) |
| `--dataset_path` | `config.dataset_path` | Override the dataset path from config |
| `--output` | `inference_results.png` | Path to save the output figure |
