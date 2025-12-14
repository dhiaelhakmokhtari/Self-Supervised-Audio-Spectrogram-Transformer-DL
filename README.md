# GTZAN Audio Classification: CNN vs. SSAST Transformer

**Course:** Deep Learning (M2 CNS - Specialization in Autonomous Systems)  

---

## Project Overview
This project investigates the application of **Self-Supervised Audio Spectrogram Transformers (SSAST)** for Music Genre Classification (MGC) on the GTZAN dataset. 

We compare a robust **CNN Baseline** against a fine-tuned **SSAST**, focusing on resolving the challenges of applying heavy Transformer architectures to small audio datasets.

### Key Features
* **Dual Architecture Study:** Comparison between a standard 5-layer CNN and the SSAST (ViT-based) architecture.
* **Dynamic Data Pipeline:** Implementation of a **RAM-Cached Dataset** with on-the-fly random cropping (10s segments from 30s tracks) to maximize generalization.
* **Advanced Regularization:** Use of **SpecAugment** (Time/Freq Masking) and **Label Smoothing** to prevent overfitting.
* **Memory Optimization:** Implementation of **Gradient Accumulation** and **Gradient Checkpointing** to fit the $O(N^2)$ attention mechanism on standard GPUs.
* **Fine-Tuning Strategy:** Utilization of **Layer-Wise Learning Rate Decay (LLRD)** to preserve pre-trained AudioSet weights while adapting the classification head.

---

## Results

*See the `report/` folder for the full PDF report containing t-SNE visualizations and detailed confusion matrices.*

---

## Download Pre-Trained Models

Due to file size limits on GitHub, the trained model weights are hosted on **Google Drive**.

👉 **[Click Here to Download Models](INSERT_YOUR_GOOGLE_DRIVE_LINK_HERE)**

**Files included:**
* `ssast_best.pth` (Our best Transformer model - Epoch 117)
* `cnn_baseline.pth` (The reference CNN model)

*Please download these files and place them in the root directory of the project to run inference.*

---

## Installation & Setup

### 1. Prerequisites
This project requires Python 3.8+ and PyTorch with CUDA support.

```bash
# Clone the repository
git clone [https://github.com/dhiaelhakmokhtari/Self-Supervised-Audio-Spectrogram-Transformer-DL](https://github.com/dhiaelhakmokhtari/Self-Supervised-Audio-Spectrogram-Transformer-DL)
cd your-repo-name

# Install dependencies
pip install torch torchaudio pandas numpy soundfile timm tqdm matplotlib seaborn scikit-learn
