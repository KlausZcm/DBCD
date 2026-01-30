# DBCD: Dual Branch Contrastive Diffusion Network for Unsupervised Building Extraction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation of the paper: **"DBCD: Dual Branch Contrastive Diffusion Network for Unsupervised Building Extraction"**.

DBCD is a novel unsupervised framework that tackles feature entanglement and boundary inconsistency in remote sensing building extraction. It integrates a **Conditional Branch** (driven by superpixel priors) and a **Diffusion Branch** (driven by edge-aware geometric constraints) via a dynamic contrastive learning mechanism.

![Framework Architecture](assets/framework.png)

## 🌟 Key Features

* **Unsupervised Paradigm:** Requires **zero** manual pixel-level annotations. Training is driven purely by intrinsic pseudo-labels derived from SpPre (Similarity-Guided Superpixel Preprocessing).
* **Dual-Branch Architecture:**
    * **Conditional Branch:** Encodes superpixel features ($z_c$) to provide semantic anchors.
    * **Diffusion Branch:** Reconstructs segmentation masks ($z_d$) with probabilistic denoising.
* **Edge-Aware Contrastive Loss:** Enforces geometric consistency by weighting boundaries in the latent space.
* **Dynamic Curriculum Learning:** Automatically adjusts loss weights ($\lambda_1, \lambda_2$) to transition from semantic alignment to geometric refinement.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/DBCD.git](https://github.com/yourusername/DBCD.git)
    cd DBCD
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    conda create -n dbcd python=3.8
    conda activate dbcd
    ```

3.  **Install dependencies:**
    ```bash
    pip install torch torchvision --extra-index-url [https://download.pytorch.org/whl/cu116](https://download.pytorch.org/whl/cu116)
    pip install numpy opencv-python scikit-image scipy tqdm pyyaml albumentations
    ```

## 📂 Data Preparation

We use the **ISPRS Vaihingen** and **Potsdam** datasets.

1.  **Download Data:** Obtain the datasets from the [ISPRS official website](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx).
2.  **Organize Raw Data:**
    Place your raw GeoTIFFs (images) in a `raw_data/` directory.
    ```text
    DBCD/
    ├── raw_data/
    │   ├── top_potsdam_2_10_RGB.tif
    │   ├── top_potsdam_2_11_RGB.tif
    │   └── ...
    ├── configs/
    └── ...
    ```
3.  **Run Preprocessing (SpPre & Patching):**
    This script performs sliding window cropping, generates SpPre pseudo-labels, and extracts superpixel features ($c_i$).
    *For Potsdam, it automatically applies the 25.2% high-confidence filtering strategy.*

    ```bash
    python data/make_patches.py
    ```
    > **Output:** This will generate `processed_data/` containing `images/`, `masks/` (pseudo-labels), and `features/` (.npy files).

## 🚀 Training

To train the DBCD model from scratch using the generated pseudo-labels:

1.  **Configure:** Check `configs/default.yaml` to adjust batch size, learning rate, or paths.
2.  **Run Training:**
    ```bash
    python train.py
    ```
    * **Checkpoints:** Saved to `checkpoints/`.
    * **Logs:** Progress bar displays `Loss`, `Diff` (Diffusion Loss), and dynamic weights `L1` (Contrastive) / `L2` (Refinement).

## ⚡ Inference & Evaluation

To evaluate the model on full-tile images using Sliding Window Inference and DDIM Sampling (100 steps):

```bash
python inference.py
```

* **Input:** Full tiles from `raw_data/` (or a specified test set).
* **Method:** Overlap-tile strategy (50% overlap) to reduce boundary artifacts.
* **Metrics:** Reports **mIoU**, **F1-score**, and **BoundF** (Boundary F-score).
* **Visualization:** Prediction masks are saved to `results/visualization/`.

## ⚙️ Project Structure

```text
DBCD/
├── configs/
│   └── default.yaml         # Hyperparameters & Paths
├── data/
│   ├── make_patches.py      # Preprocessing: Crop + SpPre + Filter
│   └── dataset.py           # PyTorch Dataset Loader
├── model/
│   ├── dbcd.py              # Dual-Branch Network Architecture
│   ├── gaussian_diffusion.py# Diffusion Process & DDIM Sampler
│   └── sppre.py             # Superpixel Preprocessing Module
├── train.py                 # Main Training Script
├── inference.py             # Inference & Evaluation Script
└── utils.py                 # Metrics (BoundF) & Edge Extraction
```

## 📝 Citation

The corresponding BibTeX citation entry will be updated once the paper is officially published and assigned a DOI.

## 📄 License

This project is licensed under the MIT License.
