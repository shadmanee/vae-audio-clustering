# VAE for Hybrid Language Music Clustering

## Overview
This project implements an unsupervised learning pipeline using Variational Autoencoders (VAE) for clustering hybrid language (English + Bangla) music tracks. The pipeline extracts latent representations from audio and lyrics, performs clustering, and compares results against baseline methods.

**Course:** CSE715 - Neural Networks & Fuzzy Systems

**Team:** 
- 1000055925: Shadmanee Tasneem Mulk
- 1000056026: Tasnim Ullah Shakib

**Dataset Sources:**
- English: [MERGE - A Bimodal Audio-Lyrics Dataset](https://arxiv.org/pdf/2407.06060) — [Download](https://zenodo.org/records/13939205/files/MERGE_Bimodal_Balanced.zip?download=1)
- Bangla: [Kaggle - BanglaSongs YouTube Autocaptioned ASR](https://www.kaggle.com/datasets/nurerazeen/banglasongs-youtube-autocaptioned-asr?resource=download)

---

## Repository Structure

```
project/
├── data/
│   ├── audio/
│   │   ├── en/                  # Full English audio files
│   │   ├── en_clips/            # Clipped English audio segments
│   │   ├── bn/                  # Full Bangla audio files
│   │   └── bn_clips/            # Clipped Bangla audio segments
│   ├── features/
│   │   ├── en_bn/               # Mel spectrogram features (English + Bangla)
│   │   └── lyrics/
│   │       ├── en/              # English lyrics text files
│   │       └── bn/              # Bangla lyrics text files
│   ├── embeddings/
│   │   └── en_bn/               # Lyrics embeddings (English + Bangla)
│   └── metadata/                # English, Bangla, and combined (en_bn) metadata
├── notebooks/
│   ├── data_preparation/
│   │   ├── en/
│   │   │   ├── metadata_creation_en.ipynb
│   │   │   ├── fetch_audio_en.ipynb
│   │   │   ├── audio_splitting_en.ipynb
│   │   │   └── fetch_lyrics_en.ipynb
│   │   ├── bn/
│   │   │   ├── metadata_creation_bn.ipynb
│   │   │   ├── fetch_audio_bn.ipynb
│   │   │   ├── audio_splitting_bn.ipynb
│   │   │   └── fetch_lyrics_bn.ipynb
│   │   └── metadata_creation_en_bn.ipynb
│   └── feature_extraction/
│       ├── audio_feature_extraction.ipynb
│       └── lyrics_feature_extraction.ipynb
├── src/
│   ├── models/
│   ├── utils/
│   ├── datasets.py
│   ├── config.py
│   └── run_files/
│       ├── 0_ae_baseline_audio_only.py
│       ├── 0_pca_baseline_audio_only.py
│       ├── 1_easy_1_basic_vae_audio_only.py
│       ├── 2_medium_1_conv_vae_audio_only.py
│       ├── 3_hard_1_beta_vae_audio_only.py
│       ├── 4_hard_1_cvae_audio_only.py
│       ├── 5_easy_2_basic_vae_audio_lyrics.py
│       ├── 6_medium_2_conv_vae_audio_lyrics.py
│       ├── 7_hard_2_beta_vae_audio_lyrics.py
│       ├── 8_hard_2_cvae_audio_lyrics.py
│       ├── 9_easy_3_basic_vae_audio_lyrics_genre.py
│       ├── 10_medium_3_conv_vae_audio_lyrics_genre.py
│       ├── 11_hard_3_beta_vae_audio_lyrics_genre.py
│       └── 12_hard_3_cvae_audio_lyrics_genre.py
├── results/
│   ├── clustering/              # Clustering plots, metrics, and clustering.csv
│   ├── trials/                  # Optuna tuning plots and training histories
│   ├── final/                   # Final model training plots and histories
│   ├── models/                  # Saved AE, VAE, and PCA model files
│   └── visualizations/          # t-SNE visualizations of latent space clusters
├── README.md
└── requirements.txt
```

---

## Setup

### 1. Clone the Repository
```bash
git clone <repo_url>
cd <repo_name>
pip install -r requirements.txt
```

### 2. Download Datasets
Download the datasets from the links above.

---

## Data Preparation

Run the following notebooks **in order** to prepare the data directory:

| Step | Notebook |
|------|----------|
| 1 | `notebooks/data_preparation/bn/metadata_creation_bn.ipynb` |
| 2 | `notebooks/data_preparation/en/metadata_creation_en.ipynb` |
| 3 | `notebooks/data_preparation/metadata_creation_en_bn.ipynb` |
| 4 | `notebooks/data_preparation/bn/fetch_audio_bn.ipynb` |
| 5 | `notebooks/data_preparation/en/fetch_audio_en.ipynb` |
| 6 | `notebooks/data_preparation/bn/audio_splitting_bn.ipynb` |
| 7 | `notebooks/data_preparation/en/audio_splitting_en.ipynb` |
| 8 | `notebooks/feature_extraction/audio_feature_extraction.ipynb` |
| 9 | `notebooks/data_preparation/bn/fetch_lyrics_bn.ipynb` |
| 10 | `notebooks/data_preparation/en/fetch_lyrics_en.ipynb` |
| 11 | `notebooks/feature_extraction/lyrics_feature_extraction.ipynb` |

---

## Running Models

All model training and clustering scripts are run from the project root using the `-m` flag.

### Step 1 — Train baseline and audio-only models
```bash
python -m src.run_files.0_ae_baseline_audio_only
python -m src.run_files.0_pca_baseline_audio_only
python -m src.run_files.0_direct_spectral_clustering_baseline_audio_only
python -m src.run_files.1_easy_1_basic_vae_audio_only
python -m src.run_files.2_medium_1_conv_vae_audio_only
python -m src.run_files.3_hard_1_beta_vae_audio_only
python -m src.run_files.4_hard_1_cvae_audio_only
```

### Step 2 — Run downstream clustering for audio + lyrics and audio + lyrics + genre
Once models from Step 1 are saved, load them and run the following scripts.
To load instead of train, comment out the training and saving code in the Step 1 files and add the loading code before running these:

```bash
python -m src.run_files.5_easy_2_basic_vae_audio_lyrics
python -m src.run_files.6_medium_2_conv_vae_audio_lyrics
python -m src.run_files.7_hard_2_beta_vae_audio_lyrics
python -m src.run_files.8_hard_2_cvae_audio_lyrics
python -m src.run_files.9_easy_3_basic_vae_audio_lyrics_genre
python -m src.run_files.10_medium_3_conv_vae_audio_lyrics_genre
python -m src.run_files.11_hard_3_beta_vae_audio_lyrics_genre
python -m src.run_files.12_hard_3_cvae_audio_lyrics_genre
```

---

## Results

All results are stored under `results/` after running the scripts:

| Directory | Contents |
|-----------|----------|
| `clustering/` | One subdirectory per model-modality combination containing elbow plots, silhouette plots, and metrics; `clustering.csv` aggregates all metrics across all model-modality-algorithm combinations |
| `trials/` | Optuna hyperparameter tuning plots and per-trial training histories for all VAE models |
| `final/` | Training plots and histories for final models trained with best Optuna parameters |
| `models/` | Saved model files for AE, VAE, and PCA (audio-only) |
| `visualizations/` | t-SNE visualizations of latent space clusters for all three clustering algorithms |

---

## Models Implemented

| Task | Model | Modalities |
|------|-------|------------|
| Easy | Basic VAE + KMeans | Audio, Audio+Lyrics, Audio+Lyrics+Genre |
| Medium | Convolutional VAE + KMeans, Agglomerative, DBSCAN | Audio, Audio+Lyrics, Audio+Lyrics+Genre |
| Hard | Beta-VAE + KMeans, Agglomerative, DBSCAN | Audio, Audio+Lyrics, Audio+Lyrics+Genre |
| Hard | Conditional VAE + KMeans, Agglomerative, DBSCAN | Audio, Audio+Lyrics, Audio+Lyrics+Genre |
| Baseline | PCA + KMeans | Audio |
| Baseline | Autoencoder + KMeans | Audio |
| Baseline | Direct Spectral Features Clustering with KMeans | Audio |