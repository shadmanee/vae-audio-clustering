# Audio Clustering with VAE

## Overview
This project performs unsupervised clustering of songs using audio feature extraction and a Variational Autoencoder (VAE). The repository includes scripts to prepare the dataset folders and organize the audio files before feature extraction and clustering.
## Dataset Sources

### English Audio Dataset
Download the English dataset from:
- Dataset name/directory: MERGE_Audio_Balanced
- URL: https://zenodo.org/records/13939205

### Bangla Audio Dataset
Download the Bangla dataset from:
- Dataset name/directory: BanglaBeats
- URL: https://www.kaggle.com/datasets/thisisjibon/banglabeats3sec

## Create the virtual environment and install the libraries
python -m venv venv
./venv/Scripts/activate
pip install -r requirements.txt

## Preparing the English Audio Folder
### 1. Metadata Creation
Run the notebook `notebooks/data_preparation/en/metadata_creation_en.ipynb`
### 2. Transfering Audio Files to Target Directory
Run the notebook `notebooks/data_preparation/en/fetch_audio_en.ipynb`
### 3. Download FFMPEG for Windows
- Download `ffmpeg-git-essentials.7z` from https://www.gyan.dev/ffmpeg/builds/
- Add the path to `ffmpeg.exe` to your system PATH in environment variables
### 4. Splitting Audio Files into 3-second Clips
Run the notebook `notebooks/data_preparation/en/audio_splitting_en.ipynb`

## Preparing the Bangla Audio Folder
Run the notebook `notebooks/data_preparation/bn/fetch_audio_bn.ipynb`

## Project Data Directory Structure
After preparing the data, your project should contain:

```text
data/
  audio/
    en/
    en_clips/
    bn_clips/
```

Where:
- data/audio/en_clips/ contains the selected English audio clips
- data/audio/bn_clips/ contains the selected Bangla audio clips