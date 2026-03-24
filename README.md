# Audio Clustering with VAE

## Overview
This project performs unsupervised clustering of songs using audio feature extraction and a Variational Autoencoder (VAE). The repository includes scripts to prepare the dataset folders and organize the audio files before feature extraction and clustering.
## Dataset Sources

### English Audio Dataset
Download the English dataset from:
- Dataset name/link: MERGE Dataset
- URL: https://zenodo.org/records/13939205

### Bangla Audio Dataset
Download the Bangla dataset from:
- Dataset name/link: BanglaBeats
- URL: https://www.kaggle.com/datasets/thisisjibon/banglabeats3sec

## Create the virtual environment and install the libraries
python -m venv venv
./venv/Scripts/activate
pip install -r requirements.txt

## Preparing the English Audio Folder
### Metadata Creation
Run the notebook `dataset_en/metadata.ipynb`
### Transfering Audio Files to Target Directory
Run the notebook `dataset_en/fetch_audio.ipynb`

## Preparing the Bangla Audio Folder
Run the notebook `dataset_bn/fetch_audio.ipynb`

## Project Data Directory Structure
After preparing the data, your project should contain:

```text
data/
  audio/
    en/
    bn/
```

Where:
- data/audio/en/ contains the selected English audio files
- data/audio/bn/ contains the selected Bangla audio files