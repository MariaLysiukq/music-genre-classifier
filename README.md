# Music Genre Classification

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)
![Librosa](https://img.shields.io/badge/librosa-audio-lightgrey.svg)

## Overview
This repository contains a Machine Learning project for classifying music genres based on audio signals. It uses `librosa` to extract Mel-Frequency Cepstral Coefficients (MFCCs) from audio files and trains a **Random Forest Classifier** to predict the genre of unseen tracks. 

The model was initially trained using a subset of the famous [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).

## Features
- **Audio Feature Extraction:** Extracts MFCCs from raw audio waveforms.
- **Machine Learning Model:** Implements a robust Random Forest Classifier.
- **Custom Inference:** Easily predict the genre of your own `.wav` files.

## Tech Stack
- **Python** (Core language)
- **librosa** (Audio and music processing)
- **scikit-learn** (Machine learning algorithms)
- **NumPy & Pandas** (Data manipulation)

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/music-genre-classifier.git](https://github.com/YOUR_USERNAME/music-genre-classifier.git)
   cd music-genre-classifier
    ```
    Create and activate a virtual environment:
   ```bash

    python -m venv venv
    # For Windows:
    .\venv\Scripts\activate
    # For macOS/Linux:
    source venv/bin/activate
    ```
    Install dependencies:
   ```bash

    pip install librosa scikit-learn numpy pandas matplotlib
    ```

Usage

    1 Place your training data inside a directory named GTZAN_student (or adjust the path in the script).

    2 Place a custom audio file (e.g., my_music.wav) in the root directory for testing.

    3 Run the classification script:
    python main.py