import os
import librosa
import numpy as np
import pickle
import argparse

# Load precomputed clean data statistics
STATS_FILE = "models/clean_data_stats.pkl"

def load_clean_stats():
    """Loads the precomputed mean and standard deviation of clean speech."""
    with open(STATS_FILE, "rb") as f:
        return pickle.load(f)

def laplacian_deviation(sample_mfcc, clean_mean, clean_std):
    """Computes the Laplacian Deviation Metric (LDM)."""
    deviation = np.abs(sample_mfcc - clean_mean) / (clean_std + 1e-6)
    return np.mean(deviation)  # Average deviation across all MFCC features

def detect_poisoned_sample(file_path, threshold=3.0):
    """Detects if an audio sample is poisoned based on deviation from clean stats."""
    clean_stats = load_clean_stats()
    clean_mean, clean_std = clean_stats["mean"], clean_stats["std"]

    # Process new audio sample
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    sample_mfcc_mean = mfcc.mean(axis=1)

    # Compute LDM
    ldm_score = laplacian_deviation(sample_mfcc_mean, clean_mean, clean_std)

    # Classify as clean or poisoned
    if ldm_score > threshold:
        print(f"⚠️ ALERT: {file_path} is POISONED! (LDM Score: {ldm_score:.2f})")
    else:
        print(f"✅ SAFE: {file_path} is CLEAN. (LDM Score: {ldm_score:.2f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect adversarial speech samples.")
    parser.add_argument("file", type=str, help="Path to the audio file to analyze.")
    args = parser.parse_args()

    detect_poisoned_sample(args.file)
