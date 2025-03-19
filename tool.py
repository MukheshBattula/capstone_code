import librosa
import numpy as np
import pickle

STATS_FILE = "../models/clean_data_stats.pkl"

def extract_mfcc(file_path, target_sr=16000):
    """Extracts MFCC features from an audio file."""
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.mean(axis=1)  

def detect_poisoned_audio(file_path):
    """Detects if an audio file is poisoned by comparing MFCC features."""
    
    
    with open(STATS_FILE, "rb") as f:
        clean_stats = pickle.load(f)
    
    
    mfcc_test = extract_mfcc(file_path)

    
    deviation = np.abs((mfcc_test - clean_stats["mean"]) / clean_stats["std"])

    # If deviation is too high, classify as poisoned
    threshold = 3.0  # Adjust this threshold if needed
    poisoned = np.any(deviation > threshold)

    if poisoned:
        print(f"⚠️ {file_path} is POISONED!")
    else:
        print(f"✅ {file_path} is CLEAN.")

# Test on an audio file

# we can add a mp3 file here and run the code
detect_poisoned_audio("../data/test_audio.mp3")
