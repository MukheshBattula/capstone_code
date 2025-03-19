import pickle
import numpy as np
import librosa

# Load saved statistics
with open("../models/clean_data_stats.pkl", "rb") as f:
    clean_stats = pickle.load(f)

# Extract and normalize MFCC features
def extract_mfcc(file_path, target_sr=16000):
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Normalize using clean speech statistics
    normalized_mfcc = (mfcc.T - clean_stats["mean"]) / clean_stats["std"]
    normalized_mfcc = normalized_mfcc.T  # Convert back to (13, time-steps)
    
    return normalized_mfcc

# Test with a sample file
mfcc_example = extract_mfcc("../data/clean_speech/common_voice_en_41236242.wav")
print(mfcc_example.shape)  # Expected: (13, time-steps)
