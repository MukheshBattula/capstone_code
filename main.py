
import os
import pickle
from preprocessing import load_audio, extract_mfcc
from adversarial_detection import compute_ldm
from logger import log_detection


CLEAN_STATS_FILE = "models/clean_data_stats.pkl"

if not os.path.exists(CLEAN_STATS_FILE):
    print("Error: Clean dataset statistics file not found!")
    exit()

with open(CLEAN_STATS_FILE, "rb") as f:
    clean_stats = pickle.load(f)


AUDIO_FILE = "data/sample.mp3"


audio, sr = load_audio(AUDIO_FILE)
mfcc_features = extract_mfcc(audio, sr)

# Compute LDM
status = compute_ldm(mfcc_features.mean(axis=1), clean_stats)

# Log results
log_detection(AUDIO_FILE, status)
print(f"Detection Result: {status}")
