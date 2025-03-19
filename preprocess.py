import os
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import pickle
from tqdm import tqdm

# Paths
RAW_DATA_DIR = "../data/clips"  
CLEAN_SPEECH_DIR = "../data/clean_speech/"
STATS_FILE = "../models/clean_data_stats.pkl"

# Ensure output directories exist
os.makedirs(CLEAN_SPEECH_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

def process_audio(file_path, target_sr=16000):
    """Loads an audio file, converts to mono, and extracts MFCC features."""
    try:
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return y, sr, mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None

def preprocess_and_extract():
    """Converts MP3 to WAV, extracts MFCC features, and saves statistics."""
    all_mfccs = []

    for file in tqdm(os.listdir(RAW_DATA_DIR)):
        if file.endswith(".mp3"):
            file_path = os.path.join(RAW_DATA_DIR, file)
            y, sr, mfcc = process_audio(file_path)

            if mfcc is not None:
                wav_path = os.path.join(CLEAN_SPEECH_DIR, file.replace(".mp3", ".wav"))
                sf.write(wav_path, y, sr)

                all_mfccs.append(mfcc.mean(axis=1))  # Mean of MFCCs over time

    all_mfccs = np.array(all_mfccs)
    clean_stats = {
        "mean": np.mean(all_mfccs, axis=0),
        "std": np.std(all_mfccs, axis=0)
    }

    with open(STATS_FILE, "wb") as f:
        pickle.dump(clean_stats, f)

    print("✅ Preprocessing complete! Clean data stats saved.")

if __name__ == "__main__":
    preprocess_and_extract()
