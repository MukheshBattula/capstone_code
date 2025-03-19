import os
import pickle
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Example model
from sklearn.svm import SVC  # Support Vector Machine
from sklearn.metrics import accuracy_score

with open("../models/clean_data_stats.pkl", "rb") as f:
    clean_stats = pickle.load(f)

DATA_DIR = "../data/clean_speech/"
MODEL_PATH = "../models/audio_classifier.pkl"

def extract_mfcc(file_path, target_sr=16000):
    """Extracts and normalizes MFCC features from an audio file."""
    y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Normalize using clean statistics
    normalized_mfcc = (mfcc.T - clean_stats["mean"]) / clean_stats["std"]
    return normalized_mfcc.T  # Convert back to (13, time-steps)

# Load training data
X, y = [], []
for file in os.listdir(DATA_DIR):
    if file.endswith(".wav"):
        file_path = os.path.join(DATA_DIR, file)
        mfcc_features = extract_mfcc(file_path)

        feature_vector = mfcc_features.mean(axis=1)  # Take mean across time
        X.append(feature_vector)

        label = 0 if "clean" in file else 1  # 0 = Clean, 1 = Poisoned
        y.append(label)

X = np.array(X)
y = np.array(y)

# Split data into training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classifier (RandomForest or SVM)
model = RandomForestClassifier(n_estimators=100, random_state=42)
# model = SVC(kernel="linear", probability=True)  # Use SVM instead if needed
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained with accuracy: {accuracy:.2f}")

# Save trained model
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
print(f"🎯 Model saved at {MODEL_PATH}")
