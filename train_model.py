import os
import cv2
import numpy as np
from utils.features import extract_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

REAL_PATH = "data/real"
FAKE_PATH = "data/fake"

X = []
y = []

# Real images
for file in os.listdir(REAL_PATH):
    img = cv2.imread(os.path.join(REAL_PATH, file))
    if img is not None:
        X.append(extract_features(img))
        y.append(0)

# Fake images
for file in os.listdir(FAKE_PATH):
    img = cv2.imread(os.path.join(FAKE_PATH, file))
    if img is not None:
        X.append(extract_features(img))
        y.append(1)

X = np.array(X)
y = np.array(y)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=150)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("✅ Model saved!")