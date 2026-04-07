import numpy as np
import cv2
from utils.fft import compute_fft

def extract_features(image):
    image = cv2.resize(image, (128, 128))
    fft_img = compute_fft(image)

    # Normalize
    fft_norm = fft_img / np.max(fft_img)

    h, w = fft_norm.shape
    center_h, center_w = h // 2, w // 2

    # Regions
    low = fft_norm[center_h-30:center_h+30, center_w-30:center_w+30]
    mid = fft_norm[center_h-80:center_h+80, center_w-80:center_w+80]
    high = np.copy(fft_norm)
    high[center_h-80:center_h+80, center_w-80:center_w+80] = 0

    # Features
    features = []

    # Basic stats
    features.append(np.mean(fft_norm))
    features.append(np.std(fft_norm))

    # Region energies
    features.append(np.mean(low))
    features.append(np.mean(mid))
    features.append(np.mean(high))

    # Ratios
    features.append(np.mean(low) / (np.mean(high) + 1e-6))
    features.append(np.mean(mid) / (np.mean(high) + 1e-6))

    # Additional texture-like info
    features.append(np.var(fft_norm))
    features.append(np.max(fft_norm))

    return features