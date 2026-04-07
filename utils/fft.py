import cv2
import numpy as np

def compute_fft(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    # Magnitude spectrum
    magnitude = np.abs(fshift)
    magnitude = np.log(magnitude + 1)

    # Normalize to 0–255
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to uint8 (THIS WAS MISSING 🔥)
    magnitude = magnitude.astype(np.uint8)

    return magnitude