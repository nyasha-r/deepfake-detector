import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import cv2

# ======================
# LOAD MODEL
# ======================
model = keras.models.load_model("deepfake_model.keras", compile=False)

IMG_SIZE = 224
THRESHOLD = 0.65

# ======================
# FACE DETECTOR
# ======================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_face(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces)


# ======================
# PREDICT FUNCTION
# ======================
def predict_image(img_path):

    if not os.path.exists(img_path):
        return "❌ Image not found"

    # 🔍 Face check
    face_count = detect_face(img_path)

    if face_count == 0:
        return "⚠️ No face detected"

    # Load image
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]
    confidence = prediction if prediction > 0.5 else (1 - prediction)

    # Decision
    if prediction > THRESHOLD:
        return f"✅ REAL ({confidence:.2f})"
    elif prediction < (1 - THRESHOLD):
        return f"❌ FAKE ({confidence:.2f})"
    else:
        return f"⚠️ UNCERTAIN ({confidence:.2f})"


# ======================
# TEST
# ======================
if __name__ == "__main__":
    result = predict_image("test.jpg")
    print(result)