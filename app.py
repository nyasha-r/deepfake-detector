import streamlit as st
import cv2
import numpy as np
import joblib

from utils.features import extract_features
from utils.fft import compute_fft

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🛡️",
    layout="wide"
)

# ===== HEADER =====
st.title("🛡️ Deepfake Detection Dashboard")
st.caption("Detect AI-generated images using Frequency Domain Analysis")

st.divider()

# ===== SIDEBAR =====
st.sidebar.title("📌 Controls")
uploaded_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png"])

st.sidebar.markdown("---")
st.sidebar.info("Model: Random Forest | Stable Version")

# ===== LOAD MODEL =====
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")

# ===== MAIN =====
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns(2)

    # ORIGINAL IMAGE
    with col1:
        st.subheader("📷 Original Image")
        st.image(img, use_column_width=True)

    # ANALYSIS
    with st.spinner("🔍 Analyzing image..."):
        features = extract_features(img)
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)

        confidence = model.predict_proba(features)[0]
        fake_prob = confidence[1]

    # FFT IMAGE
    fft_img = compute_fft(img)

    with col2:
        st.subheader("🌀 Frequency Analysis")
        st.image(fft_img, clamp=True, use_column_width=True)

    st.divider()

    # METRICS
    col1, col2, col3 = st.columns(3)
    col1.metric("Real %", f"{confidence[0]*100:.2f}")
    col2.metric("Fake %", f"{confidence[1]*100:.2f}")
    col3.metric("Confidence", f"{max(confidence)*100:.2f}")

    # BAR CHART
    st.subheader("📊 Prediction Distribution")
    st.bar_chart({
        "Real": confidence[0],
        "Fake": confidence[1]
    })

    # FINAL RESULT
    st.subheader("🧾 Final Verdict")

    if fake_prob > 0.6:
        st.error("🚨 Fake Image Detected")
    else:
        st.success("✅ Real Image")

    st.caption("⚠️ Model is probabilistic (~71% accuracy)")

else:
    st.info("👈 Upload an image from the sidebar to begin")