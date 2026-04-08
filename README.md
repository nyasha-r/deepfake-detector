# 🧠 Deepfake Detection System

A machine learning web application that detects whether an image is real or AI-generated using frequency-domain analysis (FFT).

---

## 🚀 Features

- 🌀 FFT-based feature extraction
- 🤖 Random Forest classifier
- 📊 Real-time prediction with confidence score
- 🌐 Interactive Streamlit dashboard
- 📷 Image upload support

---

## 🧠 Tech Stack

- Python
- OpenCV
- NumPy
- Scikit-learn
- Streamlit

---

## 📈 Model Performance

- Accuracy: ~71%
- Model: Random Forest
- Feature Type: Frequency-domain (FFT)

---

## 🖥️ Demo

Upload an image and get:
- Prediction (Real / Fake)
- Confidence score
- FFT visualization

---

## ⚙️ How to Run Locally

```bash
git clone https://github.com/nyasha-r/deepfake-detector.git
cd deepfake-detector
pip install -r requirements.txt
streamlit run app.py