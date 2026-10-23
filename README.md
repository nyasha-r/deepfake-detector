# 🛡️ Deepfake Detection System

An image-classification project that explores machine learning and frequency-domain analysis to distinguish between real and AI-generated images.

🚀 **Live Demo:** [Try the Deepfake Detection App](https://deepfake-detector-qrcu7unvntwgu5uc6ccqlq.streamlit.app/)

## 📌 Overview

This project implements a deepfake image detection application using image processing, frequency analysis, and machine learning. The Streamlit dashboard allows users to upload an image, visualize its frequency information, and view the model's prediction.

The repository also includes a separate deep-learning pipeline based on MobileNetV2 transfer learning.

Two approaches are explored:

- **FFT + Random Forest:** A frequency-based image-classification pipeline presented through an interactive Streamlit dashboard.
- **MobileNetV2:** A separate deep-learning pipeline for training and predicting real/fake image classes.

These are separate implementations in the repository, not a single integrated model.

## 🎯 Objectives

- Explore frequency-domain analysis for image classification.
- Extract image features and classify them using a Random Forest model.
- Develop and deploy an interactive Streamlit application.
- Experiment with transfer learning using MobileNetV2.
- Provide visual feedback through frequency analysis and prediction probabilities.

## 🧠 Approach 1: FFT-Based Detection

The deployed dashboard uses a frequency-based machine-learning pipeline.

### How it works

1. **Image upload:** Users upload a JPG or PNG image through the Streamlit sidebar.
2. **Feature extraction:** The image is processed using custom feature-extraction functions.
3. **Frequency analysis:** The Fast Fourier Transform (FFT) is used to generate a frequency visualization.
4. **Feature scaling:** Extracted features are transformed using a previously fitted scaler.
5. **Classification:** A Random Forest classifier generates real/fake prediction probabilities.
6. **Result visualization:** The application displays the original image, frequency visualization, prediction distribution, and final classification.

### Dashboard features

- Image upload and preview.
- FFT-based frequency visualization.
- Real and fake prediction probabilities.
- Prediction distribution chart.
- Final real/fake verdict.

### Files used

- `app.py` — Streamlit dashboard.
- `utils/features.py` — feature extraction.
- `utils/fft.py` — FFT visualization.
- `model/model.pkl` — trained Random Forest classifier expected by the application.
- `model/scaler.pkl` — fitted feature scaler expected by the application.

The classifier and scaler must be available at the specified paths for the dashboard to run.

## 🧠 Approach 2: MobileNetV2 Deep Learning

The repository also contains a separate deep-learning training and prediction pipeline.

### Model architecture

The training script uses MobileNetV2 pretrained on ImageNet, with a custom classification head consisting of:

- Global Average Pooling.
- Batch Normalization.
- Dense layers with 256 and 128 units.
- Dropout for regularization.
- A sigmoid output layer for binary classification.

### Training configuration

| Parameter | Configuration |
|---|---|
| Base model | MobileNetV2 |
| Input size | 224 × 224 pixels |
| Batch size | 32 |
| Maximum epochs | 15 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss function | Binary cross-entropy |
| Data augmentation | Rotation, zoom, shifts, horizontal flipping |
| Validation split | 20% |
| Callbacks | Early stopping and learning-rate reduction |

The training script saves the model as `deepfake_model.keras`.

### Prediction workflow

The separate `predict.py` script:

1. Loads the saved Keras model.
2. Checks for a face using OpenCV's Haar Cascade detector.
3. Resizes and normalizes the input image.
4. Generates a sigmoid prediction.
5. Applies a threshold to classify the image as real, fake, or uncertain.

**Note:** The real/fake label mapping must be verified against the training dataset's class indices before relying on the prediction labels.

## 🛠️ Tech Stack

- Python
- Streamlit
- OpenCV
- NumPy
- scikit-learn
- Joblib
- TensorFlow / Keras
- Pillow
- Fast Fourier Transform (FFT)
- Random Forest
- MobileNetV2

## 📂 Project Structure

```text
deepfake-detector/
├── app.py
├── train.py
├── predict.py
├── test_fft.py
├── requirements.txt
├── assets/
├── data/
│   └── train/
├── model/
│   ├── model.pkl
│   └── scaler.pkl
├── utils/
│   ├── features.py
│   └── fft.py
├── deepfake_model.keras
├── model.keras
├── model_efficientnet.keras
├── model_face_final.keras
├── model_final.keras
├── model_finetuned.keras
├── icon.png
├── test.jpg
└── README.md
```

This is a representative structure. Refer to the repository for the exact files and directory contents.

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/nyasha-r/deepfake-detector.git
cd deepfake-detector
```

If your repository uses a different name or URL, replace the example accordingly.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For the MobileNetV2 pipeline, TensorFlow must also be installed and included in `requirements.txt`.

### 3. Run the Streamlit dashboard

Ensure the Random Forest model, scaler, and utility files are available.

```bash
streamlit run app.py
```

Open the local URL shown in your terminal to use the application.

### 4. Train the MobileNetV2 model

Prepare the dataset in the directory structure expected by `train.py`, then run:

```bash
python train.py
```

The script saves the trained model as `deepfake_model.keras`.

### 5. Run MobileNetV2 prediction

Place a test image at the path specified in `predict.py`, then run:

```bash
python predict.py
```

## 📊 Model Performance

The Streamlit application displays an approximate accuracy figure of **71%**. This figure is stated in the application code and has not been independently verified from the supplied files.

The MobileNetV2 training script does not report a final held-out test-set evaluation. Its performance should therefore be measured before making comparative accuracy claims.

Recommended evaluation metrics include:

- Accuracy.
- Precision and recall.
- F1-score.
- Confusion matrix.
- Performance on unseen images.

## ⚠️ Limitations

- Predictions are probabilistic and may be incorrect.
- Results depend on the quality and diversity of the training data.
- The FFT + Random Forest dashboard and MobileNetV2 pipeline are separate implementations.
- The MobileNetV2 prediction script requires a detectable face.
- Training-label mappings should be verified before interpreting predictions.
- The system should not be treated as definitive proof that an image is authentic or manipulated.

## 🔬 Future Improvements

- Evaluate both approaches on a common test dataset.
- Verify and document class-label mappings.
- Improve model evaluation and reporting.
- Test robustness against image resizing and JPEG compression.
- Improve error handling and prediction reliability.
- Explore integrating both approaches into a unified application.

## 🎓 Applications

- AI-generated image analysis.
- Computer vision experimentation.
- Media authenticity research.
- Academic machine-learning projects.
- Digital-content integrity research.

## 👩‍💻 Author

**Nyasha Singh**

Field: Artificial Intelligence / Machine Learning

GitHub: [nyasha-r](https://github.com/nyasha-r)

---

*This project was developed for educational and research purposes. Predictions should be interpreted with appropriate caution.*
