# 🤟 SignSense — Indian Sign Language Recognition with CNN

> A convolutional neural network that classifies Indian Sign Language (ISL) hand gestures across 35 classes using grayscale image recognition.

---

## 📌 Overview

**SignSense** is a deep learning project that trains a CNN to recognize hand gestures from the **Indian Sign Language (ISL)** dataset. Given an input image of a hand sign, the model predicts which of the 35 ISL gesture classes it belongs to — covering letters and common symbols used in ISL.

The model is built with TensorFlow/Keras, uses data augmentation to improve generalization, and employs early stopping to prevent overfitting.

---

## 📂 Dataset

**Indian Sign Language (ISL) Dataset** — sourced from Kaggle

| Property | Details |
|---|---|
| Source | [Kaggle — Indian Sign Language (ISL)](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl) |
| Classes | 35 (letters and symbols used in ISL) |
| Image Size | 64 × 64 pixels (grayscale) |
| Format | Folder-per-class structure |

Download and extract the dataset, then update `dataset_path` in the notebook to point to your local directory.

---

## 🏗️ Model Architecture

A three-block CNN followed by a fully connected head:

```
Input (64 × 64 × 1 grayscale)
        │
        ▼
Conv2D(32 filters, 3×3, ReLU)
MaxPooling2D(2×2)
        │
        ▼
Conv2D(64 filters, 3×3, ReLU)
MaxPooling2D(2×2)
        │
        ▼
Conv2D(128 filters, 3×3, ReLU)
MaxPooling2D(2×2)
        │
        ▼
Flatten
Dense(128, ReLU)
Dropout(0.5)
        │
        ▼
Dense(35, Softmax)  ← Output: 35 ISL classes
```

**Optimizer:** Adam  
**Loss:** Categorical Crossentropy  
**Metric:** Accuracy

---

## ⚙️ Pipeline

```
Raw Images (folder-per-class)
        │
        ▼
Load & Resize to 64×64 grayscale
Normalize pixel values to [0, 1]
        │
        ▼
Label Encoding → One-Hot Encoding
80/20 Train-Test Split
        │
        ▼
Data Augmentation (Training only)
  ├── Rotation ±10°
  ├── Zoom ±10%
  ├── Width shift ±10%
  └── Height shift ±10%
        │
        ▼
CNN Training (up to 15 epochs)
EarlyStopping on val_loss (patience=3)
        │
        ▼
Evaluation on Test Set
Model saved as .h5
```

---

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.8
tensorflow >= 2.x
scikit-learn
numpy
matplotlib
```

Install dependencies:

```bash
pip install tensorflow scikit-learn numpy matplotlib
```

### Running the Notebook

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl) and extract it locally.
2. Open `AI_Project.ipynb` in Jupyter or Google Colab.
3. Update the dataset path in the preprocessing cell:
   ```python
   dataset_path = '/path/to/your/Indian/'
   ```
4. Run all cells top to bottom.

The trained model will be saved as `hand_sign_recognition_model.h5`.

---

## 📊 Training Details

| Parameter | Value |
|---|---|
| Input shape | 64 × 64 × 1 |
| Epochs | Up to 15 (early stopping) |
| Batch size | 32 |
| Optimizer | Adam |
| Loss function | Categorical Crossentropy |
| Early stopping patience | 3 (monitors `val_loss`) |
| Train/Test split | 80% / 20% |
| Dropout rate | 0.5 |

---

## 📁 Project Structure

```
.
├── AI_Project.ipynb                  # Full pipeline: preprocessing → training → evaluation
├── hand_sign_recognition_model.h5    # Saved model (generated after training)
└── README.md
```

---

## 🗺️ Roadmap / Future Work

- [ ] Real-time recognition via webcam using OpenCV
- [ ] Expand to full ISL vocabulary (beyond 35 classes)
- [ ] Deploy as a web or mobile app for accessibility use cases
- [ ] Experiment with transfer learning (MobileNet, EfficientNet) for higher accuracy
- [ ] Add a sequence model (LSTM) for recognizing multi-sign words

---

## 👤 Author

**Aarush Tiwari**  
Bennett University, Greater Noida

---

## 📄 License

This project is for academic and educational purposes.
