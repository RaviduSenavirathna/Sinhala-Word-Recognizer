# Sinhala Handwriting Recognition

A **Computer Vision + Deep Learning project** for recognizing **Sinhala handwritten characters** using **TensorFlow/Keras** and **OpenCV**.

This project builds an **end-to-end pipeline** that processes handwritten Sinhala text images, segments characters, and classifies them using a **deep learning model trained on a custom dataset**.

The pipeline is designed to run on **Google Colab (GPU)** as well as **local CPU environments**.

### 🎯 Key Features

- 🖼️ Image preprocessing (grayscale, blur, thresholding)
- ✂️ Character segmentation (projection-based)
- 🧠 CNN-based classification
- 📏 Standardized 128×128 input pipeline
- ⚡ Batch prediction support
- 🧩 Modular & notebook-friendly structure
- 📦 Ready for Colab training + deployment

---

## 📌 Project Overview

Sinhala handwriting recognition is a challenging computer vision task due to:

- Complex character shapes  
- Variations in writing styles  
- Noise in scanned or photographed images  

This project solves the problem by combining:

1. **Image preprocessing**
2. **Character segmentation**
3. **Deep learning classification**

The system converts raw handwritten images into predicted Sinhala characters through a **fully automated pipeline**.

### Pipeline Architecture

The recognition system follows this pipeline:
```
Raw Image
↓
Image Preprocessing (Grayscale + Edge Detection)
↓
Character Segmentation (Vertical Projection)
↓
Resize & Normalize (128×128)
↓
Deep Learning Model (CNN / ConvNeXt)
↓
Character Prediction
```

---


## Dataset Structure

The dataset must be organized by **class folders**:
```
dataset/
├── ක/
│ ├── img1.png
│ ├── img2.png
│
├── ග/
│ ├── img1.png
│
├── ත/
├── න/
├── ප/
├── ම/
├── ර/
├── ල/
├── ස/
└── හ/
```

Each folder contains images of a **single Sinhala character**.

---

## Project Structure

```
Sinhala-Handwriting-Recognition/
├── dataset/ # Raw dataset
├── dataset_processed/       # Preprocessed dataset
├── segmented_dataset/       # Character segmented images
│
├── models/
│ └── sinhala_model.keras
│
├── preprocessing.py         # Image preprocessing
├── segmentation.py          # Character segmentation
├── train.py                 # Model training
├── predict.py               # Character prediction
│
├── train.ipynb              # Training notebook
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Sinhala-Word-Recognizer.git
cd Sinhala-Word-Recognizer
```

Install dependencies:
```bash
pip install -r requirements.txt
```
---

### 🖼️ Preprocessing Pipeline

The preprocessing stage converts raw images into clean binary inputs for segmentation and classification.

Steps:
- Convert to grayscale
- Apply Gaussian blur
- Apply Otsu thresholding
- Crop tightly around characters

Example from pre_processing.py:
```
def img_processing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return thresh
```

### ✂️ Segmentation

- Uses vertical projection to split characters
- Handles multi-character Sinhala words
- Designed to minimize over-segmentation errors

### 🧠 Model Architecture

- CNN-based classifier
- Input size: 128 × 128
- Built using TensorFlow/Keras
- Supports GPU training (Colab T4 optimized)

---

### 🔧 Customization

- You can easily adapt this project for:
- Other languages (Tamil, English, etc.)
- Different input sizes
- Mobile deployment (TFLite)
- Real-time recognition systems

### 📈 Future Improvements

- 🔍 Attention-based sequence models (CRNN / Transformer)
- 📱 Mobile deployment (TFLite / ONNX)
- 🌐 Web API (FastAPI / Flask)
- 🎯 Better segmentation (connected components + ML)
- 📚 Dataset expansion & augmentation

---

## 🤝 Contributing

Pull requests are welcome!
If you’d like to improve segmentation, model accuracy, or deployment — feel free to contribute.

#### 📄 License
This project is licensed under the MIT License.

#### 👤 Author
Ravidu Pasan Senavirathna
GitHub: https://github.com/RaviduSenavirathna


### ⭐ Support
If you like this project:
- ⭐ Star the repo
- 🍴 Fork it
- 📢 Share it
