# Sinhala Handwriting Recognition

A **Computer Vision + Deep Learning project** for recognizing **Sinhala handwritten characters** using **TensorFlow/Keras** and **OpenCV**.

This project builds an **end-to-end pipeline** that processes handwritten Sinhala text images, segments characters, and classifies them using a **deep learning model trained on a custom dataset**.

The pipeline is designed to run on **Google Colab (GPU)** as well as **local CPU environments**.

---

# Project Overview

Sinhala handwriting recognition is a challenging computer vision task due to:

- Complex character shapes  
- Variations in writing styles  
- Noise in scanned or photographed images  

This project solves the problem by combining:

1. **Image preprocessing**
2. **Character segmentation**
3. **Deep learning classification**

The system converts raw handwritten images into predicted Sinhala characters through a **fully automated pipeline**.

---

# Pipeline Architecture

The recognition system follows this pipeline:

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


---

# Key Features

- Image preprocessing using **OpenCV**
- **Canny edge detection** for text extraction
- **Vertical projection segmentation** for splitting characters
- Standardized **128 × 128 input size**
- **Offline dataset preprocessing** for faster training
- **CNN / ConvNeXt-based classifier**
- Batch prediction support
- Compatible with **Google Colab GPU (T4)** and **local CPU**

---

# Technologies Used

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **Google Colab**

---

# Dataset Structure

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

# Project Structure

Sinhala-Handwriting-Recognition/

```
├── dataset/ # Raw dataset
├── dataset_processed/ # Preprocessed dataset
├── segmented_dataset/ # Character segmented images
│
├── models/
│ └── sinhala_model.keras
│
├── preprocessing.py # Image preprocessing
├── segmentation.py # Character segmentation
├── train.py # Model training
├── predict.py # Character prediction
│
├── train.ipynb # Training notebook
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Sinhala-Handwriting-Recognition.git
cd Sinhala-Handwriting-Recognition
```

Install dependencies:
```bash
pip install -r requirements.txt
```


## Future Improvements

Planned upgrades:

- Mobile deployment using TensorFlow Lite
- Improved segmentation for connected characters
- Support for complete Sinhala alphabet
- Data augmentation for larger datasets
- Real-time handwriting recognition

## Research & Educational Use

- This project was built as part of a machine learning research project for Sinhala handwriting recognition.
- It can also be adapted for:
- Other handwritten languages
- OCR research
- Computer vision learning projects
