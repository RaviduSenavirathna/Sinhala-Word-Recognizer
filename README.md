# Sinhala Handwriting Recognition 📝🇱🇰

A computer vision and deep learning project for **Sinhala handwritten character recognition**, covering **image preprocessing, segmentation, and classification** using **TensorFlow/Keras** and **OpenCV**.

---

## 📌 Project Overview

This project implements an end-to-end pipeline for recognizing handwritten Sinhala characters.  
It processes raw handwritten images, segments individual characters, and classifies them using a CNN-based deep learning model.

### Key Objectives
- Preprocess handwritten Sinhala text images
- Segment words and individual characters
- Classify Sinhala characters using deep learning
- Support lightweight and mobile-friendly models (TFLite-ready)

---

## 🧠 Features

- ✅ Grayscale and Canny-based preprocessing  
- ✅ Horizontal projection–based segmentation  
- ✅ CNN-based character classification  
- ✅ Standardized **128×128** input pipeline  
- ✅ Batch prediction with preview saving  
- ✅ Modular, reusable, and notebook-friendly code  

---

## 🗂️ Project Structure

```text
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
├── segmented_letters/
├── models/
│   └── sinhala_model.keras
├── notebooks/
│   ├── train.ipynb
│   ├── segmentation.ipynb
│   └── inference.ipynb
├── utils/
│   ├── preprocessing.py
│   ├── segmentation.py
│   └── visualization.py
├── requirements.txt
└── README.md

---