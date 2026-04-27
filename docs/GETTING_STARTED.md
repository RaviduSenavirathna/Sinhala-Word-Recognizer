# Getting Started with Sinhala Word Recognizer

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.7 or higher
- pip (Python package manager)
- Git
- A GPU (optional, but recommended for faster training)

## Quick Start

### Step 1: Clone the Repository

```bash
git clone https://github.com/RaviduSenavirathna/Sinhala-Word-Recognizer.git
cd Sinhala-Word-Recognizer
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Your Dataset
Your dataset should be organized in the following structure:

```code
dataset/
├── ක/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── ග/
│   ├── image1.png
│   └── ...
├── ත/
│   └── ...
└── ... (other Sinhala characters)
```

Each folder represents a Sinhala character class, and contains handwritten samples of that character.

### Step 4: Run the Pipeline

#### Option A: Google Colab (Recommended for Beginners)
- Open Google Colab: https://colab.research.google.com/
- Upload the notebooks to Colab
- Follow the cells in order and execute them
- Colab provides free GPU access (T4)


#### Option B: Local Environment
Follow this order:
1. Dataset Preparation - `Run dataset.ipynb`
    - Opens your raw dataset and prepares it for processing

2. Character Segmentation - `Run word_segmenting.ipynb`
    - Segments words into individual characters

3. Model Training - `Run model_training.ipynb`
    - Trains the CNN model on your preprocessed dataset

4. Prediction - `Run predict.ipynb`
    - Test your trained model on new images



## Project Structure Explained
```Code
Sinhala-Word-Recognizer/
├── docs/                      # Documentation files
│   ├── GETTING_STARTED.md     # This file
│   ├── API_REFERENCE.md       # Function documentation
│   └── TROUBLESHOOTING.md     # Common issues & solutions
│
├── dataset/                   # Raw input images
├── dataset_processed/         # Preprocessed images
├── segmented_dataset/         # Character-segmented images
│
├── dataset.ipynb              # Load & prepare dataset
├── word_segmenting.ipynb      # Character segmentation
├── model_training.ipynb       # CNN model training
├── predict.ipynb              # Inference & predictions
│
├── pre_processing.py          # Image preprocessing utilities
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview
└── LICENSE  
```

## Workflow Overview
### 1. Data Preparation Phase
- Load images from dataset folder
- Apply preprocessing (grayscale, blur, threshold)
- Save processed images
### 2. Segmentation Phase
- Take preprocessed images of words
- Segment into individual characters using vertical projection
- Standardize each character to 128×128 pixels
- Save segmented characters
### 3. Training Phase
- Load segmented character dataset
- Build CNN/ConvNeXt model
- Train on GPU (Colab recommended)
- Save trained model weights
### 4. Inference Phase
- Load trained model
- Take test image as input
- Preprocess → Segment → Predict
- Output recognized characters