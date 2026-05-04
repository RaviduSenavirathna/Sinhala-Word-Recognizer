# Step-by-Step Tutorial

Complete walkthrough of the Sinhala Word Recognizer pipeline with examples and explanations.

## Table of Contents

1. [Preparing Your Dataset](#preparing-your-dataset)
2. [Image Preprocessing](#image-preprocessing)
3. [Character Segmentation](#character-segmentation)
4. [Model Training](#model-training)
5. [Making Predictions](#making-predictions)
6. [Tips & Optimization](#tips--optimization)

---

## Preparing Your Dataset

### Understanding the Dataset Structure

Your dataset needs to follow a specific folder structure:
```
dataset/ 
├── ක/ # First Sinhala character 
│   ├── 001.png 
│   ├── 002.png 
│   ├── 003.png 
│   └── ... 
├── ග/ # Second character 
│   ├── 001.png 
│   ├── 002.png 
│   └── ... 
├── ත/ 
├── න/ 
├── ප/ 
├── ම/ 
├── ර/ 
├── ල/ 
├── ස/ 
└── හ/ # Last character 
└── ...
```


**Key Points:**
- Each folder name is the Sinhala character itself
- Each folder contains images of that character
- Images can be PNG, JPG, or other common formats
- More samples = better model accuracy

### Recommended Dataset Guidelines

| Aspect | Recommendation |
|--------|-----------------|
| Samples per character | 100-500 minimum |
| Image quality | Clear, scanned/photographed handwriting |
| Image size | 400×400 or larger |
| Balance | Similar count across all characters |
| File format | PNG or JPG |

### Loading Your Dataset

**Step 1:** Upload to Google Colab (if using Colab)

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy dataset to Colab
import shutil
shutil.copytree('/content/drive/MyDrive/dataset', '/content/dataset')
```

**Step 2:** Verify dataset in dataset.ipynb
```python
import os

dataset_path = 'dataset'
classes = os.listdir(dataset_path)

print(f"Total characters: {len(classes)}")
print(f"Characters: {sorted(classes)}\n")

# Count samples per character
for char in sorted(classes):
    count = len(os.listdir(os.path.join(dataset_path, char)))
    print(f"{char}: {count} samples")
```

**Expected Output:**
```
Total characters: 10
Characters: ['ක', 'ග', 'ත', 'න', 'ප', 'ම', 'ර', 'ල', 'ස', 'හ']

ක: 150 samples
ග: 145 samples
ත: 152 samples
...
```

# Image Preprocessing

## Why Preprocessing Matters

Raw images have:
- Noise: Dust, scanning artifacts
- Variations: Brightness, contrast differences
- Complexity: Color information not needed for shapes

**Preprocessing converts raw images into clean binary images** suitable for character recognition.

### The Preprocessing Pipeline
```
Raw Image (BGR, colored, noisy)
    ↓
1. Convert to Grayscale (simplify to single channel)
    ↓
2. Apply Gaussian Blur (reduce noise)
    ↓
3. Apply Otsu Threshold (convert to pure black & white)
    ↓
Processed Image (binary, clean)
```

### Step-by-Step Preprocessing
Open `dataset.ipynb` and follow this workflow:

**Step 1: Load and Preview Raw Image**

```python
import cv2
import matplotlib.pyplot as plt
from pre_processing import processed_img_preview

# View a sample raw image
img_path = 'dataset/ක/001.png'
img = cv2.imread(img_path)

plt.figure(figsize=(4, 4))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Raw Image")
plt.axis('off')
plt.show()
```

**Step 2: Preprocess Single Image**

```python
from pre_processing import img_processing, tight_crop

# Preprocess
processed = img_processing(img)

# Crop excess white space
cropped = tight_crop(processed)

# Display result
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cropped, cmap='gray')
plt.title("Processed")
plt.axis('off')
plt.show()
```

**Step 3: Batch Process All Dataset Images**

```python
import os
import numpy as np
from pathlib import Path

source_dir = 'dataset'
target_dir = 'dataset_processed'

# Create output directory
Path(target_dir).mkdir(exist_ok=True)

# Process all images
for character in os.listdir(source_dir):
    char_source = os.path.join(source_dir, character)
    char_target = os.path.join(target_dir, character)
    
    # Create character folder
    Path(char_target).mkdir(exist_ok=True)
    
    # Process each image
    for img_file in os.listdir(char_source):
        try:
            img_path = os.path.join(char_source, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Skipped: {img_path} (read error)")
                continue
            
            # Preprocess
            processed = img_processing(img)
            
            # Save
            output_path = os.path.join(char_target, img_file)
            cv2.imwrite(output_path, processed)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

print("✓ Preprocessing complete! Saved to dataset_processed/")
```



### Understanding Each Processing Step

1. Grayscale Conversion

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```
- Converts 3-channel (BGR) to 1-channel (grayscale)
- Keeps only brightness information
- Reduces data size by 3×

2. Gaussian Blur

```python
blur = cv2.GaussianBlur(gray, (3, 3), 0)
```
- Smooths image to reduce noise
- Kernel size (3, 3) = small neighborhood
- Sigma = 0 = auto-calculated
- Effect: Fuzzy the image slightly

3. Otsu Threshold

```python
_, thresh = cv2.threshold(
    blur, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)
```
- Converts grayscale → pure black/white
- `THRESH_OTSU`: Automatically finds best threshold value
- `THRESH_BINARY_INV`: Inverts (makes text white, background black)
- Result: Clean binary image perfect for segmentation



### Preprocessing Results
You should see:
- Before: Noisy, colored, variable contrast
- After: Clean, binary (only 0 and 255 values), consistent

